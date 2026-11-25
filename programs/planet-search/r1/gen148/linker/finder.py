"""TNO finder, generation 1.  See NOTES.md.

1. Detection (detect.py) in 4 worker processes: image + mask only, background, PSF matched filter S/N map,
   local maxima >= 3 sigma with sub-pixel peak, sharpness and flux.  Exposures processed night-round-robin so a
   partial run still covers every night; stops at a time budget.
2. Stationary sources: detection matched on >= 2 other nights within 1.5", or on 1 other night within 0.7".
3. Linking (tracklet-less HelioLinC in the Bernstein-Khushalani frame, as in reference/gpu_linker): for each
   assumed inverse distance gamma, seeds (S/N >= ~5, sane sharpness, non-stationary) are paired across nights,
   pairs turned into (alpha, beta, alpha-dot, beta-dot) and hashed; cells with >= 3 nights are verified by a line fit
   and extended to every exposure with the full >= 3 sigma catalogue.
4. Each linkage gets a full two-body + light-time least-squares orbit fit against the RA/Dec of its detections;
   score from summed S/N over covered exposures, nights, residuals -> p.
"""
import argparse, csv, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import orbit
from orbit import T0, log, left


# ----------------------------------------------------------------------------------------------- detection stage
def n_workers():
    """Detection is CPU-bound (FITS decode, WCS, block medians); the GPU part is small.  On the CUDA grader (8 CPU
    threads) use 7 workers; elsewhere (dev Mac: <= 4 threads allowed) 4.  TNO_WORKERS overrides."""
    if os.environ.get("TNO_WORKERS"):
        return max(1, int(os.environ["TNO_WORKERS"]))
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:
        ncpu = os.cpu_count() or 4
    cuda = False
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    if cuda:
        return int(max(2, min(7, ncpu - 1)))
    return int(max(1, min(4, ncpu)))


def run_detection(rows, sdir, budget, frac, nworkers=None):
    import detect
    if nworkers is None:
        nworkers = n_workers()
    log("detection workers:", nworkers)
    from multiprocessing import get_context
    n = len(rows)
    t = np.array([float(r["mjd_mid"]) for r in rows])
    night = np.floor(t - 0.5).astype(int)          # local night (Chile, UTC-4..5: mjd frac ~0.0-0.45)
    # night-round-robin order
    un = np.unique(night)
    per = [list(np.flatnonzero(night == u_)) for u_ in un]
    for p in per:
        p.sort(key=lambda i: t[i])
    order = []
    k = 0
    while len(order) < n:
        for p in per:
            if k < len(p):
                # chronological within the night: consecutive exposures pair up into same-night seeds
                j = p[k]
                if j not in order:
                    order.append(j)
        k += 1
        if k > 2 * n + 5:
            break
    order += [i for i in range(n) if i not in order]
    nmap = 40           # S/N maps (2x decimated, float16, ~250 MB each) kept for forced photometry
    tasks = [(int(i), os.path.join(sdir, rows[i]["image"]), os.path.join(sdir, rows[i]["mask"]), 3.0, k < nmap)
             for k, i in enumerate(order)]
    deadline = budget * frac
    results = {}
    ctx = get_context("spawn")
    pool = ctx.Pool(nworkers)
    inflight = []
    it = iter(tasks)
    durs = []
    done_sub = False
    # gen 148: a task whose worker died (e.g. host OOM kill) never becomes ready, and multiprocessing.Pool replaces
    # the worker but not the task, so the slot was lost until the hard stop.  A task running far longer than usual is
    # moved to `stale` (still collected if it ever finishes) and its slot is reused.
    stale = []
    try:
        while True:
            # submit
            while not done_sub and len(inflight) < nworkers:
                est = np.median(durs) if durs else 25.0
                if time.time() - T0 + est > deadline and results:
                    done_sub = True
                    break
                try:
                    tk = next(it)
                except StopIteration:
                    done_sub = True
                    break
                inflight.append((pool.apply_async(detect.detect_exposure, (tk,)), time.time()))
            if not inflight:
                break
            still = []
            for ar, ts in inflight:
                if ar.ready():
                    try:
                        o = ar.get()
                        results[o["ei"]] = o
                        durs.append(time.time() - ts)
                        log(f"exp {o['ei']}: {len(o['det'])} det fwhm {o['fwhm']:.2f} ({time.time() - ts:.1f}s) err={o['err']}")
                    except Exception as ex:
                        log("worker error", ex)
                else:
                    still.append((ar, ts))
            inflight = still
            lim_ = max(240.0, 8.0 * float(np.median(durs))) if durs else 600.0
            if any(time.time() - ts > lim_ for _, ts in inflight):
                stale += [(ar, ts) for ar, ts in inflight if time.time() - ts > lim_]
                inflight = [(ar, ts) for ar, ts in inflight if time.time() - ts <= lim_]
                log(f"detection: {len(stale)} task(s) over {lim_:.0f}s marked stale; their slots are reused")
            for ar, ts in list(stale):
                if ar.ready():
                    stale.remove((ar, ts))
                    try:
                        o = ar.get()
                        results[o["ei"]] = o
                        log(f"exp {o['ei']}: {len(o['det'])} det (stale task finished after {time.time() - ts:.1f}s)")
                    except Exception as ex:
                        log("worker error", ex)
            if not inflight and done_sub:
                break
            if time.time() - T0 > deadline + 40:      # hard stop
                log("detection hard stop")
                break
            time.sleep(0.05)
    finally:
        pool.terminate()
    return results


def valid_state(c, GM=2.959122082855911e-4):
    """gen 4b (from gen 3b): the grader drops unbound / < 25 au states; drop them before the top-200 cut."""
    try:
        s = [float(v) for v in c["state"]]
        if len(s) != 6 or not all(map(math.isfinite, s + [float(c["epoch_mjd"]), float(c["p"])])):
            return False
        r = math.sqrt(sum(v * v for v in s[:3]))
        return r >= 25.0 and sum(v * v for v in s[3:]) / 2 - GM / r < 0
    except Exception:
        return False


def dedup_sky(cs, night_t, night_o, tol_arcsec=3.0, max_in=1500):
    """gen 6c: drop candidates whose predicted track is within tol of a higher-p one on >= 3 nights (or on all
    nights if fewer than 3).  A body counts once, so such duplicates only take slots in the global top-25N cut.
    cs must be sorted by decreasing p."""
    if len(cs) < 2:
        return cs
    cs = cs[:max_in]
    D = np.zeros((len(cs), len(night_t), 3))
    for ep in set(round(float(c["epoch_mjd"]), 6) for c in cs):
        idx = [k for k, c in enumerate(cs) if round(float(c["epoch_mjd"]), 6) == ep]
        S = np.array([cs[k]["state"] for k in idx], float)
        D[idx] = orbit.predict_dirs_b(S, ep, night_t, night_o)
    need = min(3, len(night_t))
    cosl = math.cos(tol_arcsec * orbit.ARC)
    cnt = np.zeros((len(cs), len(cs)), np.int16)
    for n in range(len(night_t)):
        cnt += (D[:, n] @ D[:, n].T > cosl)          # float64: float32 cannot resolve 1 - cos(3")
    dup = np.triu(cnt >= need, 1)          # dup[i, j]: j (lower p) duplicates i
    alive = np.ones(len(cs), bool)
    for i in range(len(cs)):
        if alive[i]:
            alive[i + 1:] &= ~dup[i, i + 1:]
    return [c for c, a in zip(cs, alive) if a]


def null_rows(rows, seed=12345, used=None):
    """Copy of rows with the nights' times/observer positions permuted (a derangement, few preserved adjacent
    night pairs), so no real body links across nights; everything the linker finds on it is junk.
    gen 9b: `used` (list of permutation tuples) excludes permutations already run; the chosen one is appended."""
    import copy
    nts = sorted(set(math.floor(float(r["mjd_mid"]) - 0.5) for r in rows))
    n = len(nts)
    if n < 3:
        return None
    used = used if used is not None else []
    rng = np.random.default_rng(seed)
    best, bs = None, 1e9
    for _ in range(300):
        pm = rng.permutation(n)
        if tuple(pm.tolist()) in used:
            continue
        sc = 10 * int((pm == np.arange(n)).sum()) + int((np.diff(pm) == 1).sum()) + int((np.diff(pm) == -1).sum())
        if sc < bs:
            best, bs = pm, sc
    if best is None:
        return None
    used.append(tuple(best.tolist()))
    perm = {nts[i]: nts[best[i]] for i in range(n)}
    rows = copy.deepcopy(rows)
    ref = {}
    for r in rows:
        ref.setdefault(math.floor(float(r["mjd_mid"]) - 0.5), (r["obs_x"], r["obs_y"], r["obs_z"]))
    for r in rows:
        t = float(r["mjd_mid"]); n0 = math.floor(t - 0.5); n1 = perm[n0]
        r["mjd_mid"] = repr(t - n0 + n1)
        r["obs_x"], r["obs_y"], r["obs_z"] = ref[n1]
    return rows


COVER_MIN = float(os.environ.get("TNO_COVER_MIN", "0.25"))


def calibrate(cs, CAL, floor=0.2, nocal=0.5):
    """gen 7a: make p comparable across scenarios (the grader keeps the global top 25 x N by p).
    For a real candidate with p = x: purity(x) ~ 1 - N_null(>= x) / N_real(>= x), the null counts coming from the
    same linker run on night-permuted times.  p' = p * (floor + (1 - floor) * purity), made monotone in p.
    Without a null run, p' = nocal * p.  cs sorted by decreasing p; returns new dicts."""
    nullp = CAL.get("null_p")
    out = []
    if nullp is None:
        for c in cs:
            d = dict(c); d["p"] = round(float(c["p"]) * nocal, 6); out.append(d)
        return out
    # gen 59: a null run cut short by its time slice (dense scenarios) linked only a fraction `cover` of the gamma
    # grid, so its counts understate the real run's chance linkages; an empty, starved null used to read as purity 1.
    # Scale N_null by c_real/cover (in [0.25, 4]); below COVER_MIN also cap p' at nocal*p.  cover = 1 (dev data) = gen058.
    cover = float(CAL.get("cover", 1.0))
    starved = cover < COVER_MIN
    creal = 1.0
    try:
        import link as _lk
        i0 = CAL.get("real_c0")
        if i0 is not None and len(_lk.COVER) > i0:
            creal = min(1.0, float(_lk.COVER[i0]))     # real run's pass-0 coverage (it may be cut too)
    except Exception:
        pass
    cscale = min(4.0, max(0.25, creal / max(cover, 0.01))) if cover < 1.0 else 1.0
    asc = np.sort(np.asarray(nullp, float))
    prev = 1.0
    for k, c in enumerate(cs):
        x = float(c["p"])
        nn = cscale * (len(asc) - np.searchsorted(asc, x, side="left")) / float(CAL.get("n_null", 1))
        pur = max(0.0, 1.0 - nn / (k + 2.0))     # +1 prior: one null hit cannot zero a lone top body
        q = min(prev, x * (floor + (1 - floor) * pur))
        if starved:                              # gen 59: starved null -> never gentler than nocal
            q = min(q, x * nocal)
        prev = q
        d = dict(c); d["p"] = round(float(q), 6)
        out.append(d)
    return out


# ----------------------------------------------------------------------------------------------- main
class Ctx:
    pass


def thin_rows(rows, k):
    """gen 60: in dense scenarios (median exposures/night > TNO_DENSE_N) the linker sees at most k exposures per
    night: k/2 time-adjacent pairs spread over the night (seeds need a same-night partner among the next 3).
    Detection, hough and the null runs then finish, so linking covers the whole gamma grid (gen059: 5-10/168).
    The stack stage still reads every exposure.  Sparse scenarios: rows returned unchanged."""
    by = {}
    for r in rows:
        by.setdefault(math.floor(float(r["mjd_mid"]) - 0.5), []).append(r)
    med = float(np.median([len(v) for v in by.values()])) if by else 0.0
    if k < 2 or med <= float(os.environ.get("TNO_DENSE_N", "6")):
        return rows
    out = []
    for v in by.values():
        v = sorted(v, key=lambda r: float(r["mjd_mid"]))
        if len(v) <= k:
            out += v
            continue
        st = np.unique(np.round(np.linspace(0, len(v) - 2, max(k // 2, 1))).astype(int))
        idx = sorted(set(st.tolist()) | set((st + 1).tolist()))
        out += [v[i] for i in idx]
    out = sorted(out, key=lambda r: float(r["mjd_mid"]))
    log(f"thin_rows: {len(rows)} -> {len(out)} exposures (median {med:.0f}/night, cap {k})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    ap.add_argument("--det-frac", type=float, default=0.5)
    ap.add_argument("--save-det", default="")
    ap.add_argument("--load-det", default="")
    a = ap.parse_args()
    budget = a.budget_s
    rows = list(csv.DictReader(open(os.path.join(a.scenario, "exposures.csv"))))
    try:
        rows = thin_rows(rows, int(float(os.environ.get("TNO_LMAXPN", "6") or 0)))
    except Exception as ex:
        log("thin_rows failed", ex)
    if os.environ.get("TNO_NULL"):
        # null test (development only): permute the nights' times and observer positions -> no real tracks
        rng = np.random.default_rng(int(os.environ["TNO_NULL"]))
        nts = sorted(set(math.floor(float(r["mjd_mid"]) - 0.5) for r in rows))
        perm = dict(zip(nts, rng.permutation(nts)))
        for r in rows:
            t = float(r["mjd_mid"]); n0 = math.floor(t - 0.5); n1 = perm[n0]
            ref = [x for x in rows if math.floor(float(x["mjd_mid"]) - 0.5) == n1][0]
            r["_t"] = t - n0 + n1
            r["_o"] = (ref["obs_x"], ref["obs_y"], ref["obs_z"])
        for r in rows:
            r["mjd_mid"] = str(r.pop("_t")); r["obs_x"], r["obs_y"], r["obs_z"] = r.pop("_o")
        log("NULL TEST: nights permuted", perm)
    try:
        meta = json.load(open(os.path.join(a.scenario, "meta.json")))
    except Exception:
        meta = {}
    cands = []
    CAL = {"null_p": None}

    # gen 6c: one time + observer per night, for sky-track de-duplication of the output
    _nt = {}
    for r in rows:
        t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
        _nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
    night_t = np.array([v[0] for v in _nt.values()])
    night_o = np.array([v[1] for v in _nt.values()])

    def write():
        cs = sorted([c for c in cands if valid_state(c)], key=lambda c: -c["p"])
        try:
            cs = calibrate(cs, CAL)
        except Exception as ex:
            log("calibrate failed", ex)
        try:
            cs = dedup_sky(cs, night_t, night_o)
        except Exception as ex:
            log("dedup failed", ex)
        cs = cs[:200]
        tmp = a.out + ".tmp"
        json.dump({"candidates": cs}, open(tmp, "w"))
        os.replace(tmp, a.out)
    write()

    if a.load_det:
        results = np.load(a.load_det, allow_pickle=True).item()
    else:
        results = run_detection(rows, a.scenario, budget, a.det_frac)
        if a.save_det:
            np.save(a.save_det, results, allow_pickle=True)
    if not results:
        return
    exps = sorted(results)
    import link
    # gen 7a: runtime null calibration (see calibrate) - run first, on night-permuted times, pass 0 only
    try:
        t_det = time.time() - orbit.T0
        R = budget - t_det
        nfrac = float(os.environ.get("TNO_NULLFRAC", 0.3))
        if R > 0.4 * budget and nfrac > 0 and not os.environ.get("TNO_NULL"):
            # gen 9b: up to TNO_NNULL (3) distinct night permutations, averaged N_null (less noisy top end).
            # The first gets the gen-7a budget slice; more run only if they fit in the same overall allowance.
            used, alln, nrun, covs = [], [], 0, []
            t_null0 = time.time()
            nmax = int(os.environ.get("TNO_NNULL", 3))
            for kk in range(nmax):
                el = time.time() - orbit.T0
                if kk > 0:
                    d_last = time.time() - t_k
                    if el + 1.2 * d_last > t_det + (nfrac + 0.1) * R or el + 1.2 * d_last > 0.5 * budget:
                        break
                    b_null = (el + 1.2 * d_last) / 0.95
                else:
                    b_null = (t_det + nfrac * R) / 0.95
                rows_n = null_rows(rows, seed=12345 + 7919 * kk, used=used)
                if rows_n is None:
                    break
                t_k = time.time()
                cn = []
                old = {k: os.environ.get(k) for k in ("TNO_PASSES", "TNO_NOSTACK", "TNO_DEBUG")}
                os.environ["TNO_PASSES"] = "1"; os.environ["TNO_NOSTACK"] = "1"; os.environ.pop("TNO_DEBUG", None)
                try:
                    log(f"null calibration run {kk}: budget {b_null:.0f}s perm {used[-1]}")
                    link.GRID_ROWS[0] = rows
                    c0 = len(link.COVER)
                    link.run(a, rows_n, meta, results, cn, lambda: None, b_null)
                finally:
                    link.GRID_ROWS[0] = None
                    for k, v in old.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
                pk = [c["p"] for c in cn if valid_state(c)]
                alln += pk; nrun += 1
                # gen 59: how much of the gamma grid this null run linked (0 if it never reached linking)
                covs.append(link.COVER[c0] if len(link.COVER) > c0 else 0.0)
                CAL["null_p"] = np.sort(alln)[::-1]; CAL["n_null"] = nrun; CAL["cover"] = float(np.mean(covs))
                log(f"null run {kk} gamma coverage {covs[-1]:.2f}")
                log(f"null run {kk} done at {time.time() - orbit.T0:.0f}s ({time.time() - t_k:.0f}s): {len(pk)} "
                    f"null candidates, top p {np.sort(pk)[::-1][:5].round(3).tolist()}")
            log(f"{nrun} null runs in {time.time() - t_null0:.0f}s")
    except Exception as ex:
        import traceback; traceback.print_exc()
        log("null calibration failed", ex)
        if not CAL.get("n_null"):
            CAL["null_p"] = None
    CAL["real_c0"] = len(link.COVER)
    link.run(a, rows, meta, results, cands, write, budget)
    write()
    log("done", len(cands), "candidates")


if __name__ == "__main__":
    main()
