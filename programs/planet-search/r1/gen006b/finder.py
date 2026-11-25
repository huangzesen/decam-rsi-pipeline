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


# ----------------------------------------------------------------------------------------------- main
class Ctx:
    pass


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

    def write():
        cs = sorted([c for c in cands if valid_state(c)], key=lambda c: -c["p"])[:200]
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
    # gen 6b: within-night pixel shift-and-stack -> pseudo-exposures of night-stack detections (nstack.py)
    if not os.environ.get("TNO_NONSTACK"):
        try:
            import nstack
            el = time.time() - T0
            dl = T0 + min(0.62 * budget, el + 0.12 * budget)
            npz = nstack.add_night_stacks(rows, meta, results, dl)
            log(f"nstack: {npz} pseudo-exposures")
        except Exception as ex:
            import traceback; traceback.print_exc()
            log("nstack failed", ex)
            for k in [k for k, o in results.items() if o.get("pseudo")]:
                results.pop(k)
            del rows[len(rows) - sum(1 for r in rows if str(r.get("exp_id", "")).startswith("nstack")):]
    import link
    link.run(a, rows, meta, results, cands, write, budget)
    write()
    log("done", len(cands), "candidates")


if __name__ == "__main__":
    main()
