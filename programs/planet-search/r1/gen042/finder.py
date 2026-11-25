"""TNO finder, gen 10e: hybrid of two earlier programs, run one after the other, outputs merged.

1. linker/  = gen009b (detect-then-link, HelioLinC/BK frame, null-calibrated p; reward 0.1128).
2. stackd/  = gen009d (orbit-space GPU shift-and-stack + CEM refinement; reward 0.0977, ~72 s/scenario on the A100).
   It runs in the budget the linker leaves unused.
3. Merge: candidates whose predicted sky tracks agree (< 2.5" on >= 3 nights) are one body: keep one, boost its p
   (two independent methods agree).  Stack-only candidates get p * STACK_W.  Top 200 by p.
OUT.json is written after each stage.
"""
import argparse, csv, json, math, os, shutil, signal, subprocess, sys, time
import numpy as np

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker"))
import orbit  # noqa: E402

ARC = math.pi / 180 / 3600
STACK_W = float(os.environ.get("TNO_STACK_W", "0.7"))
MATCH_AS = 4.0   # gen 12e: was 2.5; refined linker seeds may move a little
# gen 14c: rank-balanced p for the global top-25N cut (see NOTES.md).  p' = 1 - (1 - p)(1 - Q0 exp(-rank / RQ)).
RANK_Q0 = float(os.environ.get("TNO_RANKQ0", "0.7"))  # gen 41: was 0.25
RANK_R = float(os.environ.get("TNO_RANKR", "25"))
# gen 20: interleave stack-only candidates with the linker's weak tail (see NOTES.md).
ILEAVE = int(float(os.environ.get("TNO_ILEAVE", "1") or 0))          # stack-only candidates per linker-tail candidate (0 = off)
P_STRONG = float(os.environ.get("TNO_PSTRONG", "0.8"))  # candidates at or above this p keep the top ranks


# gen 38: candidates with orbits implausible for a TNO (retrograde, or semi-major axis < 25 au) are moved to the
# bottom of the scenario's list; sorted p values are reassigned, so only the order changes (see NOTES.md).
LFRAC = float(os.environ.get("TNO_LFRAC", "0.75"))   # gen 42: linker share of the budget (env for tests)
PHYS = int(float(os.environ.get("TNO_PHYS", "1") or 0))
PHYS_AMIN = float(os.environ.get("TNO_PHYS_AMIN", "25"))
PHYS_IMAX = float(os.environ.get("TNO_PHYS_IMAX", "90"))
R_ECL = np.array([[1, 0, 0], [0, math.cos(math.radians(23.4392911)), math.sin(math.radians(23.4392911))],
                  [0, -math.sin(math.radians(23.4392911)), math.cos(math.radians(23.4392911))]])  # eq -> ecl


def implausible(c):
    s = np.array(c["state"], float)
    r, v = R_ECL @ s[:3], R_ECL @ s[3:]
    rn = float(np.linalg.norm(r))
    en = 0.5 * float(v @ v) - orbit.GM / rn
    if en >= 0:
        return True
    a = -orbit.GM / (2 * en)
    h = np.cross(r, v)
    inc = math.degrees(math.acos(max(-1.0, min(1.0, h[2] / np.linalg.norm(h)))))
    return a < PHYS_AMIN or inc > PHYS_IMAX


# gen 42: any source (L, LS, S) whose fitted orbit is highly inclined, i > max(PHYS_IALL, |beta| + PHYS_IMARG) with
# beta the body's own heliocentric ecliptic latitude (a body at latitude beta has i >= |beta|), is demoted too.
PHYS_IALL = float(os.environ.get("TNO_PHYS_IALL", "50"))     # 0 = off (gen041 behaviour)
PHYS_IMARG = float(os.environ.get("TNO_PHYS_IMARG", "25"))


def high_inc(c):
    if PHYS_IALL <= 0:
        return False
    s = np.array(c["state"], float)
    r, v = R_ECL @ s[:3], R_ECL @ s[3:]
    h = np.cross(r, v)
    inc = math.degrees(math.acos(max(-1.0, min(1.0, h[2] / np.linalg.norm(h)))))
    beta = math.degrees(math.asin(max(-1.0, min(1.0, r[2] / np.linalg.norm(r)))))
    return inc > max(PHYS_IALL, abs(beta) + PHYS_IMARG)


def phys_demote(cs):
    if not PHYS:
        return cs
    try:
        cs = sorted(cs, key=lambda c: -float(c["p"]))
        ps = [float(c["p"]) for c in cs]
        bad = [(c.get("src") == "L" and implausible(c)) or high_inc(c) for c in cs]
        order = [c for c, b in zip(cs, bad) if not b] + [c for c, b in zip(cs, bad) if b]
        log(f"phys_demote: {sum(bad)} of {len(cs)} implausible orbits moved down "
            f"({sum(bad[:25])} were in the top 25)")
        return [dict(c, p=p) for c, p in zip(order, ps)]
    except Exception as ex:
        log("phys_demote failed", ex)
        return cs


# gen 40: cross-scenario calibration by stack-only junk density (see NOTES.md).  When a scenario holds many more
# high-p stack-only candidates than CAL_R per high-p linker-found one (+ CAL_N0), every stack-only p is multiplied by
# f = max(CAL_FMIN, (CAL_R * n_conf + CAL_N0) / n_S_hi), then p is made non-increasing along the scenario's order, so
# the scenario's own order is unchanged and only the global top-25N cut is affected.  TNO_CAL=0 restores gen038.
CAL = int(float(os.environ.get("TNO_CAL", "1") or 0))
CAL_R = float(os.environ.get("TNO_CAL_R", "0.25"))
CAL_PHI = float(os.environ.get("TNO_CAL_PHI", "0.4"))
CAL_N0 = float(os.environ.get("TNO_CAL_N0", "3"))
CAL_FMIN = float(os.environ.get("TNO_CAL_FMIN", "0.6"))
CAL_PMAX = float(os.environ.get("TNO_CAL_PMAX", "0.9"))  # stack-only candidates at or above this p are not shrunk
CAL_RMIN = int(float(os.environ.get("TNO_CAL_RMIN", "0") or 0))  # 1 = running minimum (order kept; drags the L tail)


def dens_calib(cs):
    if not CAL or len(cs) < 2:
        return cs
    try:
        cs = sorted(cs, key=lambda c: -float(c["p"]))
        p = np.clip(np.nan_to_num(np.array([float(c["p"]) for c in cs]), nan=0.0), 0.0, 1.0)
        isS = np.array([c.get("src") == "S" for c in cs]) & (p < CAL_PMAX)
        hi = p >= CAL_PHI
        n_conf, n_s = int((hi & ~isS).sum()), int((hi & isS).sum())
        f = 1.0 if n_s <= 0 else float(np.clip((CAL_R * n_conf + CAL_N0) / n_s, CAL_FMIN, 1.0))
        log(f"dens_calib: {n_conf} high-p linker-found, {n_s} high-p stack-only -> stack-only p x {f:.3f}")
        if f >= 1.0:
            return cs
        q = np.where(isS, p * f, p)
        if CAL_RMIN:
            q = np.minimum.accumulate(q)
        out = [dict(c, p=float(x)) for c, x in zip(cs, q)]
        return sorted(out, key=lambda c: -c["p"])
    except Exception as ex:
        log("dens_calib failed", ex)
        return cs


def interleave(cs):
    """Reorder (not rescale): candidates with p >= P_STRONG keep the top, then the weak linker tail and the
    stack-only candidates alternate (ILEAVE stack-only per linker one), each list in its own p order.  The
    scenario's sorted p values are then reassigned along the new order, so the p distribution (and with it the
    cross-scenario calibration) is unchanged; only who holds the top-25 slots changes."""
    if ILEAVE <= 0 or len(cs) < 2:
        return cs
    cs = sorted(cs, key=lambda c: -float(c["p"]))
    ps = [float(c["p"]) for c in cs]
    top = [c for c in cs if float(c["p"]) >= P_STRONG]
    rest = [c for c in cs if float(c["p"]) < P_STRONG]
    L = [c for c in rest if c.get("src") != "S"]
    S = [c for c in rest if c.get("src") == "S"]
    order, i, j = list(top), 0, 0
    while i < len(L) or j < len(S):
        for _ in range(ILEAVE):
            if j < len(S):
                order.append(S[j]); j += 1
        if i < len(L):
            order.append(L[i]); i += 1
    log(f"interleave: {len(top)} strong, {len(L)} linker tail, {len(S)} stack-only")
    return [dict(c, p=p) for c, p in zip(order, ps)]


def rank_balance(cs):
    """Monotone within the scenario (both terms fall with rank), so the scenario's own order is unchanged.  Across
    scenarios, the low-p tail (p ~ 0.02-0.06, set mostly by each scenario's raw score scale) is ordered by rank,
    so every scenario gets ~its top 20-30 into the global cut instead of one scenario's tail taking the slots."""
    cs = sorted(cs, key=lambda c: -float(c["p"]))
    if RANK_Q0 <= 0:
        return cs
    out = []
    for r, c in enumerate(cs):
        p = min(max(float(c["p"]), 0.0), 1.0)
        q = RANK_Q0 * math.exp(-r / RANK_R)
        out.append(dict(c, p=1 - (1 - p) * (1 - q)))
    return out


def log(*a):
    print(f"[wrap {time.time() - T0:6.1f}s]", *a, flush=True)


def run(sub, scen, out, budget, hard, env=None):
    cmd = [sys.executable, "-I", os.path.join(HERE, sub, "finder.py"), "--scenario", scen, "--out", out,
           "--budget-s", f"{budget:.1f}"]
    log("start", sub, f"budget {budget:.0f}s, hard stop {hard:.0f}s")
    t = time.time()
    p = subprocess.Popen(cmd, cwd=os.path.join(HERE, sub), start_new_session=True,
                         env=dict(os.environ, **(env or {})))
    try:
        p.wait(timeout=max(5.0, hard))
    except subprocess.TimeoutExpired:
        log(sub, "over its hard stop: killing")
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            p.kill()
        p.wait()
    log("done", sub, f"rc {p.returncode} in {time.time() - t:.0f}s")


def load(path):
    try:
        return list(json.load(open(path))["candidates"])
    except Exception:
        return []


def valid(c):
    try:
        s = np.array(c["state"], float)
        if s.shape != (6,) or not np.all(np.isfinite(s)) or not math.isfinite(float(c["epoch_mjd"])):
            return False
        r = np.linalg.norm(s[:3])
        return r >= 25 and 0.5 * s[3:] @ s[3:] - orbit.GM / r < 0
    except Exception:
        return False


def tracks(cs, night_t, night_o):
    """(K, Nn, 3) unit vectors at one time per night."""
    if not cs:
        return np.zeros((0, len(night_t), 3))
    out = []
    for c in cs:
        out.append(orbit.predict_dirs(np.array(c["state"], float), float(c["epoch_mjd"]), night_t, night_o))
    return np.array(out)


def merge(A, B, night_t, night_o):
    A = [dict(c, src="L") for c in A if valid(c)]
    B = [dict(c, src="S") for c in B if valid(c)]
    for c in B:
        c["p"] = float(c["p"]) * STACK_W
    if not B:
        return dedup_sky(A, night_t, night_o)
    ua, ub = tracks(A, night_t, night_o), tracks(B, night_t, night_o)
    need = min(3, len(night_t))
    keepB = np.ones(len(B), bool)
    nmatch = 0
    if len(A):
        cosm = math.cos(MATCH_AS * ARC)
        for j in range(len(B)):
            dots = np.einsum("knd,nd->kn", ua, ub[j])          # (K, Nn)
            close = (dots > cosm).sum(1)
            hit = np.nonzero(close >= need)[0]
            if len(hit):
                i = hit[np.argmax([A[k]["p"] for k in hit])]
                pa, pb = float(A[i]["p"]), float(B[j]["p"])
                A[i]["p"] = 1 - (1 - pa) * (1 - 0.5 * pb)
                A[i]["src"] = "LS"
                keepB[j] = False
                nmatch += 1
    log(f"merge: {len(A)} linker, {len(B)} stack, {nmatch} stack candidates matched a linker one")
    allc = A + [c for c, k in zip(B, keepB) if k]
    return dens_calib(phys_demote(interleave(dedup_sky(allc, night_t, night_o))))


def dedup_sky(cs, night_t, night_o, tol_as=1.5):
    """gen 16: drop candidates whose predicted sky track is within tol_as of a higher-p one on >= 3 nights (stackd's
    own dedup compares heliocentric directions, so same-body copies at slightly different distances survived)."""
    if len(cs) < 2:
        return cs
    try:
        cs = sorted(cs, key=lambda c: -float(c["p"]))
        U = tracks(cs, night_t, night_o)
        need = min(3, len(night_t))
        cosm = math.cos(tol_as * ARC)
        keep = []
        for j in range(len(cs)):
            if keep:
                close = (np.einsum("knd,nd->kn", U[keep], U[j]) > cosm).sum(1)
                if (close >= need).any():
                    continue
            keep.append(j)
        log(f"sky dedup: {len(cs)} -> {len(keep)}")
        return [cs[k] for k in keep]
    except Exception as ex:
        log("sky dedup failed", ex)
        return cs


def write(path, cs):
    cs = rank_balance(cs)[:200]
    out = []
    for c in cs:
        d = {"epoch_mjd": float(c["epoch_mjd"]), "state": [float(x) for x in c["state"]],
             "p": float(min(max(float(c["p"]), 0.0), 1.0))}
        if c.get("mag") is not None:
            try:
                m = float(c["mag"])
                if math.isfinite(m):
                    d["mag"] = m
            except Exception:
                pass
        out.append(d)
    tmp = path + ".tmp"
    json.dump({"candidates": out}, open(tmp, "w"))
    os.replace(tmp, path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    a = ap.parse_args()
    B = a.budget_s
    scen = os.path.abspath(a.scenario)
    out = os.path.abspath(a.out)
    write(out, [])
    rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
    nt = {}
    for r in rows:
        t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
        nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
    night_t = np.array([v[0] for v in nt.values()])
    night_o = np.array([v[1] for v in nt.values()])

    outL, outS = out + ".linker.json", out + ".stack.json"
    # 1. linker (best program); leaves >= 25 % of the budget for the stack stage
    bl = B - max(140.0, (1.0 - LFRAC) * B)
    run("linker", scen, outL, bl, bl + 15)
    A = load(outL)
    try:
        write(out, merge(A, [], night_t, night_o))
    except Exception as ex:
        log("write A failed", ex)
        if os.path.exists(outL):
            shutil.copy(outL, out)
    # 2. stack (in what is left)
    bs = B - (time.time() - T0) - 12
    if bs >= 60:
        run("stackd", scen, outS, bs, bs + 6, env={"TNO_SEEDS": outL} if os.path.exists(outL) else None)
        S = load(outS)
        try:
            final = write(out, merge(A, S, night_t, night_o))
            log(f"final {len(final)} candidates; top p {[round(c['p'], 3) for c in final[:8]]}")
        except Exception as ex:
            import traceback
            traceback.print_exc()
            log("merge failed; keeping linker output", ex)
    else:
        log("no time left for the stack stage")
    for f in (() if os.environ.get("TNO_KEEP") else (outL, outS)):
        try:
            os.remove(f)
        except Exception:
            pass


if __name__ == "__main__":
    main()
