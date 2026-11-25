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
# gen 78: stack-confirmed linker candidates (LS) whose matched stack p beats their own linker p compete in the
# stack-only interleave list instead of the linker tail (0 = gen077).
LSRANK = int(float(os.environ.get("TNO_LSRANK", "1") or 0))
# gen 126: in the interleave tail, linker rows stackd did NOT confirm (src "L": stackd refined every linker seed but no
# stack row matched) get one slot per LPURE stack-only rows; confirmed LS rows keep ILEAVE and go first in the
# linker list.  Order only (sorted p reassigned).  Needs >= LPURE_MINS stack-only rows.  0 = gen125.
LPURE = int(float(os.environ.get("TNO_LPURE", "0") or 0))
LPURE_MINS = int(float(os.environ.get("TNO_LPURE_MINS", "50") or 0))
# gen 81: a linker candidate from the linker's own shift-and-stack (kind "stk": state fitted to 2 seed detections +
# model-predicted positions elsewhere) that stackd matched takes stackd's pixel-refined state (0 = gen080).
ADOPT = int(float(os.environ.get("TNO_LSADOPT", "1") or 0))
ADOPT_AS = float(os.environ.get("TNO_LSADOPT_AS", "2.0"))  # the two tracks must also agree within this on >= 2 nights
# gen 123: partial-track suppression (order only).  A stack-only row whose track passes within PNMS arcsec of a
# higher-ranked anchor row on >= PNMS_K nights (but is not a full duplicate) is moved below the other rows.
PNMS = float(os.environ.get("TNO_PNMS", "0") or 0)          # arcsec (2.0 to try); 0 = gen122 behaviour (default)
PNMS_K = int(float(os.environ.get("TNO_PNMS_K", "1") or 1))  # shared nights needed
PNMS_ANCH = os.environ.get("TNO_PNMS_ANCH", "SLS")           # anchors: "SLS" = S and LS rows, "all" = any row
EXPT = None   # (t[E], obs[E, 3], night index[E]) set in main(); <= 6 exposures per night


# gen 38: candidates with orbits implausible for a TNO (retrograde, or semi-major axis < 25 au) are moved to the
# bottom of the scenario's list; sorted p values are reassigned, so only the order changes (see NOTES.md).
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


def phys_demote(cs):
    if not PHYS:
        return cs
    try:
        cs = sorted(cs, key=lambda c: -float(c["p"]))
        ps = [float(c["p"]) for c in cs]
        bad = [c.get("src") == "L" and implausible(c) for c in cs]
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
    def _toS(c):
        return c.get("src") == "S" or (LSRANK > 0 and c.get("src") == "LS"
                                       and float(c.get("pS", 0.0)) > float(c.get("pL", 1.0)))
    L = [c for c in rest if not _toS(c)]
    S = [c for c in rest if _toS(c)]
    nLS = sum(c.get("src") == "LS" for c in S)
    if nLS:
        S = sorted(S, key=lambda c: -max(float(c["p"]), float(c.get("pS", 0.0))))
    def _ilv(L_, S_, kp):
        o_, i, j = [], 0, 0
        while i < len(L_) or j < len(S_):
            k_ = kp if (i < len(L_) and L_[i].get("src") == "L") else ILEAVE
            for _ in range(k_):
                if j < len(S_):
                    o_.append(S_[j]); j += 1
            if i < len(L_):
                o_.append(L_[i]); i += 1
        return o_
    nS0 = sum(c.get("src") == "S" for c in S)
    if LPURE > ILEAVE and nS0 >= LPURE_MINS:
        L2 = [c for c in L if c.get("src") != "L"] + [c for c in L if c.get("src") == "L"]
        new = _ilv(L2, S, LPURE)
        try:
            old = _ilv(L, S, ILEAVE)
            def _cnt(o):
                t_ = (list(top) + o)[:25]
                return "/".join(str(sum(c.get("src") == s_ for c in t_)) for s_ in ("L", "LS", "S"))
            log(f"lpure: {sum(c.get('src') == 'L' for c in L)} unconfirmed linker-tail rows at 1 per {LPURE} stack rows; "
                f"top-25 L/LS/S {_cnt(old)} -> {_cnt(new)}")
        except Exception as ex:
            log("lpure log failed", ex)
        order = list(top) + new
    else:
        order = list(top) + _ilv(L, S, ILEAVE)
    log(f"interleave: {len(top)} strong, {len(L)} linker tail, {len(S)} stack-only ({nLS} stack-confirmed LS moved in)")
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


# gen 132 (TNO_VET, default 1; 0 = gen129): forced photometry of the final candidates on the RAW images (vet.py):
# on-track S/N per exposure and the S/N at the same sky positions in other nights' exposures (static-source test).
# p is then re-weighted (vet_adjust); the time comes from stackd's budget (VET_RES seconds).
VET = int(float(os.environ.get("TNO_VET", "1") or 0))
VET_RES = float(os.environ.get("TNO_VET_RES", "10"))
VET_N = int(float(os.environ.get("TNO_VET_N", "200")))


def vet_adjust(cs, fs):
    """Return a copy of cs with p re-weighted from the vet features (None features: unchanged)."""
    out = []
    nd = 0
    for c, f in zip(cs, fs):
        p = min(max(float(c["p"]), 0.0), 1.0)
        if f is not None:
            m = vet_factor(f)
            if m < 1.0:
                nd += 1
            p = p * m
        out.append(dict(c, p=p, vet=f))
    log(f"vet: {nd} of {len(cs)} candidates demoted")
    return out


VET_LO = float(os.environ.get("TNO_VET_LO", "1.5"))     # zlbno at/below which p gets the floor factor
VET_HI = float(os.environ.get("TNO_VET_HI", "4.5"))     # zlbno at/above which p is unchanged
VET_FL = float(os.environ.get("TNO_VET_FL", "0.5"))     # floor factor
VET_ST = float(os.environ.get("TNO_VET_ST", "0.3"))     # extra factor for static-looking tracks (zoff > max(3, zlbno))


def vet_factor(f):
    """zlbno = forced on-track S/N without the brightest night: a real body has signal on several nights;
    single-night spikes and noise tracks do not.  zoff = the same sky positions at other epochs (static source)."""
    try:
        z = float(f["zlbno"]); zo = float(f["zoff"])
    except Exception:
        return 1.0
    if not (math.isfinite(z) and math.isfinite(zo)):
        return 1.0
    m = VET_FL + (1.0 - VET_FL) * min(max((z - VET_LO) / max(VET_HI - VET_LO, 1e-6), 0.0), 1.0)
    if zo > max(3.0, z):
        m *= VET_ST
    return m


def vet_run(scen, merged, deadline):
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import vet as vet_
    cs = sorted(merged, key=lambda c: -float(c["p"]))
    head, tail = cs[:VET_N], cs[VET_N:]
    nw = 4 if not os.environ.get("TNO_VET_W") else int(os.environ["TNO_VET_W"])
    fs = vet_.measure(scen, head, deadline=deadline, log=log, workers=nw)
    return vet_adjust(head, fs) + tail


def start(sub, scen, out, budget, hard, env=None, nice=0):
    cmd = [sys.executable, "-I", os.path.join(HERE, sub, "finder.py"), "--scenario", scen, "--out", out,
           "--budget-s", f"{budget:.1f}"]
    log("start", sub, f"budget {budget:.0f}s, hard stop {hard:.0f}s" + (f", nice {nice}" if nice else ""))
    pre = None
    if nice:
        def pre():
            # runs after setsid (start_new_session): on Linux with autogroup, nice alone only ranks processes inside
            # the new session, so also lower the session's autogroup priority.
            try:
                os.nice(nice)
            except Exception:
                pass
            try:
                with open("/proc/self/autogroup", "w") as f_:
                    f_.write(str(nice))
            except Exception:
                pass
    return subprocess.Popen(cmd, cwd=os.path.join(HERE, sub), start_new_session=True,
                            env=dict(os.environ, **(env or {})), preexec_fn=pre)


def run(sub, scen, out, budget, hard, env=None, p=None):
    t = time.time()
    if p is None:
        p = start(sub, scen, out, budget, hard, env)
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
    nmatch = nadopt = 0
    if len(A):
        cosm = math.cos(MATCH_AS * ARC)
        for j in range(len(B)):
            dots = np.einsum("knd,nd->kn", ua, ub[j])          # (K, Nn)
            close = (dots > cosm).sum(1)
            hit = np.nonzero(close >= need)[0]
            if len(hit):
                i = hit[np.argmax([A[k]["p"] for k in hit])]
                pa, pb = float(A[i]["p"]), float(B[j]["p"])
                A[i]["pL"] = float(A[i].get("pL", pa))
                A[i]["pS"] = max(float(A[i].get("pS", 0.0)), pb)
                A[i]["p"] = 1 - (1 - pa) * (1 - 0.5 * pb)
                A[i]["src"] = "LS"
                keepB[j] = False
                nmatch += 1
                if ADOPT and A[i].get("kind") == "stk" and pb > float(A[i].get("_adopt_p", -1.0)):
                    d_ = np.einsum("nd,nd->n", ua[i], ub[j])
                    if (d_ > math.cos(ADOPT_AS * ARC)).sum() >= min(2, len(night_t)):
                        sep = np.degrees(np.arccos(np.clip(d_, -1, 1))) * 3600
                        A[i]["state"] = [float(x) for x in B[j]["state"]]
                        A[i]["epoch_mjd"] = float(B[j]["epoch_mjd"])
                        A[i]["_adopt_p"] = pb
                        nadopt += 1
                        log(f"adopt: stk linker cand p={pa:.3f} takes stack state pS={pb:.3f}; "
                            f"per-night sep {np.round(sep, 2).tolist()}\"")
    log(f"merge: {len(A)} linker, {len(B)} stack, {nmatch} stack candidates matched a linker one, {nadopt} stk states adopted")
    allc = A + [c for c, k in zip(B, keepB) if k]
    return pnms(dens_calib(phys_demote(interleave(dedup_sky(allc, night_t, night_o)))))


def pnms(cs):
    if PNMS <= 0 or EXPT is None or len(cs) < 3:
        return cs
    try:
        cs = sorted(cs, key=lambda c: -float(c["p"]))
        ps = [float(c["p"]) for c in cs]
        t_, o_, nid = EXPT
        nn = int(nid.max()) + 1
        U = np.array([orbit.predict_dirs(np.array(c["state"], float), float(c["epoch_mjd"]), t_, o_) for c in cs])
        cosm = math.cos(PNMS * ARC)
        anch = np.array([PNMS_ANCH == "all" or c.get("src") in ("S", "LS") for c in cs])
        keep = np.ones(len(cs), bool)
        dem = []
        for j in range(1, len(cs)):
            if cs[j].get("src") != "S":
                continue
            A_ = np.flatnonzero(keep[:j] & anch[:j])
            if not len(A_):
                continue
            close = np.einsum("ked,ed->ke", U[A_], U[j]) > cosm
            shared = np.zeros(len(A_), int)
            for k in range(nn):
                shared += close[:, nid == k].any(1)
            if (shared >= PNMS_K).any():
                keep[j] = False
                dem.append(j)
        order = [cs[k] for k in np.flatnonzero(keep)] + [cs[k] for k in dem]
        log(f"pnms: {len(dem)} stack-only rows share >= {PNMS_K} night(s) with a higher row; demoted ranks {dem[:30]}")
        return [dict(c, p=p) for c, p in zip(order, ps)]
    except Exception as ex:
        log("pnms failed", ex)
        return cs


DEDUP_AS = float(os.environ.get("TNO_DEDUP_AS", "2.5"))  # gen 106: was 1.5 (duplicates survived)


def dedup_sky(cs, night_t, night_o, tol_as=None):
    """gen 16: drop candidates whose predicted sky track is within tol_as of a higher-p one on >= 3 nights (stackd's
    own dedup compares heliocentric directions, so same-body copies at slightly different distances survived)."""
    if len(cs) < 2:
        return cs
    try:
        tol_as = DEDUP_AS if tol_as is None else tol_as
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


def set_expt(rows):
    global EXPT
    try:
        g = {}
        for r in rows:
            t_ = float(r["mjd_mid"])
            g.setdefault(math.floor(t_ - 0.5), []).append((t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
        T_, O_, I_ = [], [], []
        for k, n_ in enumerate(sorted(g)):
            v = sorted(g[n_], key=lambda x: x[0])
            if len(v) > 6:
                v = [v[i] for i in np.unique(np.linspace(0, len(v) - 1, 6).round().astype(int))]
            for t_, o_ in v:
                T_.append(t_); O_.append(o_); I_.append(k)
        EXPT = (np.array(T_), np.array(O_), np.array(I_))
    except Exception as ex:
        log("expt failed", ex)
        EXPT = None


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
    set_expt(rows)

    outL, outS = out + ".linker.json", out + ".stack.json"
    # 1. linker (best program); leaves >= 25 % of the budget for the stack stage
    # gen 57: dense scenarios (many exposures per night) give the stack stage, the only co-add path, a bigger share:
    # a body at night-stack S/N 5 is ~1 sigma per exposure at 20+ exposures/night, out of the linker's reach, and the
    # stack stage's caps (MAXE, HMAX, map deadline) scale with its budget.  Sparse scenarios (<= DENSE_N exposures on
    # every night, all dev data) keep the gen056 split exactly.
    ppn = {}
    for r in rows:
        k_ = math.floor(float(r["mjd_mid"]) - 0.5)
        ppn[k_] = ppn.get(k_, 0) + 1
    med_pn = float(np.median(list(ppn.values()))) if ppn else 0.0
    dense = med_pn > float(os.environ.get("TNO_DENSE_N", "6"))
    sfrac = float(os.environ.get("TNO_SFRAC_DENSE", "0.5")) if dense else float(os.environ.get("TNO_SFRAC", "0.25"))
    log(f"{len(rows)} exposures, {len(ppn)} nights, median {med_pn:.0f}/night -> dense={dense}, stack share {sfrac}")
    bl = B - max(140.0, sfrac * B)
    # gen 101: the linker skips its own pair-seeded stack.py (TNO_LNOSTACK=1), so the time it would use goes to stackd,
    # the GPU orbit-space stack (night/window/deep seeds).  0 = gen100.
    lenv = {"TNO_NOSTACK": "1"} if int(float(os.environ.get("TNO_LNOSTACK", "0") or 0)) else None
    # gen 116: concurrent mode.  The GPU idles while the CPU-bound linker runs (~75 % of a sparse scenario's budget),
    # so stackd starts at t=0 with (almost) the whole budget instead of the ~25 % left after the linker.  It does its
    # mapping/template/fstk/window search alongside the linker and waits for the linker's final output (marker file)
    # before refinement, so the linker seeds are refined exactly as before.  TNO_CONC=0 = gen115.
    conc = int(float(os.environ.get("TNO_CONC", "1") or 0))
    doneL = outL + ".done"
    pS = None
    if conc:
        for f_ in (doneL, outL, outS):   # a stale marker would make stackd skip its wait
            try:
                os.remove(f_)
            except Exception:
                pass
        bsc = B - (time.time() - T0) - 14 - (VET_RES if VET else 0.0)
        senv = {"TNO_SEEDS": outL, "TNO_SEEDS_WAIT": doneL}
        senv["TNO_CONC_WORKERS"] = os.environ.get("TNO_CONC_WORKERS", "4")   # used on CUDA (Mac: 2, stackd/finder.py)
        pS = start("stackd", scen, outS, bsc, bsc + 6, env=senv,
                   nice=int(float(os.environ.get("TNO_CONC_NICE", "15") or 0)))
        tS0 = time.time()
    try:
        run("linker", scen, outL, bl, bl + 15, env=lenv)
    except BaseException:
        if pS is not None:   # do not leave stackd running on the GPU
            try:
                os.killpg(pS.pid, signal.SIGKILL)
            except Exception:
                pass
        raise
    A = load(outL)
    if conc:
        try:
            open(doneL, "w").write("done")
        except Exception as ex:
            log("marker failed", ex)
    try:
        write(out, merge(A, [], night_t, night_o))
    except Exception as ex:
        log("write A failed", ex)
        if os.path.exists(outL):
            shutil.copy(outL, out)
    # 2. stack (in what is left)
    bs = B - (time.time() - T0) - 12
    if conc:
        run("stackd", scen, outS, 0, max(5.0, bsc + 6 - (time.time() - tS0)), p=pS)
        S = load(outS)
        try:
            merged = merge(A, S, night_t, night_o)
            final = write(out, merged)
            log(f"final {len(final)} candidates; top p {[round(c['p'], 3) for c in final[:8]]}")
            if VET:
                try:
                    vetted = vet_run(scen, merged, T0 + B - 5)
                    final = write(out, vetted)
                    log(f"vetted {len(final)} candidates; top p {[round(c['p'], 3) for c in final[:8]]}")
                except Exception as ex:
                    import traceback
                    traceback.print_exc()
                    log("vet failed; keeping the merged output", ex)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            log("merge failed; keeping linker output", ex)
    elif bs >= 60:
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
    for f in (() if os.environ.get("TNO_KEEP") else (outL, outS, doneL)):
        try:
            os.remove(f)
        except Exception:
            pass


if __name__ == "__main__":
    main()
