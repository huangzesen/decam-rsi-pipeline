"""TNO finder, gen 9d: orbit-space GPU shift-and-stack (synthetic tracking), no catalogue linking.  See NOTES.md.

1. mapper.py (worker processes): per CCD matched-filter S/N map, max-pooled 4x4 (~1.05").
2. All exposures resampled on the GPU onto one gnomonic sky grid (1"/px) about the field centre; normalised,
   static sources (S/N>4 on >= 2 nights at the same sky pixel) zeroed, clipped.
3. Search: choose a short window of >= 3 nights.  For each trial distance d, every map is warped into the
   heliocentric "d-frame" (direction of the point at distance d seen from the Sun), where a TNO moves almost linearly.
   Brute-force all linear velocities (|v| <= escape) on 2" maps: sum of shifted maps, per-pixel best.
4. Peaks -> heliocentric states -> CEM refinement of the 5-parameter state against ALL exposures (1" maps).
"""
import argparse, csv, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import orbit
from orbit import T0, log, left, GM, C_AU

ARCR = math.pi / 180 / 3600
EPS = math.radians(23.4392911)


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def n_workers(cuda):
    if os.environ.get("TNO_WORKERS"):
        return max(1, int(os.environ["TNO_WORKERS"]))
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:
        ncpu = os.cpu_count() or 4
    return int(max(2, min(7, ncpu - 1))) if cuda else int(max(1, min(4, ncpu)))


def tframe(u0):
    u0 = u0 / np.linalg.norm(u0)
    ex = np.cross([0, 0, 1.0], u0); ex /= np.linalg.norm(ex)
    ey = np.cross(u0, ex)
    return np.stack([ex, ey, u0])


def write_out(path, cands):
    cands = sorted(cands, key=lambda c: -c["p"])[:200]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"candidates": cands}, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=300)
    a = ap.parse_args()
    budget = a.budget_s
    write_out(a.out, [])
    import torch
    dev = device()
    cuda = dev == "cuda"
    F = torch.nn.functional
    sdir = a.scenario
    rows = list(csv.DictReader(open(os.path.join(sdir, "exposures.csv"))))
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    t_all = np.array([float(r["mjd_mid"]) for r in rows])
    night_all = np.floor(t_all - 0.5).astype(int)
    # choose exposures: at most MAXE, spread over nights
    # gen 11e: scale the caps with the budget actually given (in the hybrid the stack gets ~350 s on the A100,
    # but gen009d sized MAXE/HMAX for ~72 s).  TNO_SCALE overrides (dev testing).
    SCALE = float(os.environ.get("TNO_SCALE", "0") or 0) or min(4.0, max(1.0, budget / 150.0))
    if not cuda and not os.environ.get("TNO_SCALE"):
        SCALE = 1.0
    MAXE = int((64 if SCALE >= 2 else 48) if cuda else (32 if SCALE >= 2 else 24))
    # gen 13e: dense scenarios (more exposures than MAXE) -> frames = co-adds of up to GMAX consecutive same-night
    # exposures within GSPAN days (a TNO moves <~1.5" in that time), so the MAXE device frames carry more of the data.
    GSPAN = float(os.environ.get("TNO_GSPAN", "0.014"))
    MAXR = int(os.environ.get("TNO_MAXR", "0") or 0) or (192 if cuda else 2 * MAXE)
    group = (len(rows) > MAXE or os.environ.get("TNO_GROUP") == "1") and os.environ.get("TNO_GROUP") != "0"
    GMAX = int(max(1, min(6, math.ceil(min(MAXR, len(rows)) / MAXE)))) if group else 1
    if group and os.environ.get("TNO_GROUP") == "1":
        GMAX = max(GMAX, 2)
    groups = []                       # list of lists of exposure indices (time-sorted, one night)
    for u in np.unique(night_all):
        ii = sorted(np.flatnonzero(night_all == u), key=lambda i: t_all[i])
        cur = []
        for i in ii:
            if cur and (len(cur) >= GMAX or t_all[i] - t_all[cur[0]] > GSPAN):
                groups.append(cur); cur = []
            cur.append(int(i))
        if cur:
            groups.append(cur)
    g_t = np.array([t_all[g].mean() for g in groups])
    g_night = np.floor(g_t - 0.5).astype(int)
    un = np.unique(g_night)
    per = [sorted(np.flatnonzero(g_night == u), key=lambda i: g_t[i]) for u in un]
    order = []
    k = 0
    while len(order) < min(MAXE, len(groups)):
        for p in per:
            if k < len(p) and len(order) < MAXE:
                # spread within the night: take indices from the ends/middle alternately
                idx = [0, len(p) - 1, len(p) // 2][k] if k < 3 and len(p) > 2 else k
                j = p[idx] if idx < len(p) else None
                if j is not None and j not in order:
                    order.append(j)
                else:
                    rest = [x for x in p if x not in order]
                    if rest:
                        order.append(rest[len(rest) // 2])
        k += 1
        if k > len(groups) + 3:
            break
    nraw = sum(len(groups[g]) for g in order)
    log(f"frames: {len(groups)} groups (GMAX {GMAX}, span {GSPAN} d), {len(order)} chosen covering {nraw} exposures")
    log(f"device {dev}; {len(rows)} exposures, using {len(order)} frames; nights {len(un)}; scale {SCALE:.2f}")

    # ---------------------------------------------------------------- 1. maps (worker processes)
    import mapper
    from multiprocessing import get_context
    nw = n_workers(cuda)
    ctx = get_context("spawn")
    pool = ctx.Pool(nw)
    # first member of every chosen frame first, then second members, ... (a map deadline keeps frame coverage)
    tord = []
    for m_ in range(GMAX):
        for g in order:
            if m_ < len(groups[g]):
                tord.append(groups[g][m_])
    tasks = [(int(i), os.path.join(sdir, rows[i]["image"]), os.path.join(sdir, rows[i]["mask"])) for i in tord]
    res = {}
    deadline = budget * 0.40
    it = pool.imap_unordered(mapper.map_exposure, tasks)
    while True:
        try:
            r = it.next(timeout=max(1.0, deadline - (time.time() - T0)))
        except StopIteration:
            break
        except Exception:
            log("map deadline reached")
            break
        if r["maps"]:
            res[r["ei"]] = r
        if time.time() - T0 > deadline:
            log("map deadline reached")
            break
    pool.terminate()
    # frames that got at least one mapped member; frame time / observer = mean over its mapped members
    fm = [[i for i in groups[g] if i in res] for g in order]
    fm = sorted([m for m in fm if m], key=lambda m: t_all[m].mean())
    log(f"mapped {len(res)} exposures into {len(fm)} frames")
    if len(fm) < 3:
        return
    obs_all = np.array([[float(r_["obs_x"]), float(r_["obs_y"]), float(r_["obs_z"])] for r_ in rows])
    t = np.array([t_all[m].mean() for m in fm])
    night = np.floor(t - 0.5).astype(int)
    obs = np.array([obs_all[m].mean(0) for m in fm])
    N = len(fm)
    use = [i for m in fm for i in m]
    # gen 19: per-frame depth weights (expected S/N of a fixed-magnitude point source).  TNO_DWEIGHT=0 -> all 1.
    DW = os.environ.get("TNO_DWEIGHT", "1") != "0"
    w_exp = {i: 1.0 for i in use}
    wf = np.ones(N)
    if DW:
        try:
            w_exp = depth_weights(rows, res, use)
            # frame = sum_i w_i z_i / sqrt(sum_i w_i^2) (weighted co-add below) -> frame S/N ~ sqrt(sum_i w_i^2)
            wf = np.array([math.sqrt(sum(w_exp[i] ** 2 for i in m)) for m in fm])
            wf = np.clip(wf / np.median(wf), 0.25, 2.5)
            log("depth weights per exposure:", " ".join(f"{rows[i]['band']}{w_exp[i]:.2f}" for i in sorted(use)))
            log("depth weights per frame:", " ".join(f"{x:.2f}" for x in wf))
        except Exception as ex:
            log("depth weights failed, using uniform:", repr(ex)[:200])
            w_exp = {i: 1.0 for i in use}
            wf = np.ones(N)

    # ---------------------------------------------------------------- 2. sky grid
    ra0, dec0 = math.radians(float(meta["ra"])), math.radians(float(meta["dec"]))
    TF0 = orbit.tangent_frame(ra0, dec0)
    PIX = 1.0 * ARCR
    xs, ys = [], []
    for i in use:
        for g, _ in res[i]["maps"]:
            u = orbit.radec_to_unit(g["rc"], g["dc"])
            p = TF0 @ u
            xs += [p[0] / p[2] - g["rad"], p[0] / p[2] + g["rad"]]
            ys += [p[1] / p[2] - g["rad"], p[1] / p[2] + g["rad"]]
    HALF = 4400 * PIX
    x0, x1 = max(min(xs), -HALF), min(max(xs), HALF)
    y0, y1 = max(min(ys), -HALF), min(max(ys), HALF)
    W = int((x1 - x0) / PIX) + 1
    H = int((y1 - y0) / PIX) + 1
    W += W % 2; H += H % 2
    log(f"sky grid {W}x{H} at 1\"")
    TF0_t = torch.tensor(TF0, dtype=torch.float32, device=dev)
    maps = torch.full((N, H, W), float("nan"), dtype=torch.float16, device=dev)
    def norm_(m):
        s = m[::7, ::7]
        s = s[torch.isfinite(s)].float()
        if s.numel() < 1000:
            return None
        med = torch.median(s); sd = 1.4826 * torch.median(torch.abs(s - med))
        return ((m.float() - med) / torch.clamp(sd, min=1e-3))
    for n, mem in enumerate(fm):
      if len(mem) > 1:
        acc = torch.zeros((H, W), dtype=torch.float32, device=dev)
        cnt = torch.zeros((H, W), dtype=torch.float32, device=dev)
      for i in mem:
        if len(mem) > 1:
            maps[n].fill_(float("nan"))
        for g, pm in res[i]["maps"]:
            u = orbit.radec_to_unit(g["rc"], g["dc"]); p = TF0 @ u
            cx, cy = p[0] / p[2], p[1] / p[2]
            ix0 = max(0, int((cx - g["rad"] - x0) / PIX)); ix1 = min(W, int((cx + g["rad"] - x0) / PIX) + 1)
            iy0 = max(0, int((cy - g["rad"] - y0) / PIX)); iy1 = min(H, int((cy + g["rad"] - y0) / PIX) + 1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            gx = x0 + PIX * torch.arange(ix0, ix1, device=dev, dtype=torch.float32)
            gy = y0 + PIX * torch.arange(iy0, iy1, device=dev, dtype=torch.float32)
            XI, ETA = gx[None, :].expand(len(gy), -1), gy[:, None].expand(-1, len(gx))
            U = TF0_t[2][None, None] + XI[..., None] * TF0_t[0][None, None] + ETA[..., None] * TF0_t[1][None, None]
            TL = torch.tensor(orbit.tangent_frame(g["rc"], g["dc"]), dtype=torch.float32, device=dev)
            q = U @ TL.T
            lx, ly = q[..., 0] / q[..., 2] * 1e3, q[..., 1] / q[..., 2] * 1e3
            A = torch.stack([torch.ones_like(lx), lx, ly, lx * lx, lx * ly, ly * ly], -1)
            px = A @ torch.tensor(g["cx"], dtype=torch.float32, device=dev)
            py = A @ torch.tensor(g["cy"], dtype=torch.float32, device=dev)
            ph, pw = pm.shape
            ux = (px - 1.5) / 4; uy = (py - 1.5) / 4
            inside = (ux > -0.5) & (ux < pw - 0.5) & (uy > -0.5) & (uy < ph - 0.5)
            if not bool(inside.any()):
                continue
            grid = torch.stack([2 * ux / max(pw - 1, 1) - 1, 2 * uy / max(ph - 1, 1) - 1], -1)[None]
            src = torch.from_numpy(pm.astype(np.float32)).to(dev)[None, None]
            v = F.grid_sample(src, grid, mode="nearest", align_corners=True)[0, 0]
            sub = maps[n, iy0:iy1, ix0:ix1]
            take = inside & torch.isfinite(v)
            maps[n, iy0:iy1, ix0:ix1] = torch.where(take, v.to(torch.float16), sub)
        res[i]["maps"] = None
        if len(mem) > 1:
            z = norm_(maps[n])
            if z is not None:
                f_ = torch.isfinite(z)
                wi_ = w_exp[i] if DW else 1.0
                acc += torch.where(f_, z * wi_, torch.zeros_like(z)); cnt += f_.float() * (wi_ * wi_)
                del z, f_
      if len(mem) > 1:
        maps[n] = torch.where(cnt > 0, acc / torch.sqrt(torch.clamp(cnt, min=1)),
                              torch.full_like(acc, float("nan"))).to(torch.float16)
        del acc, cnt
    del res
    # normalise each map, coverage
    cov = torch.isfinite(maps)
    for n in range(N):
        m = maps[n]
        s = m[::7, ::7][cov[n, ::7, ::7]].float()
        if s.numel() < 1000:
            cov[n] = False
            continue
        med = torch.median(s); sd = 1.4826 * torch.median(torch.abs(s - med))
        maps[n] = ((m.float() - med) / torch.clamp(sd, min=1e-3)).to(torch.float16)
    maps = torch.where(cov, maps, torch.zeros_like(maps))
    # static sources: > 4 on >= 2 nights (dilated by 1 px)
    nn = torch.zeros((H, W), dtype=torch.int16, device=dev)
    # gen 20: bins = clusters of exposures separated by gaps > 0.06 d (was fixed 0.06-d boundaries, which split one
    # night's exposures into two bins and so marked every moving body static in that night; found by injection).
    if os.environ.get("TNO_OLDSTATIC") == "1":
        tbin = np.floor((t - t.min()) / 0.06).astype(int)
    else:
        o_t = np.argsort(t)
        tbin = np.empty(len(t), int)
        tbin[o_t] = np.r_[0, np.cumsum(np.diff(t[o_t]) > 0.06)]
    for u_ in np.unique(tbin):
        idx = np.flatnonzero(tbin == u_)
        b = torch.zeros((H, W), dtype=torch.bool, device=dev)
        for n in idx:
            b |= maps[n] > 4
        b = F.max_pool2d(b[None, None].half(), 3, 1, 1)[0, 0] > 0
        nn += b.to(torch.int16)
    # gen 20: TNO_SNEED (default 2 = old).  3 (with >= 4 bins) keeps faint movers from being zeroed by one noise
    # excursion in another bin, but measured no gain on the injected test (A 11/15 vs 12/15, B 0 either way).
    nbin_ = len(np.unique(tbin))
    SNEED = int(os.environ.get("TNO_SNEED", "2") or 2)
    static = nn >= SNEED
    log(f"static fraction {float(static.float().mean()):.4f} ({nbin_} time bins, need {SNEED})")
    maps = torch.where(static[None], torch.zeros_like(maps), maps).clamp_(-3, 5)
    del nn, static
    write_out(a.out, [])

    # ---------------------------------------------------------------- 3. search window
    nights_u = np.unique(night)
    tn = np.array([t[night == u].mean() for u in nights_u])
    SPOOL = os.environ.get("TNO_SPOOL", "max")
    HMAX = int((60000 if cuda else 12000) * SCALE)
    SP = 2.0 * ARCR        # search pixel
    VRAD = 0.75            # velocity disc radius around prograde circular, in units of v_circ

    def window_cost(sel):
        ts = t[sel]; tr = 0.5 * (ts.min() + ts.max()); mdt = max(ts.max() - tr, 0.05)
        dv = 1.0 / mdt               # search px / day
        tot = 0
        for d in dgrid(sel, tr):
            vmax = math.sqrt(2 * GM / d) / d / SP
            tot += math.pi * (min(vmax, VRAD * math.sqrt(GM / d) / d / SP) / dv + 1) ** 2
        return tot

    def dgrid(sel, tr):
        o = obs[sel]; ts = t[sel] - tr
        A = np.stack([np.ones_like(ts), ts], 1)
        coef, *_ = np.linalg.lstsq(A, o, rcond=None)
        r = o - A @ coef
        r = r - np.outer(r @ TF0[2], TF0[2])
        rmax = max(np.sqrt((r ** 2).sum(1)).max(), 1e-6)
        step = 1.0 * SP / rmax
        lo, hi = 1 / 120.0, 1 / 26.0
        nst = max(1, int(math.ceil((hi - lo) / step)))
        g = [1 / (lo + (k + 0.5) * (hi - lo) / nst) for k in range(nst)]
        if not g:
            g = [40.0]
        # prior order: classical belt first
        return sorted(g, key=lambda d: abs(math.log(d / 42.0)))

    # gen 19: up to NPN frames per night in the window search (was 3, which dropped most of a dense night's depth;
    # frames are already co-adds of <= GMAX exposures).  Default 8 on CUDA, 3 elsewhere; TNO_NPN overrides.
    NPN = max(1, int(os.environ.get("TNO_NPN", "0") or 0) or (8 if cuda else 3))
    # gen 20: peaks kept per (window, distance): was a global top-400 (TNO_NPEAK), optionally per spatial tile
    # gen 29: on CUDA keep more, slightly fainter window peaks (800 per distance, threshold 4.2 instead of 400 / 4.5)
    # and refine 2x more seeds (XSEED 6): the A100 run left ~40 % of the budget idle and refinement is deadline-bounded.
    NPEAK = int(os.environ.get("TNO_NPEAK", "0") or 0) or (800 if cuda else 400)
    PTHR = float(os.environ.get("TNO_PTHR", "0") or 0) or (4.2 if cuda else 4.5)
    PTILE = int(os.environ.get("TNO_PTILE", "0") or 0) or 1
    wins = []
    for a_ in range(len(nights_u)):
        for b_ in range(a_ + 2, len(nights_u)):
            nsel = nights_u[a_:b_ + 1]
            # at most NPN frames per night (spread)
            sel = []
            for u in nsel:
                idx = np.flatnonzero(night == u)
                if len(idx) > NPN:
                    if NPN == 3:
                        idx = idx[[0, len(idx) // 2, len(idx) - 1]]
                    else:
                        idx = idx[np.unique(np.round(np.linspace(0, len(idx) - 1, NPN)).astype(int))]
                sel += list(idx)
            sel = np.array(sel)
            c = window_cost(sel)
            key = (b_ - a_ + 1 if c <= HMAX else 0, -c)
            wins.append((key, a_, b_, sel))
    wins.sort(key=lambda w: w[0], reverse=True)
    log("windows (nights, cost):", [(w[2] - w[1] + 1, int(-w[0][1])) for w in wins[:12]])
    # gen 11e: greedy list of windows: the best one first, then further windows (within HMAX) that share at
    # most one night with every window already chosen.  Searched in this order while time remains.
    chosen = []
    for w in wins:
        if chosen and -w[0][1] > 3 * HMAX:      # later windows may exceed HMAX (searched partially,
            continue                             # most probable orbits first) but not by much
        if all(len(set(range(w[1], w[2] + 1)) & set(range(c_[1], c_[2] + 1))) <= 1 for c_ in chosen):
            chosen.append(w)
        if len(chosen) >= 6:
            break
    if not chosen:
        log("no search window")
        return
    best_all = []          # list of (score, d, vx, vy, px, py, frame index) in d-frame coords
    frames = []            # (TD, tref, oref) per (window, distance)
    wi_of = []
    search_deadline = budget * 0.72
    nhyp = 0
    nfa = 0          # gen 19: frame-adds (hypotheses x frames) -- the real cost unit with NPN > 3
    z0 = TF0[2]
    pole = np.array([0, -math.sin(EPS), math.cos(EPS)])
    # gen 54: night seeds BEFORE the window search (TNO_NS_EARLY=1), on their own time slice (TNO_NS_EF x budget),
    # so a larger night-peak cap can finish all distances; the results are used where the late call used to run.
    ns_pre = None
    if os.environ.get("TNO_NIGHTSEED", "1") != "0" and os.environ.get("TNO_NS_EARLY", "0") == "1":
        try:
            NSN_ = int(os.environ.get("TNO_NS_N", "0") or 0) or (8000 if cuda else 1500)
            ns_dl_ = (time.time() - T0) + float(os.environ.get("TNO_NS_EF", "0.25")) * budget
            ns_pre = night_seeds(torch, F, dev, maps, cov, wf, t, night, obs, TF0, x0, y0, PIX, W, H,
                                 ns_dl_, NSN_, log)
            log(f"early night seeds: {len(ns_pre[0])} rows, {time.time() - T0:.0f}s elapsed")
        except Exception as ex:
            log("early night seeds failed:", ex)
            ns_pre = None
    t_search0 = time.time()
    try:
        xwin_left = [int(os.environ.get("TNO_XWIN", "6") or 0)]
        XWIN_SAFE = float(os.environ.get("TNO_XWIN_SAFE", "0.5"))
    except Exception:
        xwin_left, XWIN_SAFE = [0], 1.0
    xwin_ids = set()
    # gen 33: deep all-nights window (see NOTES.md).  After the planned windows, one window holding EVERY mapped
    # frame is searched with a classical-belt prior (|v - v_circ| <= DEEP_VRAD v_circ, DEEP_DMIN..DEEP_DMAX au) on
    # night-collapsed d-frame co-adds (frames within DEEP_GSPAN d summed after the d-frame warp, so the cost scales
    # with nights, not exposures).  Seeds weight-4-like bodies that only show in the all-exposure stack.
    DEEP = os.environ.get("TNO_DEEP", "1") != "0"
    DEEP_VRAD = float(os.environ.get("TNO_DEEP_VRAD", "0.3"))
    DEEP_DMIN = float(os.environ.get("TNO_DEEP_DMIN", "34")); DEEP_DMAX = float(os.environ.get("TNO_DEEP_DMAX", "52"))
    DEEP_GSPAN = float(os.environ.get("TNO_DEEP_GSPAN", "0.08"))
    VCON = os.environ.get("TNO_VCON", "1") != "0"           # gen 35: velocity-space contrast in the deep window
    DCLIP = float(os.environ.get("TNO_DCLIP", "4.0") or 0)   # gen 48: deep-window per-frame S/N winsorizing
    LMO = os.environ.get("TNO_LMO", "1") != "0"             # gen 36: leave-max-out co-add stack in the deep window
    deep_ids = set()
    deep_tn = [0.0, 0]            # time and frame-adds spent in the deep window (excluded from the extra-window rate)
    deep_done = [not DEEP or len(np.unique(night)) < 3]
    # gen 46: further deep windows with wider priors (vrad:dmin:dmax, comma-separated), queued one after another
    # while their estimated cost fits the time left before search_deadline (the A100 otherwise idles ~40 %).
    DEEP_MORE = []
    for tok_ in os.environ.get("TNO_DEEP2", "0.55:28:60,0.3:60:110").split(","):
        try:
            v_, a_, b_ = (float(x_) for x_ in tok_.split(":"))
            DEEP_MORE.append((v_, a_, b_))
        except Exception:
            pass
    deep_prior = {}

    def dgrid_deep(sel, tr):
        o = obs[sel]; ts = t[sel] - tr
        A = np.stack([np.ones_like(ts), ts], 1)
        coef, *_ = np.linalg.lstsq(A, o, rcond=None)
        r = o - A @ coef
        r = r - np.outer(r @ TF0[2], TF0[2])
        rmax = max(np.sqrt((r ** 2).sum(1)).max(), 1e-6)
        step = 1.0 * SP / rmax
        lo, hi = 1 / DEEP_DMAX, 1 / DEEP_DMIN
        nst = max(1, int(math.ceil((hi - lo) / step)))
        g = [1 / (lo + (k + 0.5) * (hi - lo) / nst) for k in range(nst)]
        return sorted(g, key=lambda d: abs(math.log(d / 42.0)))

    def deep_cost(sel):
        # frame-adds of a deep window: hypotheses (restricted disc, deep distances) x collapsed co-adds
        ts = t[sel]; tr = 0.5 * (ts.min() + ts.max()); mdt = max(ts.max() - tr, 0.05)
        dv = 1.0 / mdt
        hy = 0.0
        for d in dgrid_deep(sel, tr):
            vcs = math.sqrt(GM / d) / d / SP
            hy += math.pi * (DEEP_VRAD * vcs / dv + 1) ** 2
        ts = np.sort(ts); ng = 1; c0 = ts[0]
        for x in ts[1:]:
            if x - c0 > DEEP_GSPAN:
                ng += 1; c0 = x
        return hy * ng, hy

    def deep_pick(avail, rate):
        # contiguous night range: most nights whose estimated cost fits the time left, then most depth (sum wf^2)
        best_ = None
        for a_ in range(len(nights_u)):
            for b_ in range(a_ + 2, len(nights_u)):
                sel_ = np.flatnonzero((night >= nights_u[a_]) & (night <= nights_u[b_]))
                c_, hy_ = deep_cost(sel_)
                fits = c_ * rate <= avail or (os.environ.get("TNO_DEEP_ALL") == "1" and a_ == 0 and b_ == len(nights_u) - 1)
                key_ = (fits, (b_ - a_ + 1) if fits else 0, float((wf[sel_] ** 2).sum()) if fits else -c_)
                if best_ is None or key_ > best_[0]:
                    best_ = (key_, a_, b_, sel_, c_, hy_)
        return best_

    for wi, w in enumerate(chosen):
      el_ = time.time() - T0
      if wi > 0:
          # time left must cover this window's estimated cost (scaled from the first window's rate)
          rate = (time.time() - t_search0 - deep_tn[0]) / max(nfa - deep_tn[1], 1)
          if wi not in xwin_ids and el_ + rate * window_cost(w[3]) * len(w[3]) * 0.5 > search_deadline and not os.environ.get("TNO_WINFORCE"):
              log(f"skip window {wi}: not enough time")
              continue
      sel = w[3]
      ts = t[sel]; tref = 0.5 * (ts.min() + ts.max()); mdt = max(ts.max() - tref, 0.05)
      dv = 1.0 / mdt
      if wi in deep_ids:
          DEEP_VRAD, DEEP_DMIN, DEEP_DMAX = deep_prior[wi]
          log(f"deep prior: vrad {DEEP_VRAD}, {DEEP_DMIN}-{DEEP_DMAX} au")
      dg = dgrid_deep(sel, tref) if wi in deep_ids else dgrid(sel, tref)
      log(f"window {wi}: {len(np.unique(night[sel]))} nights, {len(sel)} exposures, span {ts.max() - ts.min():.2f} d, "
          f"est hyps {window_cost(sel):.0f}, {len(dg)} distances")
      # review fix: chunked (identical result; max_pool2d is per channel) so a many-exposure extra window does not
      # materialise a full float32 copy of maps[sel] (up to ~20 GB on a large field); OOM here ends only an extra window.
      smaps = None                         # free the previous window's maps first
      try:
          if SPOOL == "avg":
              # gen 18: mean-pool to the 2" search grid, then re-normalise each frame (median/MAD of covered px)
              sm_ = []
              for i_ in range(0, len(sel), 8):
                  a_ = F.avg_pool2d(maps[sel[i_:i_ + 8]].float()[None], 2, 2)[0]
                  for j_ in range(a_.shape[0]):
                      s_ = a_[j_, ::5, ::5]; s_ = s_[s_ != 0]
                      if s_.numel() > 1000:
                          med_ = torch.median(s_); sd_ = 1.4826 * torch.median(torch.abs(s_ - med_))
                          a_[j_] = torch.where(a_[j_] != 0, (a_[j_] - med_) / torch.clamp(sd_, min=1e-3), a_[j_])
                  sm_.append(a_.clamp_(-3, 5).half())
              smaps = torch.cat(sm_, 0)
          else:
              smaps = torch.cat([F.max_pool2d(maps[sel[i_:i_ + 8]].float()[None], 2, 2)[0].half()
                                 for i_ in range(0, len(sel), 8)], 0)       # (n, H2, W2)
      except Exception as ex:
          if wi not in xwin_ids:
              raise
          log(f"extra window {wi} setup failed:", repr(ex)[:200])
          smaps = None
          try:
              if cuda:
                  torch.cuda.empty_cache()
          except Exception:
              pass
          if wi not in deep_ids:
              continue
          dg = []                            # deep window: skip its search but still reach the extra-window block
      H2, W2 = smaps.shape[1:] if smaps is not None else (0, 0)
      t_win0, nfa_win0 = time.time(), nfa
      oref = obs[sel].mean(0)
      if wi == 0:
          tref0, oref0, dv0 = tref, oref, dv
      for di, d in enumerate(dg):
        try:
              if time.time() - T0 > search_deadline:
                  break
              # d-frame centre: heliocentric direction of the field centre at distance d
              b = oref @ z0; rho = -b + math.sqrt(b * b - oref @ oref + d * d)
              w0 = (oref + rho * z0) / d
              TD = tframe(w0)
              vmax = math.sqrt(2 * GM / d) / d / SP
              # margins: where does the sky centre fall in the d-frame for each exposure
              offs = []
              for n in sel:
                  o = obs[n]; b = o @ z0; rr = -b + math.sqrt(b * b - o @ o + d * d)
                  q = TD @ ((o + rr * z0) / d)
                  offs.append([q[0] / q[2] / SP, q[1] / q[2] / SP])
              offs = np.array(offs)
              mx = int(np.abs(offs).max() + vmax * mdt + 8)
              HD, WD = H2 + 2 * mx, W2 + 2 * mx
              # d-frame maps
              TD_t = torch.tensor(TD, dtype=torch.float32, device=dev)
              gxd = (torch.arange(WD, device=dev, dtype=torch.float32) - (WD - 1) / 2) * SP
              gyd = (torch.arange(HD, device=dev, dtype=torch.float32) - (HD - 1) / 2) * SP
              wsel = wf[sel]
              dts = t[sel] - tref
              gid_ = np.arange(len(sel))
              if wi in deep_ids:
                  # gen 33: collapse consecutive frames (time-sorted) within DEEP_GSPAN d into one d-frame co-add at
                  # their weighted mean time; residual smear <= v * span / 2 (< 1" for v <= 20"/d at span 0.08 d)
                  o_t = np.argsort(dts)
                  grp_, cur_ = [], [int(o_t[0])]
                  for j_ in o_t[1:]:
                      if dts[j_] - dts[cur_[0]] > DEEP_GSPAN:
                          grp_.append(cur_); cur_ = []
                      cur_.append(int(j_))
                  grp_.append(cur_)
                  dts_c = np.zeros(len(grp_))
                  for g_, mm_ in enumerate(grp_):
                      gid_[mm_] = g_
                      ww_ = np.asarray(wsel, dtype=float)[mm_] ** 2 if DW else np.ones(len(mm_))
                      dts_c[g_] = float((dts[mm_] * ww_).sum() / max(ww_.sum(), 1e-9))
                  dts = dts_c
                  if di == 0:
                      log(f"deep window: {len(sel)} frames collapsed to {len(grp_)} co-adds")
              Dm = torch.zeros((int(gid_.max()) + 1, HD, WD), dtype=torch.float16, device=dev)
              for k_, n in enumerate(sel):
                  rows_ = []
                  src_ = smaps[k_].float()
                  if wi in deep_ids and DCLIP > 0:
                      # gen 48: winsorize each frame's S/N map at +-DCLIP before the deep co-add, so single-frame
                      # outliers (CRs, star residuals, satellite trails) cannot build the 11-16 sigma noise maxima;
                      # a weight-4 body has per-frame S/N ~1-2 and is untouched.
                      src_ = src_.clamp(-DCLIP, DCLIP)
                  for y0_ in range(0, HD, 1024):
                      gy_ = gyd[y0_:y0_ + 1024]
                      Wd = TD_t[2][None, None] + gxd[None, :, None] * TD_t[0][None, None] + gy_[:, None, None] * TD_t[1][None, None]
                      Wd = Wd / torch.linalg.norm(Wd, dim=-1, keepdim=True)
                      X = d * Wd - torch.tensor(obs[n] - 0.0, dtype=torch.float32, device=dev)
                      q = X @ TF0_t.T
                      sx = q[..., 0] / q[..., 2]; sy = q[..., 1] / q[..., 2]
                      ux = (sx - x0) / (2 * PIX) - 0.25; uy = (sy - y0) / (2 * PIX) - 0.25
                      grid = torch.stack([2 * ux / (W2 - 1) - 1, 2 * uy / (H2 - 1) - 1], -1)[None]
                      rows_.append(F.grid_sample(src_[None, None], grid, mode="nearest",
                                                 padding_mode="zeros", align_corners=True)[0, 0].half())
                  fr_ = torch.cat(rows_, 0)
                  if DW:
                      fr_ *= float(wsel[k_])
                  Dm[int(gid_[k_])] += fr_
                  del fr_
              # velocities: grid in search px/day; ordered by closeness to prograde circular
              vc = math.sqrt(GM / d) * np.cross(pole, w0); vc = vc / np.linalg.norm(vc) * math.sqrt(GM / d)
              vcx, vcy = (TD[0] @ vc) / d / SP, (TD[1] @ vc) / d / SP
              nv = int(vmax / dv) + 1
              vv = np.arange(-nv, nv + 1) * dv
              VX, VY = np.meshgrid(vv, vv)
              vcs = math.hypot(vcx, vcy)
              ok = (VX ** 2 + VY ** 2 <= (vmax + dv) ** 2) & ((VX - vcx) ** 2 + (VY - vcy) ** 2 <= ((DEEP_VRAD if wi in deep_ids else VRAD) * vcs + dv) ** 2)
              VX, VY = VX[ok], VY[ok]
              o_ = np.argsort((VX - vcx) ** 2 + (VY - vcy) ** 2)
              VX, VY = VX[o_], VY[o_]
              P = mx
              Hc, Wc = HD - 2 * P, WD - 2 * P
              best = torch.full((Hc, Wc), -1e4, dtype=torch.float16, device=dev)
              bidx = torch.zeros((Hc, Wc), dtype=torch.int32, device=dev)
              norm = 1.0 / math.sqrt(float((wsel ** 2).sum())) if DW else 1.0 / math.sqrt(len(sel))
              t1 = time.time()
              # gen 35: velocity-space contrast (deep window only, TNO_VCON).  Per-pixel mean/std of the stack
              # over ALL velocity hypotheses; the peak score becomes (best - mean) / std.  A real body lights one
              # velocity; pixel-coherent artefacts (residual static sources, halos, edges) light many and are damped.
              vcon_ = wi in deep_ids and VCON and Dm.shape[0] >= 3
              if vcon_:
                  s1_ = torch.zeros((Hc, Wc), dtype=torch.float32, device=dev)
                  s2_ = torch.zeros((Hc, Wc), dtype=torch.float32, device=dev)
                  nacc_ = 0
              # gen 36: leave-max-out (deep window only, TNO_LMO, needs VCON).  Each hypothesis' stack is the sum of
              # its co-adds MINUS the largest single co-add term, so one co-add outlier (cosmic ray, fast mover,
              # satellite residue: a one-velocity signal VCON cannot damp) contributes nothing, while a real body
              # spread over K co-adds keeps (K-1)/K of its sum.  Co-adds are first shifted to zero median on covered
              # pixels (max-pooled S/N maps are biased positive), so the max term is not just the most-frames co-add.
              lmo_ = vcon_ and LMO and Dm.shape[0] >= 4
              if lmo_:
                  for k_ in range(Dm.shape[0]):
                      v_ = Dm[k_, ::3, ::3].float(); v_ = v_[v_ != 0]
                      if v_.numel() > 100:
                          m_ = float(torch.median(v_))
                          Dm[k_] = torch.where(Dm[k_] != 0, Dm[k_] - m_, Dm[k_])
                      del v_
              for h in range(len(VX)):
                  if (h & 63) == 0 and time.time() - T0 > search_deadline:
                      break
                  acc = None
                  mxs_ = None
                  for k_ in range(Dm.shape[0]):
                      sx = int(round(VX[h] * dts[k_] - offs[k_, 0] * 0)); sy = int(round(VY[h] * dts[k_]))
                      sl = Dm[k_, P + sy:P + sy + Hc, P + sx:P + sx + Wc]
                      acc = sl.clone() if acc is None else acc.add_(sl)
                      if lmo_:
                          mxs_ = sl.clone() if mxs_ is None else torch.maximum(mxs_, sl, out=mxs_)
                  if lmo_:
                      acc.sub_(mxs_)
                  if vcon_:
                      af_ = acc.float(); s1_ += af_; s2_.addcmul_(af_, af_); del af_; nacc_ += 1
                  better = acc > best
                  best = torch.where(better, acc, best)
                  bidx = torch.where(better, torch.full_like(bidx, h), bidx)
                  nhyp += 1
                  nfa += Dm.shape[0]
              if dev == "mps":
                  torch.mps.synchronize()
              try:
                  if vcon_ and nacc_ < 31:
                      bs = torch.full((Hc, Wc), -1e4)      # too few hypotheses for a contrast score: no peaks
                  elif vcon_:
                      nh_ = float(nacc_)
                      mu_ = s1_ / nh_
                      sd_ = torch.sqrt(torch.clamp(s2_ / nh_ - mu_ * mu_, min=0))
                      # floor: median std over covered pixels (uncovered/edge pixels have ~0 spread)
                      cv_ = sd_[::4, ::4]; cv_ = cv_[cv_ > 0]
                      fl_ = float(torch.median(cv_)) * 0.5 if cv_.numel() > 100 else 1.0
                      zc_ = (best.float() - mu_) / torch.clamp(sd_, min=max(fl_, 1e-3))
                      # (not on the usual S/N scale; noise maxima ~3.5-4.5 for 1e3-1e4 velocities; PTHR still applies)
                      zc_ = torch.where(best > -1e3, zc_, torch.full_like(zc_, -1e4))
                      bs = zc_.cpu()
                      del mu_, sd_, zc_, s1_, s2_
                  else:
                      bs = (best.float() * norm).cpu()
                  bic = bidx.cpu()
                  pk = (bs == F.max_pool2d(bs[None, None], 9, 1, 4)[0, 0]) & (bs > PTHR)
                  yy, xx = torch.nonzero(pk, as_tuple=True)
                  sc = bs[yy, xx]
                  nsat = int((sc >= 0.98 * sc.max()).sum()) if sc.numel() else 0
                  if sc.numel() > NPEAK:
                      if PTILE > 1:
                          # gen 20: top-k per spatial tile (PTILE x PTILE), so bright/saturated regions cannot
                          # take every peak slot of this distance
                          tid = (yy * PTILE // Hc) * PTILE + (xx * PTILE // Wc)
                          kt = max(1, NPEAK // (PTILE * PTILE))
                          keep_ = []
                          for t_ in range(PTILE * PTILE):
                              ii_ = torch.nonzero(tid == t_, as_tuple=True)[0]
                              if ii_.numel() > kt:
                                  ii_ = ii_[torch.topk(sc[ii_], kt).indices]
                              keep_.append(ii_)
                          top = torch.cat(keep_)
                      else:
                          top = torch.topk(sc, NPEAK).indices
                      yy, xx, sc = yy[top], xx[top], sc[top]
                  hh = bic[yy, xx].numpy()
                  yy, xx, sc = yy.numpy(), xx.numpy(), sc.numpy()
              except Exception as ex:
                  log("peak extraction failed:", ex)
                  yy = xx = hh = sc = np.zeros(0)
              for j in range(len(sc)):
                  # d-frame pixel coords (centre of Dm grid at (WD-1)/2)
                  px = (xx[j] + P - (WD - 1) / 2) * SP; py = (yy[j] + P - (HD - 1) / 2) * SP
                  best_all.append((float(sc[j]), d, VX[hh[j]] * SP, VY[hh[j]] * SP, px, py, len(frames)))
              frames.append((TD, tref, oref)); wi_of.append(wi)
              log(f"d={d:.1f} au: {min(h + 1, len(VX))}/{len(VX)} velocities in {time.time() - t1:.1f}s, "
                  f"peaks {len(sc)}, max {sc.max() if len(sc) else 0:.2f}, near-max {nsat}")
              del Dm, best, bidx
        except Exception as ex:
          # gen 15A: an extra (time-driven) window may be far larger than the planned ones; on failure (e.g. GPU
          # out of memory) stop that window and keep what was found.  Planned windows keep the old behaviour.
          if wi not in xwin_ids:
              raise
          log(f"extra window {wi} failed at d={d:.1f}:", repr(ex)[:200])
          Dm = best = bidx = None
          try:
              if cuda:
                  torch.cuda.empty_cache()
          except Exception:
              pass
          break
      # gen 15A: time-driven extra windows.  When the planned windows are done and the measured rate predicts
      # that another window fits before search_deadline, append the deepest one (most nights) that fits and is
      # new: more nights than any searched window, or >= 2 nights not yet searched.  TNO_XWIN=0 disables.
      if wi in deep_ids:
          deep_tn[0] += time.time() - t_win0; deep_tn[1] += nfa - nfa_win0
      try:
          if wi == len(chosen) - 1 and not deep_done[0] and nhyp > 0:
              deep_done[0] = True
              avail = search_deadline - (time.time() - T0) - 10
              if avail > 5:
                  rate = (time.time() - t_search0) / max(nfa, 1)
                  pk_ = deep_pick(avail, rate * ((2.0 if LMO else 1.6) if VCON else 1.0))
                  if pk_ is not None:
                      deep_prior[len(chosen)] = (DEEP_VRAD, DEEP_DMIN, DEEP_DMAX)
                      deep_ids.add(len(chosen)); xwin_ids.add(len(chosen))
                      chosen.append(((0, 0), pk_[1], pk_[2], pk_[3]))
                      log(f"deep window queued: nights {pk_[1]}-{pk_[2]}, {len(pk_[3])} frames, est hyps {pk_[5]:.0f}, "
                          f"est {pk_[4] * rate:.0f}s (fits {pk_[0][0]}), {avail:.0f}s to search deadline")
          elif wi == len(chosen) - 1 and wi in deep_ids and DEEP_MORE and nhyp > 0:
              # gen 46: next wider-prior deep window, only if its estimated cost fits
              avail = search_deadline - (time.time() - T0) - 10
              rate = deep_tn[0] / max(deep_tn[1], 1)
              while DEEP_MORE and avail > 5:
                  DEEP_VRAD, DEEP_DMIN, DEEP_DMAX = DEEP_MORE.pop(0)
                  pk_ = deep_pick(avail, rate * 1.15)
                  nmax_ = max([len(c_[3]) for i_, c_ in enumerate(chosen) if i_ not in deep_ids] or [0])
                  if pk_ is not None and pk_[0][0] and len(pk_[3]) > max(nmax_, 6):
                      deep_prior[len(chosen)] = (DEEP_VRAD, DEEP_DMIN, DEEP_DMAX)
                      deep_ids.add(len(chosen)); xwin_ids.add(len(chosen))
                      chosen.append(((0, 0), pk_[1], pk_[2], pk_[3]))
                      log(f"deep window {len(deep_ids)} queued (vrad {DEEP_VRAD}, {DEEP_DMIN}-{DEEP_DMAX} au): "
                          f"nights {pk_[1]}-{pk_[2]}, {len(pk_[3])} frames, est hyps {pk_[5]:.0f}, "
                          f"est {pk_[4] * rate * 1.15:.0f}s, {avail:.0f}s to search deadline")
                      break
                  log(f"deep prior vrad {DEEP_VRAD} {DEEP_DMIN}-{DEEP_DMAX} au does not fit ({avail:.0f}s left)")
          if wi == len(chosen) - 1 and xwin_left[0] > 0 and nhyp > 0:
              el_ = time.time() - T0
              rate = (time.time() - t_search0 - deep_tn[0]) / max(nfa - deep_tn[1], 1)
              done = set((c_[1], c_[2]) for i_, c_ in enumerate(chosen) if i_ not in deep_ids)
              searched_n = set()
              cn_ = [c_ for i_, c_ in enumerate(chosen) if i_ not in deep_ids]
              for c_ in cn_:
                  searched_n |= set(range(c_[1], c_[2] + 1))
              maxn = max(c_[2] - c_[1] + 1 for c_ in cn_)
              pick = None
              for w2 in sorted(wins, key=lambda w_: (-(w_[2] - w_[1] + 1), -w_[0][1])):
                  if (w2[1], w2[2]) in done:
                      continue
                  nn2 = w2[2] - w2[1] + 1
                  new_n = len(set(range(w2[1], w2[2] + 1)) - searched_n)
                  if not (nn2 > maxn or new_n >= 2):
                      continue
                  # at least the first (most probable) distance of the window must fit; the search loops stop
                  # at search_deadline, so a partially searched window still covers the classical belt first.
                  ts2 = t[w2[3]]
                  nd2 = max(1, len(dgrid(w2[3], 0.5 * (ts2.min() + ts2.max()))))
                  if el_ + XWIN_SAFE * rate * (-w2[0][1]) * len(w2[3]) / nd2 + 10 > search_deadline:
                      continue
                  pick = w2
                  break
              if pick is not None:
                  xwin_left[0] -= 1
                  xwin_ids.add(len(chosen))
                  chosen.append(pick)
                  log(f"extra window (time-driven): nights {pick[1]}-{pick[2]}, est hyps {-pick[0][1]:.0f}, "
                      f"rate {rate * 1e3:.3f} ms/hyp, elapsed {el_:.0f}s of search deadline {search_deadline:.0f}s")
      except Exception as ex:
          log("extra window selection failed:", ex)
    log(f"search done: {nhyp} hypotheses, {len(best_all)} raw peaks")
    # gen 11e: seeds from several windows: round-robin by within-window rank (score as tie-break), so a later
    # window's seeds are not crowded out by the first window's noise peaks.
    NSEED = int(1500 * min(2.0, SCALE))
    # gen 15A: the refinement loop is deadline-bounded, so with time left on CUDA keep more seeds (round-robin order
    # unchanged; extra seeds are only refined if time remains).  TNO_XSEED=1 restores the old cap.
    try:
        XSEED = float(os.environ.get("TNO_XSEED", "0") or 0) or (6.0 if cuda else 1.0)
        NSEED = int(NSEED * max(1.0, XSEED))
    except Exception as ex:
        log("xseed failed:", ex)
    wof = [wi_of[r[6]] for r in best_all]
    rk = np.zeros(len(best_all))
    for wv_ in set(wof):
        ii = [j for j in range(len(best_all)) if wof[j] == wv_]
        ii.sort(key=lambda j: -best_all[j][0])
        # gen 46: the wider-prior deep windows (after the first deep one) get seed slots at 1/3 the rate
        slow_ = 3.0 if (wv_ in deep_ids and wv_ != min(deep_ids)) else 1.0
        for r_, j in enumerate(ii):
            rk[j] = r_ * slow_
    oo = sorted(range(len(best_all)), key=lambda j: (rk[j], -best_all[j][0]))[:NSEED]
    best_all = [best_all[j] for j in oo]
    log(f"seeds: {len(best_all)} from windows {sorted(set(wof))}")
    # gen 36: deep-window seeds are kind 3, kept OUT of the p null (TNO_DNULL=0: gen035 behaviour, they count as 2).
    # The deep window's prior-restricted, differently scored peaks refine to a noise population of their own; mixed
    # into the null they made p depend on what the deep window does (measured: null MAD 0.98 -> 1.36 with LMO).
    sdeep = np.array([wi_of[r[6]] in deep_ids for r in best_all], bool) if os.environ.get("TNO_DNULL", "1") != "0" \
        else np.zeros(len(best_all), bool)

    # ---------------------------------------------------------------- 4. states, refinement on all exposures
    def to_state(r):
        sc, d, vx, vy, px, py, fi = r
        TD, tref, oref = frames[fi]
        wv = TD[2] + px * TD[0] + py * TD[1]
        wn = np.linalg.norm(wv); w = wv / wn
        X = d * w
        V = d * (vx * TD[0] + vy * TD[1]) / wn
        V = V - (V @ w) * w
        lt = np.linalg.norm(X - oref) / C_AU
        return np.r_[X, V], tref - lt

    t_ep = tref0
    tref, oref, dv = tref0, oref0, dv0
    states = []
    for r in best_all:
        s, ep = to_state(r)
        # move state to common epoch t_ep (two-body)
        x = orbit.propagate(s[:3], s[3:], np.array([t_ep - ep, t_ep - ep + 1e-3]))
        states.append(np.r_[x[0], (x[1] - x[0]) / 1e-3])
    states = np.array(states) if states else np.zeros((0, 6))
    seed_sc = -np.arange(len(best_all), dtype=float)     # refinement order = round-robin order above
    # gen 12e: linker candidates (wrapper passes their OUT file in TNO_SEEDS) are refined first, on all exposures,
    # so the wrapper's merge can confirm (boost) them with pixel-level evidence or correct their orbits.
    Ls = []
    try:
        sf = os.environ.get("TNO_SEEDS")
        if sf and os.path.exists(sf):
            for c_ in json.load(open(sf))["candidates"][:int(os.environ.get("TNO_NSEEDL", "200"))]:
                s_ = np.array(c_["state"], float)
                if s_.shape != (6,) or not np.all(np.isfinite(s_)) or np.linalg.norm(s_[:3]) < 20:
                    continue
                dt_ = t_ep - float(c_["epoch_mjd"])
                x = orbit.propagate(s_[:3], s_[3:], np.array([dt_, dt_ + 1e-3]))
                Ls.append(np.r_[x[0], (x[1] - x[0]) / 1e-3])
    except Exception as ex:
        log("linker seeds failed:", ex)
        Ls = []
    if Ls:
        states = np.vstack([np.array(Ls), states])
        seed_sc = np.r_[1e6 - np.arange(len(Ls), dtype=float), seed_sc]
        log(f"{len(Ls)} linker seeds added (refined first)")
    skind = np.zeros(0, int)
    # gen 31 (v2): per-night co-add seeds, refined right after the linker seeds.  TNO_NIGHTSEED=0 disables.
    if os.environ.get("TNO_NIGHTSEED", "1") != "0":
        try:
            tns0 = time.time()
            NSN = int(os.environ.get("TNO_NS_N", "0") or 0) or (8000 if cuda else 1500)
            ns_dl = min(budget * float(os.environ.get("TNO_NS_DL", "0.80")), (time.time() - T0) + float(os.environ.get("TNO_NS_DLF", "0.15")) * budget)
            if ns_pre is not None:
                nrows, nfr = ns_pre
            else:
                nrows, nfr = night_seeds(torch, F, dev, maps, cov, wf, t, night, obs, TF0, x0, y0, PIX, W, H,
                                         ns_dl, NSN, log)
            off_ = len(frames)
            frames.extend(nfr)
            NSs, nsc = [], []
            for r in nrows[:4 * NSN]:
                s_, ep_ = to_state((r[0], r[1], r[2], r[3], r[4], r[5], r[6] + off_))
                x = orbit.propagate(s_[:3], s_[3:], np.array([t_ep - ep_, t_ep - ep_ + 1e-3]))
                st_ = np.r_[x[0], (x[1] - x[0]) / 1e-3]
                if np.all(np.isfinite(st_)):
                    NSs.append(st_); nsc.append(r[0])
            if NSs:
                NSs = np.array(NSs)
                # dedup: a night seed is dropped if on >= 2 nights its predicted position is within ~4" of an
                # accepted night seed or a linker seed (cells of 4", 3x3 neighbourhood marked on accept)
                fr_ = np.array([np.flatnonzero(night == u)[0] for u in nights_u])
                occ = [set() for _ in fr_]
                def cells_(S_):
                    uu = orbit.predict_dirs_b(S_, t_ep, t[fr_], obs[fr_])
                    qq = uu @ TF0.T
                    return (np.floor(qq[..., 0] / qq[..., 2] / (4 * PIX)).astype(np.int64),
                            np.floor(qq[..., 1] / qq[..., 2] / (4 * PIX)).astype(np.int64))
                def mark_(kx_, ky_):
                    for k_ in range(len(fr_)):
                        for ddx in (-1, 0, 1):
                            for ddy in (-1, 0, 1):
                                occ[k_].add((kx_[k_] + ddx, ky_[k_] + ddy))
                if len(Ls):
                    lx_, ly_ = cells_(np.array(Ls))
                    for j in range(len(Ls)):
                        mark_(lx_[j], ly_[j])
                kx, ky = cells_(NSs)
                kp = []
                for j in range(len(NSs)):
                    hit_ = sum(1 for k_ in range(len(fr_)) if (kx[j, k_], ky[j, k_]) in occ[k_])
                    if hit_ >= 2:
                        continue
                    kp.append(j); mark_(kx[j], ky[j])
                    if len(kp) >= NSN:
                        break
                NSs = NSs[kp]
                nL = len(Ls)
                states = np.vstack([states[:nL], NSs, states[nL:]])
                # interleaved 1:1 with the window seeds (TNO_NS_ILV=0: all night seeds first) so window seeds still get
                # refined and serve as the null population for the p calibration (night seeds are mostly real bodies)
                if os.environ.get("TNO_NS_ILV", "1") != "0":
                    nsk = -np.arange(len(NSs), dtype=float) - 0.5
                else:
                    nsk = 5e5 - np.arange(len(NSs), dtype=float)
                seed_sc = np.r_[seed_sc[:nL], nsk, seed_sc[nL:]]
                skind = np.r_[np.zeros(nL, int), np.ones(len(NSs), int), np.full(len(states) - nL - len(NSs), 2)]
                log(f"night seeds: {len(NSs)} seeds added after dedup (from {len(nrows)} rows), top score "
                    f"{nsc[0]:.2f}, in {time.time() - tns0:.1f}s")
            else:
                log(f"night seeds: none ({time.time() - tns0:.1f}s)")
        except Exception as ex:
            import traceback
            traceback.print_exc()
            log("night seeds failed:", repr(ex)[:300])
    if len(skind) != len(states):
        skind = np.r_[np.zeros(len(Ls), int), np.full(len(states) - len(Ls), 2)]
    try:
        if len(sdeep) and sdeep.any() and len(states) >= len(sdeep):
            skind[len(states) - len(sdeep):][sdeep] = 3      # window seeds are the last len(best_all) states
            log(f"deep-window seeds (kind 3, outside the p null): {int(sdeep.sum())} of {len(sdeep)}")
    except Exception as ex:
        log("deep seed kinds failed:", ex)
    mflat = maps.reshape(N, -1)
    covf = cov.reshape(N, -1)
    nights_idx = [np.flatnonzero(night == u) for u in nights_u]

    def score_states(S):
        """S (K,6) at t_ep -> per-exposure values (K,N) and coverage."""
        u = orbit.predict_dirs_b(S, t_ep, t, obs)            # (K,N,3)
        q = u @ TF0.T
        ix = np.rint((q[..., 0] / q[..., 2] - x0) / PIX).astype(np.int64)
        iy = np.rint((q[..., 1] / q[..., 2] - y0) / PIX).astype(np.int64)
        okk = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        lin = np.where(okk, iy * W + ix, 0) + np.arange(N)[None] * (H * W)
        lt_ = torch.from_numpy(lin.ravel()).to(dev)
        v = maps.reshape(-1)[lt_].float().reshape(lin.shape).cpu().numpy()
        c = cov.reshape(-1)[lt_].cpu().numpy().reshape(lin.shape) & okk
        v = np.where(c, v, 0)
        return v, c

    wf2 = wf ** 2

    def total(v, c):
        if not DW:
            nc = np.maximum(c.sum(1), 1)
            return v.sum(1) / np.sqrt(nc)
        # weighted stack over covered frames (v is 0 where not covered)
        return (v * wf[None]).sum(1) / np.sqrt(np.maximum((c * wf2[None]).sum(1), 1e-3))

    def to_cands(S, sc, nn_, kd=None):
        out = []
        ref_ = sc
        try:
            # gen 31 (v2): null population = refined window seeds (night/linker seeds are mostly real bodies)
            if kd is not None and int((kd == 2).sum()) >= 30:
                ref_ = sc[kd == 2]
            elif kd is not None and int((kd >= 2).sum()) >= 30:
                ref_ = sc[kd >= 2]
        except Exception:
            ref_ = sc
        med = np.median(ref_); mad = 1.4826 * np.median(np.abs(ref_ - med)) + 1e-3
        z_ = (sc - med) / mad
        for j in range(len(S)):
            if nn_[j] < 3 or not np.all(np.isfinite(S[j])):
                continue
            r = np.linalg.norm(S[j, :3]); v2 = S[j, 3:] @ S[j, 3:]
            if r < 25 or 0.5 * v2 - GM / r >= 0:
                continue
            p = 1 / (1 + math.exp(min(60.0, max(-60.0, -(z_[j] - 4.5) / 0.8)))) * (1.0 if nn_[j] >= 4 else 0.8)
            out.append(dict(epoch_mjd=float(t_ep), state=[float(x) for x in S[j]], p=float(min(max(p, 1e-4), 0.999))))
        return out

    # gen 32: TNO_GATE2 (default 1.5; 0 = gen031 gate) -> for tracks whose weighted all-exposure stack reaches GATE2_T, a night also counts
    # as "seen" when its own night stack (weighted, S/N units) reaches GATE2, not only when one exposure reaches 2.5.
    # Weight-4 bodies (all-exposure S/N >= 5, but no single exposure at 2.5 on 3 nights) are otherwise always cut.
    GATE2 = float(os.environ.get("TNO_GATE2", "1.5") or 0)
    GATE2_T = float(os.environ.get("TNO_GATE2_T", "5.5"))

    # gen 62: TNO_GATE3 (0 = gen061 gate) -> a track whose all-exposure stack reaches GATE2_T, that is strong (night stack or one
    # exposure >= GATE3; per-exposure maps are clamped at 5) on >= 2 nights and has map coverage on >= 3 nights, is credited with
    # 3 nights. Targets weight-4 bodies "bright on one or two nights and too weak to see on the others", always cut before.
    GATE3 = float(os.environ.get("TNO_GATE3", "4.5") or 0)

    def nights_good(v, c):
        g = np.zeros(len(v), int)
        big = total(v, c) >= GATE2_T if (GATE2 > 0 or GATE3 > 0) else None
        nstrong = np.zeros(len(v), int); ncov = np.zeros(len(v), int)
        for idx in nights_idx:
            vmax_ = v[:, idx].max(1)
            ok_ = vmax_ >= 2.5
            wn = wf[idx]
            ns_ = (v[:, idx] * wn[None]).sum(1) / np.sqrt(np.maximum((c[:, idx] * (wn ** 2)[None]).sum(1), 1e-3))
            cv_ = c[:, idx].any(1)
            if GATE2 > 0:
                ok_ = ok_ | (big & (ns_ >= GATE2) & cv_)
            g += ok_.astype(int)
            if GATE3 > 0:
                nstrong += (((ns_ >= GATE3) | (vmax_ >= GATE3)) & cv_).astype(int)
                ncov += cv_.astype(int)
        if GATE3 > 0:
            g = np.where(big & (nstrong >= 2) & (ncov >= 3), np.maximum(g, 3), g)
        return g

    if len(states) == 0:
        return
    # dedup seeds by predicted positions on window exposures
    v0, c0 = score_states(states)
    write_out(a.out, to_cands(states, total(v0, c0), nights_good(v0, c0)))
    log(f"wrote {len(states)} seed states (pre-refinement)")

    rng = np.random.default_rng(1)
    K = 384 if cuda else 192
    refined, rsc, rng_, rkd = [], [], [], []
    order_ = np.argsort(-seed_sc)
    # gen 34: cross-validated rerank (see NOTES.md) needs time after refinement
    CV = os.environ.get("TNO_CV", "1") != "0"
    ref_deadline = budget * (float(os.environ.get("TNO_CV_REF", "0.92")) if CV else 0.95)

    def cem(s, colmask=None, K_=K, iters=6):
        r = np.linalg.norm(s[:3])
        TL = tframe(s[:3])
        # sigmas: position 1.5" (at distance r), tangential velocity, radial distance
        sig = np.array([1.5 * ARCR * r, 1.5 * ARCR * r, 0.6 * dv * SP * r, 0.6 * dv * SP * r, 0.03 * r])
        mu = np.zeros(5)
        bestS, bestv = s, -1e9
        for itn in range(iters):
            Z = mu[None] + rng.standard_normal((K_, 5)) * sig[None]
            Z[0] = mu
            S = np.repeat(s[None], K_, 0)
            S[:, :3] += Z[:, 0:1] * TL[0] + Z[:, 1:2] * TL[1]
            # distance change: scale position along line of sight from the reference observer
            los = S[:, :3] - oref; losn = los / np.linalg.norm(los, axis=1)[:, None]
            S[:, :3] += Z[:, 4:5] * losn
            S[:, 3:] += Z[:, 2:3] * TL[0] + Z[:, 3:4] * TL[1]
            v, c = score_states(S)
            if colmask is not None:
                v = v * colmask[None]; c = c & colmask[None]
            tot = total(v, c)
            el = np.argsort(-tot)[:max(8, K_ // 16)]
            if tot[el[0]] > bestv:
                bestv, bestS = tot[el[0]], S[el[0]].copy()
            mu = Z[el].mean(0); sig = np.maximum(Z[el].std(0), sig * 0.3)
        return bestS

    # gen 63: astrometric polish.  CEM maximises the summed S/N on 1" clamped maps with a 1.5" position sigma, so on long
    # arcs a found body's state can sit several arcsec off on one night (injection: an S/N 12 body at rank 23 was 5.5" off
    # on its first night, 0.4-1.1" on the other two -> not a hit).  For the top candidates: centroid the S/N peak within
    # POL_R" of the prediction on every covered frame, least-squares orbit fit to those positions with outlier clipping,
    # keep the new state only if it is bound, fits to < 1", and its stack S/N does not drop.
    POLISH = int(float(os.environ.get("TNO_POLISH", "1") or 0))
    POL_R = int(os.environ.get("TNO_POL_R", "10"))
    POL_N = int(os.environ.get("TNO_POL_N", "200" if cuda else "120"))
    POL_MIN = float(os.environ.get("TNO_POL_MIN", "2.5"))
    POL_DGRID = int(os.environ.get("TNO_POL_DGRID", "1"))
    POL_DLIST = [26.0, 28, 31, 34, 37, 40, 43, 46, 50, 55, 62, 70, 80]

    def polish(cands_):
        if not cands_:
            return cands_
        tp0 = time.time()
        dl_ = budget * 0.985
        order_p = sorted(range(len(cands_)), key=lambda i: -cands_[i]["p"])[:POL_N]
        S0 = np.array([cands_[i]["state"] for i in order_p], float)
        ep_ = float(cands_[order_p[0]]["epoch_mjd"])
        u = orbit.predict_dirs_b(S0, ep_, t, obs)
        q = u @ TF0.T
        fx = (q[..., 0] / q[..., 2] - x0) / PIX
        fy = (q[..., 1] / q[..., 2] - y0) / PIX
        R_ = POL_R
        ix = np.rint(fx).astype(np.int64); iy = np.rint(fy).astype(np.int64)
        inb = (ix >= R_ + 1) & (ix < W - R_ - 1) & (iy >= R_ + 1) & (iy < H - R_ - 1)
        ixc = np.where(inb, ix, R_ + 1); iyc = np.where(inb, iy, R_ + 1)
        d_ = np.arange(-R_, R_ + 1)
        K_, N_ = ix.shape
        lin = (np.arange(N_)[None, :, None, None] * (H * W) + (iyc[..., None, None] + d_[None, None, :, None]) * W
               + (ixc[..., None, None] + d_[None, None, None, :]))
        lt_ = torch.from_numpy(lin.reshape(-1)).to(dev)
        P = maps.reshape(-1)[lt_].float().reshape(K_, N_, 2 * R_ + 1, 2 * R_ + 1).cpu().numpy()
        Cv = cov.reshape(-1)[lt_].reshape(K_, N_, 2 * R_ + 1, 2 * R_ + 1).cpu().numpy()
        P = np.where(Cv, P, 0.0)
        P[~inb] = 0.0
        # only peaks inside the circle of radius R_
        rr2 = d_[:, None] ** 2 + d_[None, :] ** 2
        P[..., rr2 > R_ * R_] = 0.0
        nimp = nfix = 0
        out_ = list(cands_)

        def _bad(x_):
            r_ = float(np.linalg.norm(x_[:3]))
            return r_ < 25 or 0.5 * float(x_[3:] @ x_[3:]) - GM / r_ >= 0

        def dgrid_fit(s0, ep, tt0, oo0, uu0, nn0):
            n0 = int(np.argmin(np.abs(tt0 - ep)))
            o = oo0[n0]
            u0 = np.asarray(s0[:3], float) - o
            u0 /= np.linalg.norm(u0)
            vt = np.asarray(s0[3:], float)
            best = None
            for dd in POL_DLIST:
                if time.time() - T0 > dl_:
                    break
                ou = float(o @ u0)
                rho = -ou + math.sqrt(max(ou * ou - float(o @ o) + dd * dd, 0.0))
                p0 = o + rho * u0
                ph = p0 / np.linalg.norm(p0)
                vv = vt - (vt @ ph) * ph
                if np.linalg.norm(vv) <= 0:
                    continue
                st0 = np.r_[p0, vv / np.linalg.norm(vv) * math.sqrt(GM / dd)]
                tt, oo, uu, nn_a = tt0, oo0, uu0, nn0
                st = st0
                ok = True
                for _ in range(len(tt0)):
                    try:
                        st, res = orbit.fit_orbit(st, ep, tt, oo, uu, iters=15, vr_sig=0.0003, d_fix=dd, d_sig=0.02 * dd)
                    except Exception:
                        ok = False
                        break
                    if not np.all(np.isfinite(st)):
                        ok = False
                        break
                    if res.max() <= 1.0:
                        break
                    w = int(np.argmax(res))
                    tt, oo, uu, nn_a = (np.delete(x_, w, 0) for x_ in (tt, oo, uu, nn_a))
                    if len(np.unique(nn_a)) < 3 or len(tt) < 4:
                        ok = False
                        break
                if not ok or res.max() > 1.0 or _bad(st):
                    continue
                key = (len(tt), -float(res.max()))
                if best is None or key > best[0]:
                    best = (key, st)
            return None if best is None else best[1]
        DBG_ = os.environ.get("TNO_POL_DEBUG")
        dbg_ = []
        for k, ci in enumerate(order_p):
            if time.time() - T0 > dl_:
                break
            pts_t, pts_o, pts_u, pts_n = [], [], [], []
            for n in range(N_):
                if not inb[k, n]:
                    continue
                m = P[k, n]
                j = int(np.argmax(m)); py, px = divmod(j, 2 * R_ + 1)
                if m[py, px] < POL_MIN or py in (0, 2 * R_) or px in (0, 2 * R_):
                    continue
                if m[py, px] >= 4.99:
                    # clamped plateau (maps clip at 5): argmax is its first (top-left) pixel, so take the plateau's
                    # centre (pixels >= 4.99 within 3 px of the argmax) instead of a biased 3x3 centroid
                    y0_, y1_ = max(py - 3, 0), min(py + 4, 2 * R_ + 1)
                    x0_, x1_ = max(px - 3, 0), min(px + 4, 2 * R_ + 1)
                    yy, xx = np.nonzero(m[y0_:y1_, x0_:x1_] >= 4.99)
                    cy, cx = float(yy.mean() + y0_), float(xx.mean() + x0_)
                else:
                    w3 = np.clip(m[py - 1:py + 2, px - 1:px + 2], 0, None)
                    if w3.sum() <= 0:
                        continue
                    cy = py + float((w3.sum(1) * np.array([-1, 0, 1])).sum() / w3.sum())
                    cx = px + float((w3.sum(0) * np.array([-1, 0, 1])).sum() / w3.sum())
                X = x0 + (ixc[k, n] + cx - R_) * PIX
                Y = y0 + (iyc[k, n] + cy - R_) * PIX
                v_ = TF0.T @ np.array([X, Y, 1.0])
                pts_t.append(t[n]); pts_o.append(obs[n]); pts_u.append(v_ / np.linalg.norm(v_)); pts_n.append(night[n])
            if DBG_:
                dbg_.append(dict(state=[float(x) for x in S0[k]], epoch_mjd=ep_, npts=len(pts_t), nights=len(np.unique(pts_n)), why="pts"))
            if len(np.unique(pts_n)) < 3 or len(pts_t) < 4:
                continue
            tt = np.array(pts_t); oo = np.array(pts_o); uu = np.array(pts_u); nn_a = np.array(pts_n)
            st = S0[k].copy()
            okp = True
            for _ in range(len(tt)):
                try:
                    st, res = orbit.fit_orbit(st, ep_, tt, oo, uu)
                except Exception:
                    okp = False
                    break
                if not np.all(np.isfinite(st)):
                    okp = False
                    break
                if res.max() <= 1.0:
                    break
                w = int(np.argmax(res))
                tt, oo, uu, nn_a = (np.delete(x_, w, 0) for x_ in (tt, oo, uu, nn_a))
                if len(np.unique(nn_a)) < 3 or len(tt) < 4:
                    okp = False
                    break
            if DBG_:
                dbg_[-1].update(why="fit", npts2=len(tt), res=float(res.max()) if okp else -1.0)
            if DBG_:
                dbg_[-1].update(okp=bool(okp))
            if okp and res.max() <= 1.0 and _bad(st):
                # short arcs leave distance / radial velocity loose: retry from the CEM state with a tight vr prior
                st_ok = None
                for vs_ in (0.0003, 0.0001):
                    try:
                        s_, r_ = orbit.fit_orbit(S0[k], ep_, tt, oo, uu, iters=12, vr_sig=vs_)
                    except Exception:
                        continue
                    if np.all(np.isfinite(s_)) and not _bad(s_) and r_.max() <= 1.0:
                        st_ok = s_
                        break
                if DBG_:
                    dbg_[-1].update(why="unbound", fixed=st_ok is not None)
                st = st_ok
                if st is None:
                    okp = False
            if (not okp or res.max() > 1.0) and POL_DGRID and time.time() - T0 < dl_:
                # gen064: distance-grid fit -- pin |r| to each grid distance (4 free parameters + vr prior), clip
                # outliers from the FULL point set, keep the bound fit with the most points / smallest residual
                st = dgrid_fit(S0[k], ep_, np.array(pts_t), np.array(pts_o), np.array(pts_u), np.array(pts_n))
                if DBG_:
                    dbg_[-1].update(dgrid=st is not None)
                if st is None:
                    continue
            elif not okp or res.max() > 1.0:
                continue
            v, c = score_states(np.array([S0[k], st]))
            tot2 = total(v, c); ng2 = nights_good(v, c)
            if DBG_:
                dbg_[-1].update(why="score", tot=[float(x) for x in tot2], ng=[int(x) for x in ng2], new=[float(x) for x in st])
            if tot2[1] < 0.97 * tot2[0] or ng2[1] < 3:
                continue
            nimp += 1
            dmax = float(np.max(np.degrees(np.arccos(np.clip(np.sum(
                orbit.predict_dirs(st, ep_, t, obs) * u[k], 1), -1, 1))) * 3600))
            if dmax > 2.0:
                nfix += 1
            out_[ci] = dict(cands_[ci], state=[float(x) for x in st])
        if DBG_:
            json.dump(dbg_, open(DBG_, "w"))
        log(f"polish: {nimp} of {len(order_p)} states replaced ({nfix} moved > 2\" somewhere) in {time.time() - tp0:.1f}s")
        return out_

    for jj, j in enumerate(order_):
        if time.time() - T0 > ref_deadline:
            break
        bestS = cem(states[j].copy())
        v, c = score_states(bestS[None])
        refined.append(bestS); rsc.append(total(v, c)[0]); rng_.append(nights_good(v, c)[0]); rkd.append(skind[j])
        if (jj % 50) == 49:
            R = np.array(refined)
            write_out(a.out, dedup(to_cands(R, np.array(rsc), np.array(rng_), np.array(rkd)), score_states, t_ep))
    if refined:
        R = np.array(refined)
        cands = to_cands(R, np.array(rsc), np.array(rng_), np.array(rkd))
        cands = dedup(cands, score_states, t_ep)
        write_out(a.out, cands)
        if CV:
            try:
                cands = cv_rerank(cands, R, np.array(rkd), np.array(rsc), cem, score_states, total, t, night, nights_u, budget, T0, log)
                write_out(a.out, cands)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                log("cv rerank failed:", repr(ex)[:300])
        if POLISH:
            try:
                cands = polish(cands)
                write_out(a.out, cands)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                log("polish failed:", repr(ex)[:300])
        try:
            rk_ = np.array(rkd)
            log(f"refined by kind (linker/night/window/deep): {[int((rk_ == k_).sum()) for k_ in range(4)]}; "
                f"window-seed median {np.median(np.array(rsc)[rk_ == 2]) if (rk_ == 2).any() else 0:.2f}")
        except Exception:
            pass
        log(f"seed-score median {np.median(rsc):.2f}, max {np.max(rsc):.2f}; refined {len(refined)}; output {len(cands)}; top p {[round(c['p'], 3) for c in sorted(cands, key=lambda c: -c['p'])[:8]]}")


def _ns_gpu_pairs(torch, dev, Q, pairs, tN, tref, vcx, vcy, vcs, VRAD, vmax, TOL, TD, d, oN, TF0, x0, y0, PIX, W, H,
                  NMs, gsel, GOOD=2.5, MAXE=6_000_000):
    """gen 51: GPU version of the night-seed pair search + forced night read-out for one distance.
    Pairs (a, b) are found by sorting night b's d-frame peaks by x and expanding each night-a peak's x window
    (searchsorted + repeat_interleave), then exact radius and |v| <= vmax tests; every surviving pair is
    projected onto every night and read from that night's (max-pooled) co-add; only candidates with >= 3 nights
    at >= GOOD are returned (as numpy: C (K, 4) = (x, y, vx, vy) at tref, vals (K, nN), covc (K, nN))."""
    f32 = torch.float32
    nN = len(Q)
    Qt = [torch.as_tensor(np.asarray(q, np.float64), dtype=f32, device=dev) for q in Q]
    TDt = torch.as_tensor(TD, dtype=f32, device=dev); TFt = torch.as_tensor(TF0, dtype=f32, device=dev)
    oNt = torch.as_tensor(np.asarray(oN), dtype=f32, device=dev)
    vct = torch.tensor([vcx, vcy], dtype=f32, device=dev)
    outC, outV, outK = [], [], []
    for a, b in pairs:
        Qa, Qb = Qt[a], Qt[b]
        if len(Qa) == 0 or len(Qb) == 0:
            continue
        dt = float(tN[b] - tN[a])
        R = dt * (VRAD * vcs) + TOL
        ctr = Qa + dt * vct[None]
        xs, ordb = torch.sort(Qb[:, 0].contiguous())
        Qbs = Qb[ordb]
        lo = torch.searchsorted(xs, (ctr[:, 0] - R).contiguous())
        hi = torch.searchsorted(xs, (ctr[:, 0] + R).contiguous())
        cnt = (hi - lo)
        # process night-a peaks in chunks so the expanded pair list stays below MAXE
        cs = torch.cumsum(cnt, 0)
        tot = int(cs[-1].item()) if len(cs) else 0
        if tot == 0:
            continue
        nch = max(1, int(math.ceil(tot / MAXE)))
        bounds = torch.searchsorted(cs, torch.linspace(0, tot, nch + 1, device=dev)[1:-1].to(cs.dtype).contiguous())
        edges = [0] + [int(x) + 1 for x in bounds.tolist()] + [len(Qa)]
        for e0, e1 in zip(edges[:-1], edges[1:]):
            if e1 <= e0:
                continue
            c_ = cnt[e0:e1]
            n_ = int(c_.sum().item())
            if n_ == 0:
                continue
            ia = torch.repeat_interleave(torch.arange(e0, e1, device=dev), c_)
            st = torch.cumsum(c_, 0) - c_
            ib = lo[e0:e1].repeat_interleave(c_) + (torch.arange(n_, device=dev) - st.repeat_interleave(c_))
            dlt = Qbs[ib] - ctr[ia]
            ok = (dlt * dlt).sum(1) <= R * R
            ia, ib = ia[ok], ib[ok]
            if ia.numel() == 0:
                continue
            v = (Qbs[ib] - Qa[ia]) / dt
            okv = torch.sqrt((v * v).sum(1)) <= vmax + TOL / dt
            ia, v = ia[okv], v[okv]
            if ia.numel() == 0:
                continue
            p0 = Qa[ia] + v * (tref - tN[a])
            K = p0.shape[0]
            vals = torch.zeros((K, nN), dtype=f32, device=dev)
            covc = torch.zeros((K, nN), dtype=torch.bool, device=dev)
            for k in range(nN):
                P = p0 + v * float(tN[k] - tref)
                wv = TDt[2][None] + P[:, 0:1] * TDt[0][None] + P[:, 1:2] * TDt[1][None]
                X = d * wv / torch.linalg.norm(wv, dim=1, keepdim=True) - oNt[k][None]
                q = X @ TFt.T
                ix = torch.round((q[:, 0] / q[:, 2] - x0) / PIX).long()
                iy = torch.round((q[:, 1] / q[:, 2] - y0) / PIX).long()
                okp = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
                lin = torch.where(okp, iy * W + ix, torch.zeros_like(ix))
                vv = NMs[gsel[k]][lin].float()
                vals[:, k] = torch.where(okp, vv, torch.zeros_like(vv)); covc[:, k] = okp & (vv != 0)
            ng = ((vals >= GOOD) & covc).sum(1)
            g = ng >= 3
            if bool(g.any()):
                outC.append(torch.cat([p0[g], v[g]], 1)); outV.append(vals[g]); outK.append(covc[g])
    if not outC:
        return None, None, None
    C = torch.cat(outC).cpu().double().numpy()
    return C, torch.cat(outV).cpu().numpy(), torch.cat(outK).cpu().numpy()


def night_seeds(torch, F, dev, maps, cov, wf, t, night, obs, TF0, x0, y0, PIX, W, H, deadline, NS, log_):
    """gen 31 (v2): per-night co-add seeds.  Each night's frames are co-added (weighted, S/N units), local maxima
    with night z >= NZ are linked across nights: for each distance d (heliocentric d-frame, motion linear), a pair
    of peaks on two nights gives (vx, vy); keep plausible rates, predict the other nights and read the (3x3
    max-pooled) night co-add there; require >= 3 nights with value >= 2.5.  Returns (rows, frames) where rows are
    (score, d, vx, vy, px, py, local frame index) like the window search's and frames are (TD, tref, oref)."""
    from scipy.spatial import cKDTree
    NZ = float(os.environ.get("TNO_NS_Z", "3.7"))
    # gen 51: pair search + forced read-out on the GPU (opt-in TNO_NS_GPU=1; default = gen050 cKDTree/numpy path), so a large
    # night-peak cap fits (gen050: B bodies with peaks on >= 3 nights 4/15 at cap 1500, 12/15 at 12000).
    GPUP = os.environ.get("TNO_NS_GPU", "0") == "1"
    CAP = int(os.environ.get("TNO_NS_CAP", "0") or 0) or ((12000 if GPUP else 4000) if str(dev).startswith("cuda") else 1500)
    TOL = float(os.environ.get("TNO_NS_TOL", "2.5")) * ARCR     # pairing tolerance (rad, 1"/px grid)
    GOOD = float(os.environ.get("TNO_NS_GOOD", "2.5"))
    TRIP = int(os.environ.get("TNO_NS_TRIP", "0") or 0)          # gen 52: nights with a matching peak required
    TRT = float(os.environ.get("TNO_NS_TRT", "1.0"))            # peak match radius in units of TOL
    SCMIN = float(os.environ.get("TNO_NS_SCMIN", "0") or 0)      # min forced night-sum score
    VRAD = 0.75
    nu = np.unique(night)
    if len(nu) < 3:
        return [], []
    TILE = os.environ.get("TNO_NS_TILE", "0") != "0"
    NT_ = int(os.environ.get("TNO_NS_NTILE", "8"))            # tiles per axis (NT_ x NT_)
    TS = max(32, int(math.ceil(max(H, W) / NT_)))
    ntx = int(math.ceil(W / TS)); nty = int(math.ceil(H / TS))
    QT = max(1, int(float(os.environ.get("TNO_NS_TFRAC", "0.75")) * CAP / (ntx * nty)))
    # gen 31: motion-compensated night co-adds.  Within a night a TNO moves ~150"/hr / d (Earth reflex), so a
    # plain co-add of a night spanning hours smears it over many 1" pixels.  Each night is co-added for G_k
    # inverse-distance groups (G_k grows with the night's time span), shifting every frame by the predicted
    # displacement of a circular-orbit body at that group's distance; the distance loop below uses, per night,
    # the group nearest to its d.  TNO_NS_MC=0 restores the plain co-add (one group).
    MC = os.environ.get("TNO_NS_MC", "1") != "0"
    MSTEP = float(os.environ.get("TNO_NS_MSTEP", "1.2"))        # max residual smear between groups (arcsec)
    GMX = int(os.environ.get("TNO_NS_GMAX", "0") or 0) or (16 if str(dev).startswith("cuda") else 6)
    MEMB = float(os.environ.get("TNO_NS_MEMGB", "0") or 0) or (8.0 if str(dev).startswith("cuda") else 2.5)
    il_lo = 1 / float(os.environ.get("TNO_NS_DMAX", "75")); il_hi = 1 / float(os.environ.get("TNO_NS_DMIN", "28"))
    z0_ = TF0[2]
    pole_ = np.array([0, -math.sin(EPS), math.cos(EPS)])
    def shift_px(d, tt, oo, tr, orf):
        """pixel displacement (dx, dy) at times tt from observers oo, relative to (tr, orf), of a circular-orbit
        body at heliocentric distance d toward the field centre"""
        b_ = orf @ z0_; rho = -b_ + math.sqrt(max(b_ * b_ - orf @ orf + d * d, 1e-9))
        X0 = orf + rho * z0_
        vc = np.cross(pole_, X0 / d); vc = vc / np.linalg.norm(vc) * math.sqrt(GM / d)
        def pix(X):
            q = X @ TF0.T
            return q[..., 0] / q[..., 2] / PIX, q[..., 1] / q[..., 2] / PIX
        x_r, y_r = pix(X0 - orf)
        X = X0[None] + np.outer(tt - tr, vc) - oo
        xx_, yy_ = pix(X)
        return xx_ - x_r, yy_ - y_r
    def shifted(m, dx, dy):
        """m (H, W) shifted so that out[y, x] = m[y + dy, x + dx] (zero outside)"""
        if dx == 0 and dy == 0:
            return m
        o = torch.zeros_like(m)
        ys0, ys1 = max(0, -dy), min(H, H - dy); xs0, xs1 = max(0, -dx), min(W, W - dx)
        if ys1 > ys0 and xs1 > xs0:
            o[ys0:ys1, xs0:xs1] = m[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        return o
    nidx = [np.flatnonzero(night == u) for u in nu]
    spans = np.array([float(t[i].max() - t[i].min()) * 24.0 for i in nidx])
    # groups per night: span (hr) * 150"/hr * delta(1/d) / MSTEP, capped, and capped by total map memory
    Gk = np.ones(len(nu), int)
    if MC:
        Gk = np.clip(np.ceil(spans * 150.0 * (il_hi - il_lo) / MSTEP).astype(int), 1, GMX)
        mem_ = lambda G: float(G.sum()) * H * W * 2 / 1e9
        while mem_(Gk) > MEMB and Gk.max() > 1:
            Gk = np.maximum(1, Gk - (Gk == Gk.max()).astype(int))
    tN, oN, PK, ZK, GC, GI = [], [], [], [], [], []
    NMs = torch.empty((int(Gk.sum()), H * W), dtype=torch.float16, device=dev); nm_i = 0
    for k_, u in enumerate(nu):
        idx = nidx[k_]
        tN.append(float(t[idx].mean())); oN.append(obs[idx].mean(0))
        G = int(Gk[k_])
        cen = [1 / (il_lo + (g + 0.5) * (il_hi - il_lo) / G) for g in range(G)]
        GC.append(np.array([1 / c for c in cen])); GI.append([])
        for g in range(G):
            if G > 1 or (MC and spans[k_] > 0):
                sx, sy = shift_px(cen[g], t[idx], obs[idx], tN[-1], oN[-1])
            else:
                sx = sy = np.zeros(len(idx))
            acc = torch.zeros((H, W), dtype=torch.float32, device=dev)
            cw = torch.zeros((H, W), dtype=torch.float32, device=dev)
            for jn, n in enumerate(idx):
                dx_, dy_ = int(round(float(sx[jn]))), int(round(float(sy[jn])))
                acc += shifted(maps[n].float(), dx_, dy_) * float(wf[n])
                cw += shifted(cov[n].float(), dx_, dy_) * float(wf[n] ** 2)
            z = torch.where(cw > 0, acc / torch.sqrt(torch.clamp(cw, min=1e-3)), torch.zeros_like(acc))
            del acc, cw
            mp = F.max_pool2d(z[None, None], 5, 1, 2)[0, 0]
            pk = (z == mp) & (z >= NZ)
            yy, xx = torch.nonzero(pk, as_tuple=True)
            sc = z[yy, xx]
            if sc.numel() > CAP:
                if TILE:
                    # gen037 tile: spatially tiled cap -- top CAP/ntiles peaks per tile, rest filled globally
                    yn = yy.cpu().numpy(); xn = xx.cpu().numpy(); sn = sc.float().cpu().numpy()
                    tid = (yn // TS) * ntx + (xn // TS)
                    o_ = np.lexsort((-sn, tid))
                    ts_ = tid[o_]
                    st_ = np.r_[0, np.flatnonzero(np.diff(ts_)) + 1]
                    rk_ = np.arange(len(ts_)) - np.repeat(st_, np.diff(np.r_[st_, len(ts_)]))
                    keep = np.zeros(len(sn), bool); keep[o_[rk_ < QT]] = True
                    rest = np.flatnonzero(~keep)
                    nf = CAP - int(keep.sum())
                    if nf > 0 and len(rest):
                        keep[rest[np.argsort(-sn[rest])[:nf]]] = True
                    top = torch.from_numpy(np.flatnonzero(keep)).to(yy.device)
                else:
                    top = torch.topk(sc, CAP).indices
                yy, xx, sc = yy[top], xx[top], sc[top]
            GI[-1].append(nm_i)
            NMs[nm_i] = F.max_pool2d(z[None, None], 3, 1, 1)[0, 0].reshape(-1).half(); nm_i += 1
            del z, mp, pk
            PK.append(np.stack([xx.cpu().numpy(), yy.cpu().numpy()], 1).astype(np.float64))
            ZK.append(sc.float().cpu().numpy())
            if time.time() - T0 > deadline:
                log_("night seeds: deadline during co-adds")
                return [], []
    tN = np.array(tN); oN = np.array(oN); nN = len(nu)
    log_(f"night seeds: groups per night {Gk.tolist()} (spans hr {np.round(spans, 2).tolist()}); "
         f"peaks per map {[len(p) for p in PK]}")
    # sky unit vectors of night peaks
    def sky_u(P):
        xi = x0 + P[:, 0] * PIX; eta = y0 + P[:, 1] * PIX
        u = TF0[2][None] + xi[:, None] * TF0[0][None] + eta[:, None] * TF0[1][None]
        return u / np.linalg.norm(u, axis=1)[:, None]
    UA = [sky_u(P) for P in PK]
    if os.environ.get("TNO_NS_DUMP"):
        try:
            nmap = np.concatenate([np.full(int(Gk[k]), k) for k in range(len(nu))])
            np.savez(os.environ["TNO_NS_DUMP"], u=np.concatenate(UA), z=np.concatenate(ZK),
                     night=np.concatenate([np.full(len(P), nmap[i]) for i, P in enumerate(PK)]),
                     tN=np.array(tN), TF0=TF0)
        except Exception as e_:
            log_(f"night seeds: dump failed {e_}")
    tref = 0.5 * (tN.min() + tN.max()); oref = oN.mean(0)
    # distance grid: step so the observer's non-linear motion shifts predictions by <= ~1.5"
    A_ = np.stack([np.ones_like(tN), tN - tref], 1)
    coef, *_ = np.linalg.lstsq(A_, oN, rcond=None)
    r_ = oN - A_ @ coef; r_ = r_ - np.outer(r_ @ TF0[2], TF0[2])
    rmax = max(np.sqrt((r_ ** 2).sum(1)).max(), 1e-6)
    step = float(os.environ.get("TNO_NS_DSTEP", "3.0")) * ARCR / rmax
    lo = 1 / float(os.environ.get("TNO_NS_DMAX", "75")); hi = 1 / float(os.environ.get("TNO_NS_DMIN", "28"))
    nst = max(1, int(math.ceil((hi - lo) / step)))
    dg = sorted([1 / (lo + (k + 0.5) * (hi - lo) / nst) for k in range(nst)], key=lambda d: abs(math.log(d / 42.0)))
    log_(f"night seeds: {nN} nights, {len(dg)} distances")
    z0 = TF0[2]
    pole = np.array([0, -math.sin(EPS), math.cos(EPS)])
    DTMAX = float(os.environ.get("TNO_NS_DTMAX", "25"))
    pairs = [(a, b) for a in range(nN) for b in range(a + 1, nN) if tN[b] - tN[a] <= DTMAX]
    if not pairs:
        pairs = [(a, a + 1) for a in range(nN - 1)]
    rows, frames = [], []
    TF0t = TF0
    nd = 0; ncand = 0
    for d in dg:
        if time.time() - T0 > deadline:
            log_("night seeds: deadline in distance loop")
            break
        b_ = oref @ z0; rho = -b_ + math.sqrt(b_ * b_ - oref @ oref + d * d)
        w0 = (oref + rho * z0) / d
        TD = tframe(w0)
        vmax = math.sqrt(2 * GM / d) / d
        vc = np.cross(pole, w0); vc = vc / np.linalg.norm(vc) * math.sqrt(GM / d)
        vcx, vcy = (TD[0] @ vc) / d, (TD[1] @ vc) / d
        vcs = math.hypot(vcx, vcy)
        # night peaks -> d-frame (per night, the co-add group nearest in 1/d)
        gsel = [GI[k][int(np.argmin(np.abs(GC[k] - 1 / d)))] for k in range(nN)]
        Q = []
        for k in range(nN):
            o = oN[k]; u = UA[gsel[k]]
            bb = u @ o; rr = -bb + np.sqrt(np.maximum(bb * bb - o @ o + d * d, 0))
            X = (o[None] + rr[:, None] * u) / d
            q = X @ TD.T
            Q.append(np.stack([q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]], 1))
        if GPUP:
            C, vals, covc = _ns_gpu_pairs(torch, dev, Q, pairs, tN, tref, vcx, vcy, vcs, VRAD, vmax, TOL, TD, d, oN,
                                          TF0t, x0, y0, PIX, W, H, NMs, gsel)
            if C is None:
                nd += 1; continue
        else:
          trees = [cKDTree(q) if len(q) else None for q in Q]
          cand = []
          for a, b in pairs:
            if trees[b] is None or len(Q[a]) == 0:
                continue
            dt = tN[b] - tN[a]
            ctr = Q[a] + dt * np.array([vcx, vcy])[None]
            lst = trees[b].query_ball_point(ctr, dt * (VRAD * vcs) + TOL)
            ia = np.repeat(np.arange(len(lst)), [len(l) for l in lst])
            if len(ia) == 0:
                continue
            ib = np.concatenate([np.asarray(l, dtype=np.int64) for l in lst])
            v = (Q[b][ib] - Q[a][ia]) / dt
            okv = (np.hypot(v[:, 0], v[:, 1]) <= vmax + TOL / dt)
            ia, ib, v = ia[okv], ib[okv], v[okv]
            p0 = Q[a][ia] + v * (tref - tN[a])
            cand.append(np.c_[p0, v])
          if not cand:
            nd += 1; continue
          C = np.concatenate(cand, 0)
          # predicted sky pixel on every night
          Kc = len(C)
          vals = np.zeros((Kc, nN), np.float32); covc = np.zeros((Kc, nN), bool)
          for k in range(nN):
            P = C[:, :2] + C[:, 2:] * (tN[k] - tref)
            wv = TD[2][None] + P[:, 0:1] * TD[0][None] + P[:, 1:2] * TD[1][None]
            X = d * wv / np.linalg.norm(wv, axis=1)[:, None] - oN[k][None]
            q = X @ TF0t.T
            ix = np.rint((q[:, 0] / q[:, 2] - x0) / PIX).astype(np.int64)
            iy = np.rint((q[:, 1] / q[:, 2] - y0) / PIX).astype(np.int64)
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            lin = torch.from_numpy(np.where(ok, iy * W + ix, 0)).to(dev)
            vv = NMs[gsel[k]][lin].float().cpu().numpy()
            vals[:, k] = np.where(ok, vv, 0); covc[:, k] = ok & (vv != 0)
        ng = ((vals >= GOOD) & covc).sum(1)
        ncov = np.maximum(covc.sum(1), 1)
        sc = np.clip(vals, -2, 6).sum(1) / np.sqrt(ncov)
        keep = np.flatnonzero(ng >= 3)
        # gen 52 (gen051 Next #1): strict gate -- a night *peak* (not only a map value) within TRT x TOL of the
        # predicted position on >= TRIP nights, and forced night-sum score >= SCMIN.  Lets a larger peak cap
        # be used without flooding refinement with noise pairs.  TRIP=0 (default) = gen051 behaviour.
        if TRIP > 0 and len(keep):
            if GPUP:
                trees = [cKDTree(q) if len(q) else None for q in Q]
            npk = np.zeros(len(keep), np.int64)
            Ck = C[keep]
            for k in range(nN):
                if trees[k] is None:
                    continue
                P = Ck[:, :2] + Ck[:, 2:] * (tN[k] - tref)
                dd, _ = trees[k].query(P, k=1, distance_upper_bound=TRT * TOL)
                npk += np.isfinite(dd)
            keep = keep[(npk >= TRIP) & (sc[keep] >= SCMIN)]
        ncand += len(keep)
        if len(keep) > 2 * NS:
            keep = keep[np.argsort(-sc[keep])[:2 * NS]]
        fi = len(frames)
        frames.append((TD, tref, oref))
        for j in keep:
            rows.append((float(sc[j]), d, float(C[j, 2]), float(C[j, 3]), float(C[j, 0]), float(C[j, 1]), fi,
                         int(ng[j])))
        nd += 1
        if len(rows) > 20 * NS:
            rows.sort(key=lambda r: -r[0]); rows = rows[:4 * NS]
    del NMs
    log_(f"night seeds: {nd}/{len(dg)} distances, {ncand} 3-night candidates")
    rows.sort(key=lambda r: -r[0])
    return rows, frames


BAND_F = {"g": 0.8, "r": 1.0, "i": 1.0, "z": 0.8, "Y": 0.5, "VR": 1.15}


def depth_weights(rows, res, use):
    """gen 19: per-exposure depth weight ~ expected S/N of a fixed-magnitude point source.
    Primary: 10^(0.4 zp) / (sky sigma * fwhm) (zp from the image header, else exposures.csv magzero; sky sigma
    from the CCDs, fwhm in px).  Fallback (no zp): sqrt(exptime)/seeing * band factor, calibrated onto the primary
    scale by the median ratio over exposures that have both.  Normalised to median 1, clipped to [0.25, 2.5]."""
    def fnum(x):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return None
    prim, fb = {}, {}
    for i in use:
        r = rows[i]; m = res[i]
        zp = m.get("zp")
        if zp is None:
            zp = fnum(r.get("magzero"))
            if zp is not None and not (15 < zp < 40):
                zp = None
        sky, fw = m.get("sky"), m.get("fwhm")
        if zp is not None and sky is not None and fw is not None and np.isfinite(sky) and np.isfinite(fw) \
                and sky > 0 and fw > 0:
            prim[i] = 10 ** (0.4 * (zp - 30.0)) / (sky * fw)
        et = fnum(r.get("exptime")) or 0.0
        see = fnum(r.get("seeing_arcsec"))
        if not see or see <= 0.3:
            see = (fw * 0.263) if (fw is not None and np.isfinite(fw) and fw > 0) else 1.2
        b = (r.get("band") or "").strip()
        fb[i] = math.sqrt(max(et, 1.0)) / see * BAND_F.get(b, BAND_F.get(b[:1], 1.0))
    both = [i for i in prim if i in fb]
    k = float(np.median([prim[i] / fb[i] for i in both])) if both else 1.0
    raw = np.array([prim[i] if i in prim else k * fb[i] for i in use])
    raw = raw / np.median(raw)
    w = np.clip(raw, 0.25, 2.5)
    log(f"depth weights: {len(prim)}/{len(use)} exposures with zero point")
    return {i: float(x) for i, x in zip(use, w)}


def dedup(cands, score_states, t_ep):
    """Drop candidates whose position at t_ep is within 4" of a better one (same body)."""
    cands = sorted(cands, key=lambda c: -c["p"])
    keep, dirs = [], []
    for c in cands:
        s = np.array(c["state"])
        u = s[:3] / np.linalg.norm(s[:3])
        if any(float(u @ d) > math.cos(6 * ARCR) for d in dirs):
            continue
        keep.append(c); dirs.append(u)
        if len(keep) >= 200:
            break
    return keep


def cv_rerank(cands, R, rkd, rsc, cem, score_states, total, t, night, nights_u, budget, T0, log_):
    """gen 34: two-fold cross-validated stack S/N.  Exposures are split into halves A/B, alternating in time order
    inside every night (so both halves span all nights and the orbit fit does not extrapolate).  The state is
    re-fitted (CEM from the candidate state, full sigmas) on A only and its S/N measured on B, and vice versa;
    cv = (S/N_B + S/N_A) / sqrt(2).  For a real body cv ~ its all-exposure S/N; for refined noise ~ N(0,1), because
    the fit never sees the half it is scored on.  Candidates are re-ordered by z = CV_A * z_cv + (1 - CV_A) * z_tot
    (each z against the null = refined window seeds, kind 2) and keep the old sorted p values."""
    N = len(t)
    half = np.zeros(N, bool)
    for ni, u in enumerate(nights_u):
        idx = np.flatnonzero(night == u)
        idx = idx[np.argsort(t[idx])]
        for k, e in enumerate(idx):
            half[e] = ((k + ni) % 2) == 0
    mA, mB = half, ~half
    if mA.sum() < 2 or mB.sum() < 2:
        return cands
    KC = int(os.environ.get("TNO_CV_K", "128")); IT = int(os.environ.get("TNO_CV_IT", "5"))
    W_ = float(os.environ.get("TNO_CV_A", "0.7"))
    dl = budget * float(os.environ.get("TNO_CV_DL", "0.965"))

    def cvs(S):
        out = []
        for s in S:
            sA = cem(np.array(s, float), mA, KC, IT)
            sB = cem(np.array(s, float), mB, KC, IT)
            v, c = score_states(np.array([sA, sB]))
            vb = (v[0] * mB, c[0] & mB); va = (v[1] * mA, c[1] & mA)
            zb = total(vb[0][None], vb[1][None])[0]; za = total(va[0][None], va[1][None])[0]
            out.append((za + zb) / math.sqrt(2))
        return np.array(out)

    t0 = time.time()
    nm_ = (rkd == 2) if (rkd == 2).sum() >= 30 else (rkd >= 2)
    nul = R[nm_] if nm_.sum() >= 30 else R
    rr = np.random.default_rng(7)
    nul = nul[rr.permutation(len(nul))[:int(os.environ.get("TNO_CV_NNULL", "60"))]]
    Sc = np.array([c["state"] for c in cands], float)
    vt, ct = score_states(np.concatenate([nul, Sc]) if len(Sc) else nul)
    tot_all = total(vt, ct)
    tn, tc = tot_all[:len(nul)], tot_all[len(nul):]
    cvn = []
    for s in nul:
        if time.time() - T0 > dl:
            break
        cvn.append(cvs([s])[0])
    if len(cvn) < 20:
        log_(f"cv rerank: only {len(cvn)} null evaluations; skipped")
        return cands
    cvn = np.array(cvn)
    mn, sn = np.median(cvn), 1.4826 * np.median(np.abs(cvn - np.median(cvn))) + 1e-3
    tn = rsc[nm_] if nm_.sum() >= 30 else rsc     # same null as to_cands
    mt, st = np.median(tn), 1.4826 * np.median(np.abs(tn - np.median(tn))) + 1e-3
    order = [int(j) for j in np.argsort([-c["p"] for c in cands], kind="stable")]
    zs, done = {}, 0
    for j in order:                       # evaluated in p order; the rest keep their p and slot
        if time.time() - T0 > dl:
            break
        zt = (tc[j] - mt) / st
        cv = cvs([Sc[j]])[0]
        zs[j] = W_ * (cv - mn) / sn + (1 - W_) * zt
        done += 1
    if done < 2:
        log_(f"cv rerank: {done} candidates evaluated; skipped")
        return cands
    # re-ORDER only: the evaluated candidates (a prefix of the p order) are sorted by the blended z and take over
    # that prefix's sorted p values, so the scenario's p distribution (cross-scenario calibration, merge boosts,
    # wrapper interleave) is unchanged and only who holds which rank changes.
    E = order[:done]
    pv = sorted([float(cands[j]["p"]) for j in E], reverse=True)
    Enew = sorted(E, key=lambda j: -zs[j])
    out = [dict(cands[j], p=pv[k]) for k, j in enumerate(Enew)] + [cands[j] for j in order[done:]]
    log_(f"cv rerank: null cv median {mn:.2f} MAD {sn:.2f} (n {len(cvn)}), tot median {mt:.2f} MAD {st:.2f}; "
         f"{done}/{len(cands)} candidates evaluated in {time.time() - t0:.1f}s; top p "
         f"{[round(c['p'], 3) for c in sorted(out, key=lambda c: -c['p'])[:8]]}")
    return out


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        import traceback
        traceback.print_exc()
        log("fatal error; keeping last OUT.json")
