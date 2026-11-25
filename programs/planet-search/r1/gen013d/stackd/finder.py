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
    un = np.unique(night_all)
    per = [sorted(np.flatnonzero(night_all == u), key=lambda i: t_all[i]) for u in un]
    order = []
    k = 0
    while len(order) < min(MAXE, len(rows)):
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
        if k > len(rows) + 3:
            break
    log(f"device {dev}; {len(rows)} exposures, using {len(order)}; nights {len(un)}; scale {SCALE:.2f}")

    # ---------------------------------------------------------------- 1. maps (worker processes)
    import mapper
    from multiprocessing import get_context
    nw = n_workers(cuda)
    ctx = get_context("spawn")
    pool = ctx.Pool(nw)
    tasks = [(int(i), os.path.join(sdir, rows[i]["image"]), os.path.join(sdir, rows[i]["mask"])) for i in order]
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
    use = sorted(res.keys(), key=lambda i: t_all[i])
    log(f"mapped {len(use)} exposures")
    if len(use) < 3:
        return
    t = t_all[use]
    night = night_all[use]
    obs = np.array([[float(rows[i]["obs_x"]), float(rows[i]["obs_y"]), float(rows[i]["obs_z"])] for i in use])
    N = len(use)

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
    for n, i in enumerate(use):
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
    tbin = np.floor((t - t.min()) / 0.06).astype(int)     # 1.4 h bins: a TNO moves > 3" between bins
    for u_ in np.unique(tbin):
        idx = np.flatnonzero(tbin == u_)
        b = torch.zeros((H, W), dtype=torch.bool, device=dev)
        for n in idx:
            b |= maps[n] > 4
        b = F.max_pool2d(b[None, None].half(), 3, 1, 1)[0, 0] > 0
        nn += b.to(torch.int16)
    static = nn >= 2
    log(f"static fraction {float(static.float().mean()):.4f}")
    maps = torch.where(static[None], torch.zeros_like(maps), maps).clamp_(-3, 5)
    del nn, static
    write_out(a.out, [])

    # ---------------------------------------------------------------- 3. search window
    nights_u = np.unique(night)
    tn = np.array([t[night == u].mean() for u in nights_u])
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

    def dgrid(sel, tr, lo=1 / 120.0, hi=1 / 26.0):
        o = obs[sel]; ts = t[sel] - tr
        A = np.stack([np.ones_like(ts), ts], 1)
        coef, *_ = np.linalg.lstsq(A, o, rcond=None)
        r = o - A @ coef
        r = r - np.outer(r @ TF0[2], TF0[2])
        rmax = max(np.sqrt((r ** 2).sum(1)).max(), 1e-6)
        step = 1.0 * SP / rmax
        nst = max(1, int(math.ceil((hi - lo) / step)))
        g = [1 / (lo + (k + 0.5) * (hi - lo) / nst) for k in range(nst)]
        if not g:
            g = [40.0]
        # prior order: classical belt first
        return sorted(g, key=lambda d: abs(math.log(d / 42.0)))

    wins = []
    for a_ in range(len(nights_u)):
        for b_ in range(a_ + 2, len(nights_u)):
            nsel = nights_u[a_:b_ + 1]
            # at most 3 exposures per night (spread)
            sel = []
            for u in nsel:
                idx = np.flatnonzero(night == u)
                if len(idx) > 3:
                    idx = idx[[0, len(idx) // 2, len(idx) - 1]]
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
    # gen 13d: a final ALL-nights window (weight-4 bodies are below the seed threshold in any 3-4 night window),
    # restricted to the most probable orbits: classical-belt distances, velocities near prograde circular.
    # Searched only in the time the other windows leave before the search deadline (deadline-cut, prior-ordered).
    FULL = os.environ.get("TNO_FULL", "1" if cuda else "0") == "1"
    if FULL and len(nights_u) >= 4:
        kpn = 3 if 3 * len(nights_u) <= 40 else 2          # GPU memory: d-frame maps of <= ~40 exposures
        selF = []
        for u in nights_u:
            idx = np.flatnonzero(night == u)
            if len(idx) > kpn:
                idx = idx[np.linspace(0, len(idx) - 1, kpn).round().astype(int)]
            selF += list(idx)
        selF = np.array(selF)
        if len(np.unique(night[selF])) > max(c_[2] - c_[1] + 1 for c_ in chosen):
            chosen.append(((0, 0), 0, len(nights_u) - 1, selF, "full"))
    best_all = []          # list of (score, d, vx, vy, px, py, frame index) in d-frame coords
    frames = []            # (TD, tref, oref) per (window, distance)
    wi_of = []
    search_deadline = budget * 0.72
    nhyp = 0
    z0 = TF0[2]
    pole = np.array([0, -math.sin(EPS), math.cos(EPS)])
    t_search0 = time.time()
    for wi, w in enumerate(chosen):
      el_ = time.time() - T0
      isfull = len(w) > 4
      if wi > 0 and not isfull:
          # time left must cover this window's estimated cost (scaled from the first window's rate)
          rate = (time.time() - t_search0) / max(nhyp, 1)
          if el_ + rate * window_cost(w[3]) * 0.5 > search_deadline and not os.environ.get("TNO_WINFORCE"):
              log(f"skip window {wi}: not enough time")
              continue
      sel = w[3]
      ts = t[sel]; tref = 0.5 * (ts.min() + ts.max()); mdt = max(ts.max() - tref, 0.05)
      dv = 1.0 / mdt
      dg = dgrid(sel, tref, 1 / 52.0, 1 / 34.0) if isfull else dgrid(sel, tref)
      vrad_w = float(os.environ.get("TNO_FULL_VRAD", "0.3")) if isfull else VRAD
      log(f"window {wi}: {len(np.unique(night[sel]))} nights, {len(sel)} exposures, span {ts.max() - ts.min():.2f} d, "
          f"est hyps {window_cost(sel):.0f}, {len(dg)} distances{' (full, vrad %.2f)' % vrad_w if isfull else ''}")
      smaps = F.max_pool2d(maps[sel].float()[None], 2, 2)[0].half()       # (n, H2, W2)
      H2, W2 = smaps.shape[1:]
      oref = obs[sel].mean(0)
      if wi == 0:
          tref0, oref0, dv0 = tref, oref, dv
      for di, d in enumerate(dg):
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
          Dm = torch.zeros((len(sel), HD, WD), dtype=torch.float16, device=dev)
          for k_, n in enumerate(sel):
              rows_ = []
              for y0_ in range(0, HD, 1024):
                  gy_ = gyd[y0_:y0_ + 1024]
                  Wd = TD_t[2][None, None] + gxd[None, :, None] * TD_t[0][None, None] + gy_[:, None, None] * TD_t[1][None, None]
                  Wd = Wd / torch.linalg.norm(Wd, dim=-1, keepdim=True)
                  X = d * Wd - torch.tensor(obs[n] - 0.0, dtype=torch.float32, device=dev)
                  q = X @ TF0_t.T
                  sx = q[..., 0] / q[..., 2]; sy = q[..., 1] / q[..., 2]
                  ux = (sx - x0) / (2 * PIX) - 0.25; uy = (sy - y0) / (2 * PIX) - 0.25
                  grid = torch.stack([2 * ux / (W2 - 1) - 1, 2 * uy / (H2 - 1) - 1], -1)[None]
                  rows_.append(F.grid_sample(smaps[k_].float()[None, None], grid, mode="nearest",
                                             padding_mode="zeros", align_corners=True)[0, 0].half())
              Dm[k_] = torch.cat(rows_, 0)
          # velocities: grid in search px/day; ordered by closeness to prograde circular
          vc = math.sqrt(GM / d) * np.cross(pole, w0); vc = vc / np.linalg.norm(vc) * math.sqrt(GM / d)
          vcx, vcy = (TD[0] @ vc) / d / SP, (TD[1] @ vc) / d / SP
          nv = int(vmax / dv) + 1
          vv = np.arange(-nv, nv + 1) * dv
          VX, VY = np.meshgrid(vv, vv)
          vcs = math.hypot(vcx, vcy)
          ok = (VX ** 2 + VY ** 2 <= (vmax + dv) ** 2) & ((VX - vcx) ** 2 + (VY - vcy) ** 2 <= (vrad_w * vcs + dv) ** 2)
          VX, VY = VX[ok], VY[ok]
          o_ = np.argsort((VX - vcx) ** 2 + (VY - vcy) ** 2)
          VX, VY = VX[o_], VY[o_]
          dts = t[sel] - tref
          P = mx
          Hc, Wc = HD - 2 * P, WD - 2 * P
          best = torch.full((Hc, Wc), -1e4, dtype=torch.float16, device=dev)
          bidx = torch.zeros((Hc, Wc), dtype=torch.int32, device=dev)
          norm = 1.0 / math.sqrt(len(sel))
          t1 = time.time()
          for h in range(len(VX)):
              if (h & 63) == 0 and time.time() - T0 > search_deadline:
                  break
              acc = None
              for k_ in range(len(sel)):
                  sx = int(round(VX[h] * dts[k_] - offs[k_, 0] * 0)); sy = int(round(VY[h] * dts[k_]))
                  sl = Dm[k_, P + sy:P + sy + Hc, P + sx:P + sx + Wc]
                  acc = sl.clone() if acc is None else acc.add_(sl)
              better = acc > best
              best = torch.where(better, acc, best)
              bidx = torch.where(better, torch.full_like(bidx, h), bidx)
              nhyp += 1
          if dev == "mps":
              torch.mps.synchronize()
          try:
              bs = (best.float() * norm).cpu()
              bic = bidx.cpu()
              pk = (bs == F.max_pool2d(bs[None, None], 9, 1, 4)[0, 0]) & (bs > 4.5)
              yy, xx = torch.nonzero(pk, as_tuple=True)
              sc = bs[yy, xx]
              if sc.numel() > 400:
                  top = torch.topk(sc, 400).indices
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
              f"peaks {len(sc)}, max {sc.max() if len(sc) else 0:.2f}")
          del Dm, best, bidx
    log(f"search done: {nhyp} hypotheses, {len(best_all)} raw peaks")
    # gen 11e: seeds from several windows: round-robin by within-window rank (score as tie-break), so a later
    # window's seeds are not crowded out by the first window's noise peaks.
    NSEED = int(1500 * min(2.0, SCALE))
    wof = [wi_of[r[6]] for r in best_all]
    rk = np.zeros(len(best_all))
    for wv_ in set(wof):
        ii = [j for j in range(len(best_all)) if wof[j] == wv_]
        ii.sort(key=lambda j: -best_all[j][0])
        for r_, j in enumerate(ii):
            rk[j] = r_
    oo = sorted(range(len(best_all)), key=lambda j: (rk[j], -best_all[j][0]))[:NSEED]
    best_all = [best_all[j] for j in oo]
    log(f"seeds: {len(best_all)} from windows {sorted(set(wof))}")

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

    def total(v, c):
        nc = np.maximum(c.sum(1), 1)
        return v.sum(1) / np.sqrt(nc)

    def to_cands(S, sc, nn_):
        out = []
        med = np.median(sc); mad = 1.4826 * np.median(np.abs(sc - med)) + 1e-3
        z_ = (sc - med) / mad
        for j in range(len(S)):
            if nn_[j] < 3 or not np.all(np.isfinite(S[j])):
                continue
            r = np.linalg.norm(S[j, :3]); v2 = S[j, 3:] @ S[j, 3:]
            if r < 25 or 0.5 * v2 - GM / r >= 0:
                continue
            p = 1 / (1 + math.exp(-(z_[j] - 4.5) / 0.8)) * (1.0 if nn_[j] >= 4 else 0.8)
            out.append(dict(epoch_mjd=float(t_ep), state=[float(x) for x in S[j]], p=float(min(max(p, 1e-4), 0.999))))
        return out

    def nights_good(v, c):
        g = np.zeros(len(v), int)
        for idx in nights_idx:
            g += (v[:, idx].max(1) >= 2.5).astype(int)
        return g

    if len(states) == 0:
        return
    # dedup seeds by predicted positions on window exposures
    v0, c0 = score_states(states)
    write_out(a.out, to_cands(states, total(v0, c0), nights_good(v0, c0)))
    log(f"wrote {len(states)} seed states (pre-refinement)")

    rng = np.random.default_rng(1)
    K = 384 if cuda else 192
    refined, rsc, rng_ = [], [], []
    order_ = np.argsort(-seed_sc)
    ref_deadline = budget * 0.95
    for jj, j in enumerate(order_):
        if time.time() - T0 > ref_deadline:
            break
        s = states[j].copy()
        r = np.linalg.norm(s[:3])
        TL = tframe(s[:3])
        # sigmas: position 1.5" (at distance r), tangential velocity, radial distance
        sig = np.array([1.5 * ARCR * r, 1.5 * ARCR * r, 0.6 * dv * SP * r, 0.6 * dv * SP * r, 0.03 * r])
        mu = np.zeros(5)
        bestS, bestv = s, -1e9
        for itn in range(6):
            Z = mu[None] + rng.standard_normal((K, 5)) * sig[None]
            Z[0] = mu
            S = np.repeat(s[None], K, 0)
            S[:, :3] += Z[:, 0:1] * TL[0] + Z[:, 1:2] * TL[1] + Z[:, 4:5] * TL[2] * 0
            # distance change: scale position along line of sight from the reference observer
            los = S[:, :3] - oref; losn = los / np.linalg.norm(los, axis=1)[:, None]
            S[:, :3] += Z[:, 4:5] * losn
            S[:, 3:] += Z[:, 2:3] * TL[0] + Z[:, 3:4] * TL[1]
            v, c = score_states(S)
            tot = total(v, c)
            el = np.argsort(-tot)[:max(8, K // 16)]
            if tot[el[0]] > bestv:
                bestv, bestS = tot[el[0]], S[el[0]].copy()
            mu = Z[el].mean(0); sig = np.maximum(Z[el].std(0), sig * 0.3)
        v, c = score_states(bestS[None])
        refined.append(bestS); rsc.append(total(v, c)[0]); rng_.append(nights_good(v, c)[0])
        if (jj % 50) == 49:
            R = np.array(refined)
            write_out(a.out, dedup(to_cands(R, np.array(rsc), np.array(rng_)), score_states, t_ep))
    if refined:
        R = np.array(refined)
        cands = to_cands(R, np.array(rsc), np.array(rng_))
        cands = dedup(cands, score_states, t_ep)
        write_out(a.out, cands)
        log(f"seed-score median {np.median(rsc):.2f}, max {np.max(rsc):.2f}; refined {len(refined)}; output {len(cands)}; top p {[round(c['p'], 3) for c in sorted(cands, key=lambda c: -c['p'])[:8]]}")


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


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        import traceback
        traceback.print_exc()
        log("fatal error; keeping last OUT.json")
