"""gen 104 (arm B): fast full-span shift-and-stack (TNO_FSTK=1).

All mapped frames, night-collapsed (frames within GSPAN d summed after the d-frame warp, as the deep window does), are
searched over a classical-belt prior (d DMIN..DMAX au, |v| in [VLO, VHI] v_circ, direction within ANG deg of prograde
circular) on the 2" max-pooled, winsorized (+-DCLIP) S/N maps.

Kernel: each night co-add (int8-quantised by default) is a flat 1-D array with row stride WD, so the shifted view for a
hypothesis is ONE contiguous range; K contiguous adds + one maximum per hypothesis (the argmax hypothesis is recovered
afterwards only at the kept peaks, by a gather over all searched hypotheses).  Measured on MPS ~4x faster per add than
strided 2-D slices; a one-hot conv2d batching of velocities was tried and is ~15x SLOWER on MPS (not kept).

Returns rows (score, d, vx, vy, px, py, frame index) in the same units as the window search (best_all) and frames
(TD, tref, oref).  Never raises (caller wraps it too).
"""
import math, os, time, json
import numpy as np


def fstk_search(torch, F, dev, maps, obs, t, wf, DW, TF0, TF0_t, x0, y0, PIX, SP, z0, pole, GM, tframe, T0,
                deadline, log, fi0, C_AU=173.1446, orbit=None):
    rows_out, frames_out = [], []
    DMIN = float(os.environ.get("TNO_FSTK_DMIN", "30")); DMAX = float(os.environ.get("TNO_FSTK_DMAX", "70"))
    VLO = float(os.environ.get("TNO_FSTK_VLO", "0.85")); VHI = float(os.environ.get("TNO_FSTK_VHI", "1.12"))
    ANG = math.radians(float(os.environ.get("TNO_FSTK_ANG", "25")))
    GSPAN = float(os.environ.get("TNO_FSTK_GSPAN", "0.08"))
    DCLIP = float(os.environ.get("TNO_FSTK_CLIP", "4.0"))
    BS = 0
    I8 = os.environ.get("TNO_FSTK_I8", "1") != "0"
    NPK = int(os.environ.get("TNO_FSTK_NPK", "300"))
    PTH = float(os.environ.get("TNO_FSTK_PTHR", "3.0"))
    DSTEP = float(os.environ.get("TNO_FSTK_DSTEP", "1.0"))     # distance step in units of the dgrid_deep step
    MAXD = int(os.environ.get("TNO_FSTK_MAXD", "0") or 0)       # debug: at most this many distances
    CHB = float(os.environ.get("TNO_FSTK_CHMB", "384")) * 1e6   # conv output chunk size (bytes)
    probes = []
    try:
        pf = os.environ.get("TNO_FSTK_PROBE")    # diagnostic only: json list of {state, epoch_mjd, tag}
        if pf and os.path.exists(pf):
            probes = json.load(open(pf))
    except Exception:
        probes = []
    t_all0 = time.time()
    N = maps.shape[0]
    sel = np.arange(N)
    ts = t[sel]; tref = 0.5 * (ts.min() + ts.max()); mdt = max(ts.max() - tref, 0.05)
    dv = 1.0 / mdt
    oref = obs[sel].mean(0)
    # 2" pooled, winsorized maps
    sm_ = []
    for i_ in range(0, N, 4):
        a_ = F.max_pool2d(maps[i_:i_ + 4].float()[None], 2, 2)[0]
        if DCLIP > 0:
            a_ = a_.clamp_(-DCLIP, DCLIP)
        sm_.append(a_.half())
    smaps = torch.cat(sm_, 0); del sm_
    H2, W2 = smaps.shape[1:]
    # distance grid (dgrid_deep step logic)
    o = obs[sel]; tt = ts - tref
    A = np.stack([np.ones_like(tt), tt], 1)
    coef, *_ = np.linalg.lstsq(A, o, rcond=None)
    r = o - A @ coef
    r = r - np.outer(r @ TF0[2], TF0[2])
    rmax = max(np.sqrt((r ** 2).sum(1)).max(), 1e-6)
    step = DSTEP * SP / rmax
    lo, hi = 1 / DMAX, 1 / DMIN
    nst = max(1, int(math.ceil((hi - lo) / step)))
    dg = [1 / (lo + (k + 0.5) * (hi - lo) / nst) for k in range(nst)]
    dg = sorted(dg, key=lambda d: abs(math.log(d / 42.0)))
    DLIST = [float(x) for x in os.environ.get("TNO_FSTK_DLIST", "").split(",") if x.strip()]   # debug: explicit distances
    if DLIST:
        dg = DLIST
    if MAXD > 0:
        dg = dg[:MAXD]
    # night collapse
    dts = t[sel] - tref
    o_t = np.argsort(dts)
    grp_, cur_ = [], [int(o_t[0])]
    for j_ in o_t[1:]:
        if dts[j_] - dts[cur_[0]] > GSPAN:
            grp_.append(cur_); cur_ = []
        cur_.append(int(j_))
    grp_.append(cur_)
    gid_ = np.zeros(N, int)
    wsel = np.asarray(wf, float)[sel]
    dts_c = np.zeros(len(grp_))
    for g_, mm_ in enumerate(grp_):
        gid_[mm_] = g_
        ww_ = wsel[mm_] ** 2 if DW else np.ones(len(mm_))
        dts_c[g_] = float((dts[mm_] * ww_).sum() / max(ww_.sum(), 1e-9))
    K = len(grp_)
    norm = 1.0 / math.sqrt(float((wsel ** 2).sum())) if DW else 1.0 / math.sqrt(N)
    log(f"fstk: {N} frames -> {K} co-adds, {len(dg)} distances {DMIN}-{DMAX} au, map {W2}x{H2}, "
        f"{deadline - (time.time() - T0):.0f}s")
    ntot = 0; tconv = 0.0
    pstat = []
    for di, d in enumerate(dg):
        if time.time() - T0 > deadline:
            log(f"fstk: deadline at distance {di}/{len(dg)}")
            break
        Dm = None
        try:
            t_d0 = time.time()
            b = oref @ z0; rho = -b + math.sqrt(b * b - oref @ oref + d * d)
            w0 = (oref + rho * z0) / d
            TD = tframe(w0)
            vmax = math.sqrt(2 * GM / d) / d / SP
            offs = []
            for n in sel:
                o_ = obs[n]; b_ = o_ @ z0; rr = -b_ + math.sqrt(b_ * b_ - o_ @ o_ + d * d)
                q = TD @ ((o_ + rr * z0) / d)
                offs.append([q[0] / q[2] / SP, q[1] / q[2] / SP])
            offs = np.array(offs)
            mx = int(np.abs(offs).max() + vmax * mdt + 8)
            HD, WD = H2 + 2 * mx, W2 + 2 * mx
            TD_t = torch.tensor(TD, dtype=torch.float32, device=dev)
            gxd = (torch.arange(WD, device=dev, dtype=torch.float32) - (WD - 1) / 2) * SP
            gyd = (torch.arange(HD, device=dev, dtype=torch.float32) - (HD - 1) / 2) * SP
            Dm = torch.zeros((K, HD, WD), dtype=torch.float16, device=dev)
            for k_, n in enumerate(sel):
                rows_ = []
                src_ = smaps[k_].float()
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
                    del Wd, X, q, sx, sy, ux, uy, grid
                fr_ = torch.cat(rows_, 0)
                if DW:
                    fr_ *= float(wsel[k_])
                Dm[int(gid_[k_])] += fr_
                del fr_, rows_, src_
            # zero-median co-adds on covered px (max-pooled S/N is biased positive)
            for k_ in range(K):
                v_ = Dm[k_, ::3, ::3].float(); v_ = v_[v_ != 0]
                if v_.numel() > 100:
                    m_ = float(torch.median(v_))
                    Dm[k_] = torch.where(Dm[k_] != 0, Dm[k_] - m_, Dm[k_])
                del v_
            t_warp = time.time() - t_d0
            # velocity grid (search px / day) inside the prior, ordered by closeness to prograde circular
            vc = np.cross(pole, w0); vc = vc / np.linalg.norm(vc) * math.sqrt(GM / d)
            vcx, vcy = (TD[0] @ vc) / d / SP, (TD[1] @ vc) / d / SP
            vcs = math.hypot(vcx, vcy)
            nv = int(VHI * vcs / dv) + 2
            ii = np.arange(-nv, nv + 1)
            IX, IY = np.meshgrid(ii, ii)
            VX, VY = (IX * dv).ravel(), (IY * dv).ravel()
            sp_ = np.hypot(VX, VY)
            ang_ = np.abs(np.arctan2(VX * vcy - VY * vcx, VX * vcx + VY * vcy))
            ok = (sp_ >= VLO * vcs - dv) & (sp_ <= VHI * vcs + dv) & (ang_ <= ANG + dv / max(vcs, 1e-9))
            VX, VY = VX[ok], VY[ok]
            o_ = np.argsort((VX - vcx) ** 2 + (VY - vcy) ** 2)
            VX, VY = VX[o_], VY[o_]
            nprior = len(VX)
            P = mx
            Hc, Wc = H2, W2
            # gen 104: flat contiguous kernel.  Each co-add is a 1-D array with row stride WD; the shifted (Hc, Wc) view
            # of shift (sx, sy) is the contiguous range [(P+sy)*WD + P+sx, ... + Hc*WD) evaluated at full row width
            # (columns >= Wc are margin garbage, discarded).  Contiguous adds run ~4x faster than strided slices on MPS.
            # Optional int8 (TNO_FSTK_I8, default 1): scale so the sum of the co-adds' max |value| fits in int8.
            amax = [float(Dm[k_].abs().max()) for k_ in range(K)]
            qs = 127.0 / max(sum(amax), 1e-6)
            use8 = I8 and qs >= 1.5
            if use8:
                Dq = torch.round(Dm.float().mul_(qs)).to(torch.int8).reshape(K, -1)
                qn = norm / qs
            else:
                Dq = Dm.reshape(K, -1); qn = norm
            del Dm; Dm = None
            L = Hc * WD
            best = torch.full((L,), -127 if use8 else -1e4, dtype=Dq.dtype, device=dev)
            acc = torch.empty((L,), dtype=Dq.dtype, device=dev)
            SX = np.round(np.outer(VX, dts_c)).astype(np.int64)      # (nh, K)
            SY = np.round(np.outer(VY, dts_c)).astype(np.int64)
            OFF = (P + SY) * WD + (P + SX)
            t_c0 = time.time()
            nh = 0
            for h in range(len(VX)):
                if (h & 63) == 0 and time.time() - T0 > deadline:
                    break
                of_ = OFF[h]
                if K == 1:
                    acc.copy_(Dq[0, of_[0]:of_[0] + L])
                else:
                    torch.add(Dq[0, of_[0]:of_[0] + L], Dq[1, of_[1]:of_[1] + L], out=acc)
                    for k_ in range(2, K):
                        acc.add_(Dq[k_, of_[k_]:of_[k_] + L])
                torch.maximum(best, acc, out=best)
                nh += 1
            if dev == "mps":
                torch.mps.synchronize()
            elif dev == "cuda":
                torch.cuda.synchronize()
            dtc = time.time() - t_c0
            tconv += dtc; ntot += nh
            del acc
            # peaks
            bs = best.reshape(Hc, WD)[:, :Wc].float() * qn
            del best
            pk = (bs == F.max_pool2d(bs[None, None], 9, 1, 4)[0, 0]) & (bs > PTH)
            yy, xx = torch.nonzero(pk, as_tuple=True)
            sc = bs[yy, xx]
            nab = [int((sc > z_).sum()) for z_ in (5.0, 6.0, 7.0)]
            if sc.numel() > NPK:
                top = torch.topk(sc, NPK).indices
                yy, xx, sc = yy[top], xx[top], sc[top]
            # best hypothesis at each kept peak: re-evaluate every searched hypothesis there (gather)
            hh = np.zeros(len(sc), np.int64)
            if len(sc) and nh > 0:
                pf_ = (yy * WD + xx).to(torch.int64)
                bv_ = torch.full((len(sc),), -1e9, device=dev); bi_ = torch.zeros(len(sc), dtype=torch.int64, device=dev)
                OFF_t = torch.tensor(OFF[:nh], dtype=torch.int64, device=dev)
                for h0 in range(0, nh, 256):
                    v_ = torch.zeros((min(256, nh - h0), len(sc)), device=dev)
                    for k_ in range(K):
                        v_ += Dq[k_][OFF_t[h0:h0 + 256, k_][:, None] + pf_[None]].float()
                    m_, a_ = v_.max(0)
                    bt_ = m_ > bv_
                    bv_ = torch.where(bt_, m_, bv_); bi_ = torch.where(bt_, a_ + h0, bi_)
                hh = bi_.cpu().numpy()
                del OFF_t
            yy, xx, sc = yy.cpu().numpy(), xx.cpu().numpy(), sc.cpu().numpy()
            fi = fi0 + len(frames_out)
            for j in range(len(sc)):
                px = (xx[j] + P - (WD - 1) / 2) * SP; py = (yy[j] + P - (HD - 1) / 2) * SP
                rows_out.append((float(sc[j]), d, VX[hh[j]] * SP, VY[hh[j]] * SP, px, py, fi))
            frames_out.append((TD, tref, oref))
            # diagnostics: probe states (stack at the probe's nearest hypothesis +-1 velocity step, 3x3 px; best-map max)
            for pr in probes:
                try:
                    s_ = np.array(pr["state"], float)
                    lt = 0.0
                    for _ in range(2):
                        x2 = orbit.propagate(s_[:3], s_[3:], np.array([tref - lt - pr["epoch_mjd"], tref - lt - pr["epoch_mjd"] + 1e-3]))
                        lt = np.linalg.norm(x2[0] - oref) / C_AU
                    Xp = x2[0]; Vp = (x2[1] - x2[0]) / 1e-3
                    rp = np.linalg.norm(Xp)
                    if abs(1 / rp - 1 / d) > 0.75 * (hi - lo) / nst and not DLIST:
                        continue
                    if DLIST and abs(rp - d) > 0.3:
                        continue
                    wv = Xp / rp
                    q = TD @ wv
                    ppx, ppy = q[0] / q[2], q[1] / q[2]
                    wn = math.sqrt(1 + ppx ** 2 + ppy ** 2)
                    Vperp = Vp - (Vp @ wv) * wv
                    pvx = (TD[0] @ Vperp) * wn / d / SP; pvy = (TD[1] @ Vperp) * wn / d / SP
                    xi = int(round(ppx / SP + (WD - 1) / 2 - P)); yi = int(round(ppy / SP + (HD - 1) / 2 - P))
                    if not (5 <= xi < Wc - 5 and 5 <= yi < Hc - 5):
                        log(f"fstk probe {pr.get('tag', '?')}: off map"); continue
                    bestv = -1e9
                    for ddx in (-1, 0, 1):
                        for ddy in (-1, 0, 1):
                            vx_ = (round(pvx / dv) + ddx) * dv; vy_ = (round(pvy / dv) + ddy) * dv
                            for ey in (-1, 0, 1):
                                for ex in (-1, 0, 1):
                                    a_ = 0.0
                                    for k_ in range(K):
                                        sxx = int(round(vx_ * dts_c[k_])); syy = int(round(vy_ * dts_c[k_]))
                                        a_ += float(Dq[k_, (P + syy + yi + ey) * WD + P + sxx + xi + ex])
                                    bestv = max(bestv, a_ * qn)
                    bl = float(bs[yi - 2:yi + 3, xi - 2:xi + 3].max())
                    rk = int((sc > bl).sum()) if len(sc) else 0
                    inprior = (VLO * vcs <= math.hypot(pvx, pvy) <= VHI * vcs) and \
                        abs(math.atan2(pvx * vcy - pvy * vcx, pvx * vcx + pvy * vcy)) <= ANG
                    pstat.append((pr.get("tag", "?"), d, bestv, bl, rk))
                    log(f"fstk probe {pr.get('tag', '?')}: d={d:.2f} (r {rp:.2f}), |v|/vc {math.hypot(pvx, pvy) / vcs:.3f}, "
                        f"in prior {inprior}, exact-hyp local max {bestv:.2f}, best-map 5x5 max {bl:.2f}, "
                        f"rank among kept peaks {rk}, map max {float(sc.max()) if len(sc) else 0:.2f}")
                except Exception as ex:
                    log("fstk probe failed:", repr(ex)[:120])
            log(f"fstk d={d:.2f}: {nh}/{nprior} hyps in {dtc:.1f}s ({nh / max(dtc, 1e-6):.0f} hyp/s, K {K}, "
                f"{'int8 q%.2f' % qs if use8 else 'fp16'}), warp {t_warp:.1f}s, max {float(sc.max()) if len(sc) else 0:.2f}, "
                f"peaks>5/6/7 {nab}, kept {len(sc)}")
            del Dq, bs, pk
            Dm = None
        except Exception as ex:
            log(f"fstk failed at d={d:.1f}:", repr(ex)[:300])
            Dm = None
            try:
                if dev == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            break
    del smaps
    log(f"fstk done: {ntot} hypotheses, kernel {tconv:.1f}s ({ntot / max(tconv, 1e-6):.0f} hyp/s), {len(rows_out)} rows, "
        f"{time.time() - t_all0:.1f}s total")
    return rows_out, frames_out
