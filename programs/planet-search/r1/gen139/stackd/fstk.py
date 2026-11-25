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



def _dkey(d):   # gen 135: see stackd/finder.py _dkey (TNO_DORD_K)
    try:
        K_ = float(os.environ.get("TNO_DORD_K", "1.6"))
    except Exception:
        K_ = 1.6
    if not (K_ > 0):
        K_ = 1.6
    return math.log(42.0 / d) if d <= 42.0 else K_ * math.log(d / 42.0)

def fstk_search(torch, F, dev, maps, obs, t, wf, DW, TF0, TF0_t, x0, y0, PIX, SP, z0, pole, GM, tframe, T0,
                deadline, log, fi0, C_AU=173.1446, orbit=None, prior=None, excl=None):
    rows_out, frames_out = [], []
    DMIN = float(os.environ.get("TNO_FSTK_DMIN", "30")); DMAX = float(os.environ.get("TNO_FSTK_DMAX", "70"))
    VLO = float(os.environ.get("TNO_FSTK_VLO", "0.85")); VHI = float(os.environ.get("TNO_FSTK_VHI", "1.12"))
    ANG = math.radians(float(os.environ.get("TNO_FSTK_ANG", "25")))
    # gen 136 (extension passes, see stackd/finder.py TNO_FSTK_EXT): prior = (VLO, VHI, ANG deg) replaces the core prior;
    # excl = (VLO, VHI, ANG deg) of the previous pass, whose velocity cells (same tolerance formula) are skipped, so an
    # extension pass searches exactly the hypotheses no earlier pass searched.
    if prior is not None:
        VLO, VHI, ANG = float(prior[0]), float(prior[1]), math.radians(float(prior[2]))
    XLO = XHI = XANG = None
    if excl is not None:
        XLO, XHI, XANG = float(excl[0]), float(excl[1]), math.radians(float(excl[2]))
    # gen 126 (TNO_FSTK_TIGHT, default 1; 0 = gen125): canvas margin = shift reach only (see the mx line)
    TIGHT = os.environ.get("TNO_FSTK_TIGHT", "1") != "0"
    GSPAN = float(os.environ.get("TNO_FSTK_GSPAN", "0.08"))
    DCLIP = float(os.environ.get("TNO_FSTK_CLIP", "2.5"))  # gen 107: was 4.0
    BS = 0
    I8 = os.environ.get("TNO_FSTK_I8", "1") != "0"
    NPK = int(os.environ.get("TNO_FSTK_NPK", "300"))
    PTH = float(os.environ.get("TNO_FSTK_PTHR", "3.0"))
    DSTEP = float(os.environ.get("TNO_FSTK_DSTEP", "1.0"))     # distance step in units of the dgrid_deep step
    MAXD = int(os.environ.get("TNO_FSTK_MAXD", "0") or 0)       # debug: at most this many distances
    CHB = float(os.environ.get("TNO_FSTK_CHMB", "384")) * 1e6   # conv output chunk size (bytes)
    # gen 109 (arm H): re-score the kept peaks on softer-clipped (RCLIP) co-adds over a local sub-step grid of
    # position (+-0.5 px) and velocity (+-0.5 dv) offsets, continuous track rounded per co-add; keep the best.
    # TNO_FSTK_RS=0 reproduces gen108 exactly.  RPOOL = pool size factor (peaks rescored = RPOOL x NPK, best NPK kept).
    RS = os.environ.get("TNO_FSTK_RS", "1") == "1"
    RCLIP = float(os.environ.get("TNO_FSTK_RCLIP", "3.0"))
    RPOOL = float(os.environ.get("TNO_FSTK_RPOOL", "50") or 1.0)   # gen 115: 1 -> 50 (15000 coarse peaks re-scored, best NPK kept; 1 = gen114)
    RSUB = int(os.environ.get("TNO_FSTK_RSUB", "2") or 2)       # sub-steps per half unit: grid -0.5..0.5 step 0.5/RSUB
    if RS and DCLIP <= 0:
        RS = False
    # gen 139 (TNO_FSTK_NAT, default 1; 0 = gen138): native-resolution re-score.  The best NATN clip-3 re-scored peaks of a
    # slice are scored once more on the UNPOOLED 1" S/N maps, every frame at its own time, along a continuous track mapped
    # through the exact ray geometry (no 2x2 max-pool, no nearest-cell warp, no per-co-add integer shift rounding), over a
    # local grid of +-1 cell position (0.5 steps) x +-1 dv velocity (0.25 steps), then +-0.25 cell x +-0.125 dv around the best; the best NPK are kept
    # (rows carry the refined position / velocity).  (TNO_FSTK_NPR/NPS/NVR/NVS set the first grid; TNO_FSTK_NATN the pool, TNO_FSTK_NCLIP the clip.)
    NAT = RS and os.environ.get("TNO_FSTK_NAT", "1") != "0"
    NATN = int(float(os.environ.get("TNO_FSTK_NATN", "2000") or 0))
    NCLIP = float(os.environ.get("TNO_FSTK_NCLIP", "3.0") or 3.0)
    probes = []
    PROBE_OLD = os.environ.get("TNO_FSTK_PROBE_OLD", "0") == "1"   # gen 127: 1 = gen126's (wrong at d != r) probe position
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
    # gen 129 (TNO_FSTK_FINE, default 0 = gen128; 1 = on, measured neutral, see NOTES): warp each frame from a STRIDE-1 2x2 max-pool of the 1" map
    # (window i covers 1" px i, i+1, centre i+0.5) instead of the stride-2 2" pool, so the nearest-neighbour sample is
    # within +-0.25 canvas cell of the needed position instead of +-0.5.  Same per-cell noise distribution (2x2 max).
    FINE = os.environ.get("TNO_FSTK_FINE", "0") == "1"
    FINE_MB = float(os.environ.get("TNO_FSTK_FINE_MB", "6000"))   # precompute the stride-1 maps only under this size
    smaps = smaps1 = None
    if FINE:
        LIM_ = max(DCLIP, RCLIP) if RS else DCLIP
        H1, W1 = maps.shape[1] - 1, maps.shape[2] - 1

        def _src1(i_):
            a_ = F.max_pool2d(maps[i_:i_ + 1].float()[None], 2, 1)[0, 0]
            return a_.clamp_(-LIM_, LIM_) if DCLIP > 0 else a_
        H2, W2 = maps.shape[1] // 2, maps.shape[2] // 2      # canvas size as the stride-2 path
        if N * H1 * W1 * 2 < FINE_MB * 1e6:
            try:
                smaps1 = torch.stack([_src1(i_).half() for i_ in range(N)])
            except Exception:
                smaps1 = None
        log(f"fstk fine: stride-1 2x2 max source {W1}x{H1}, {'precomputed' if smaps1 is not None else 'per frame'}")
    else:
        # 2" pooled, winsorized maps
        sm_ = []
        for i_ in range(0, N, 4):
            a_ = F.max_pool2d(maps[i_:i_ + 4].float()[None], 2, 2)[0]
            if DCLIP > 0:
                a_ = a_.clamp_(-max(DCLIP, RCLIP) if RS else -DCLIP, max(DCLIP, RCLIP) if RS else DCLIP)
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
    dg = sorted(dg, key=_dkey)
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
    nat_mu = [None]   # gen 139: per-frame medians of the clipped 1" maps (computed once, lazily)
    nfail = 0   # gen 114: a failing slice is skipped (continue); stop only after 3 failures in a row
    pstat = []
    # gen 127 (TNO_FSTK_ADAPT, default 0 = gen126, NOT shipped: lost the threshold body B9 in a paired run, see NOTES): coverage-adaptive distance grid.  After ADN slices, predict the
    # time the remaining slices need (measured warp + other overhead per slice, + prior hypotheses / measured hyp/s);
    # if they do not fit in the deadline, re-plan the remaining slices once on the smallest coarser step (x1.25 .. x AMAX)
    # that fits (else AMAX), skipping new slices within half a new step of a done one, same _dkey order (gen 135).
    # Corrected truth probes (gen127) show a body's d-frame track is within ~0.4 px of linear at half a DSTEP-1 step and
    # ~0.7 px at half a 1.75 step (B8 stays the slice's #0-1 kept peak at 0.23 au off), while the Mac reaches only
    # 9-33 of 129 slices; so when time-limited, wider distance coverage beats density.  Identical to gen126 when the
    # predicted time fits.
    ADAPT = os.environ.get("TNO_FSTK_ADAPT", "0") != "0" and not DLIST and MAXD == 0
    AMAX = float(os.environ.get("TNO_FSTK_AMAX", "1.75"))
    ADN = int(os.environ.get("TNO_FSTK_ADN", "3"))
    sl_t = []          # per finished slice: (total s, warp s, kernel s, hyps)
    done_d = []
    replanned = False

    def _nprior(d_):
        b_ = oref @ z0; rho_ = -b_ + math.sqrt(b_ * b_ - oref @ oref + d_ * d_)
        w0_ = (oref + rho_ * z0) / d_
        TD_ = tframe(w0_)
        vc_ = np.cross(pole, w0_); vc_ = vc_ / np.linalg.norm(vc_) * math.sqrt(GM / d_)
        vcx_, vcy_ = (TD_[0] @ vc_) / d_ / SP, (TD_[1] @ vc_) / d_ / SP
        vcs_ = math.hypot(vcx_, vcy_)
        nv_ = int(VHI * vcs_ / dv) + 2
        ii_ = np.arange(-nv_, nv_ + 1)
        IX_, IY_ = np.meshgrid(ii_, ii_)
        VX_, VY_ = (IX_ * dv).ravel(), (IY_ * dv).ravel()
        sp2 = np.hypot(VX_, VY_)
        an2 = np.abs(np.arctan2(VX_ * vcy_ - VY_ * vcx_, VX_ * vcx_ + VY_ * vcy_))
        return int(((sp2 >= VLO * vcs_ - dv) & (sp2 <= VHI * vcs_ + dv) & (an2 <= ANG + dv / max(vcs_, 1e-9))).sum())

    di = -1
    while di + 1 < len(dg):
        di += 1
        if ADAPT and not replanned and len(sl_t) >= ADN:
            replanned = True
            try:
                rem_t = deadline - (time.time() - T0)
                st_ = np.array(sl_t, float)
                hr_ = max(st_[:, 3].sum() / max(st_[:, 2].sum(), 1e-6), 1.0)
                ov_ = float(np.mean(st_[:, 0] - st_[:, 2]))

                def _cost(lst):
                    return sum(ov_ + _nprior(x_) / hr_ for x_ in lst)
                need_ = _cost(dg[di:])
                if need_ > 0.97 * rem_t:
                    pick_ = None
                    for f_ in (1.25, 1.5, AMAX):
                        if f_ > AMAX + 1e-9 or f_ <= DSTEP + 1e-9:
                            continue
                        st2 = step * f_ / DSTEP
                        n2 = max(1, int(math.ceil((hi - lo) / st2)))
                        g2 = [1 / (lo + (k + 0.5) * (hi - lo) / n2) for k in range(n2)]
                        g2 = [x_ for x_ in g2 if all(abs(1 / x_ - 1 / y_) > 0.5 * st2 for y_ in done_d)]
                        g2 = sorted(g2, key=_dkey)
                        c2 = _cost(g2)
                        pick_ = (f_, g2, c2)
                        if c2 <= 0.97 * rem_t:
                            break
                    if pick_ is not None:
                        dg = dg[:di] + pick_[1]
                        log(f"fstk adapt: {len(sl_t)} slices done at {hr_:.0f} hyp/s, ov {ov_:.1f}s/slice; remaining need "
                            f"{need_:.0f}s > {rem_t:.0f}s left -> step x{pick_[0]:.2f}: {len(pick_[1])} slices, predicted {pick_[2]:.0f}s")
                else:
                    log(f"fstk adapt: remaining {len(dg) - di} slices need {need_:.0f}s of {rem_t:.0f}s left, grid kept")
            except Exception as ex:
                log("fstk adapt failed:", repr(ex)[:200])
            if di >= len(dg):
                break
        d = dg[di]
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
            # gen 126 (TNO_FSTK_TIGHT): the kernel only reads canvas pixels (row+P+SY, col+P+SX) for the central H2 x W2
            # output block, with |SX|,|SY| <= (VHI*vcs+dv)*mdt+0.5 < vmax*mdt (rescore/probes add <= 1 px + 1 dv, clamped).
            # The footprint offsets |offs| (parallax, ~300 px at 42 au) were never read but widened WD/HD, i.e. the
            # memory-bound kernel length Hc*WD and the HD*WD warp.  Row coordinates (xx+P-(WD-1)/2)*SP do not depend on P.
            mx_old = int(np.abs(offs).max() + vmax * mdt + 8)
            mx = int(vmax * mdt + 8) if TIGHT else mx_old
            HD, WD = H2 + 2 * mx, W2 + 2 * mx
            TD_t = torch.tensor(TD, dtype=torch.float32, device=dev)
            gxd = (torch.arange(WD, device=dev, dtype=torch.float32) - (WD - 1) / 2) * SP
            gyd = (torch.arange(HD, device=dev, dtype=torch.float32) - (HD - 1) / 2) * SP
            Dm = torch.zeros((K, HD, WD), dtype=torch.float16, device=dev)
            Dm3 = None
            if RS:
                try:
                    Dm3 = torch.zeros((K, HD, WD), dtype=torch.float16, device=dev)
                except Exception:
                    Dm3 = None
            for k_, n in enumerate(sel):
                rows_ = []
                if FINE:
                    src_ = smaps1[k_].float() if smaps1 is not None else _src1(k_)
                else:
                    src_ = smaps[k_].float()
                for y0_ in range(0, HD, 1024):
                    gy_ = gyd[y0_:y0_ + 1024]
                    Wd = TD_t[2][None, None] + gxd[None, :, None] * TD_t[0][None, None] + gy_[:, None, None] * TD_t[1][None, None]
                    Wd = Wd / torch.linalg.norm(Wd, dim=-1, keepdim=True)
                    X = d * Wd - torch.tensor(obs[n] - 0.0, dtype=torch.float32, device=dev)
                    q = X @ TF0_t.T
                    sx = q[..., 0] / q[..., 2]; sy = q[..., 1] / q[..., 2]
                    if FINE:   # 1" px i centred at u1 = i; stride-1 window index = u1 - 0.5
                        ux = (sx - x0) / PIX - 0.5; uy = (sy - y0) / PIX - 0.5
                        grid = torch.stack([2 * ux / (W1 - 1) - 1, 2 * uy / (H1 - 1) - 1], -1)[None]
                    else:
                        ux = (sx - x0) / (2 * PIX) - 0.25; uy = (sy - y0) / (2 * PIX) - 0.25
                        grid = torch.stack([2 * ux / (W2 - 1) - 1, 2 * uy / (H2 - 1) - 1], -1)[None]
                    rows_.append(F.grid_sample(src_[None, None], grid, mode="nearest",
                                               padding_mode="zeros", align_corners=True)[0, 0].half())
                    del Wd, X, q, sx, sy, ux, uy, grid
                fr_ = torch.cat(rows_, 0)
                if Dm3 is not None:
                    Dm3[int(gid_[k_])] += (fr_ * float(wsel[k_])) if DW else fr_
                    fr_ = fr_.clamp(-DCLIP, DCLIP)
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
                if Dm3 is not None:
                    v_ = Dm3[k_, ::3, ::3].float(); v_ = v_[v_ != 0]
                    if v_.numel() > 100:
                        m_ = float(torch.median(v_))
                        Dm3[k_] = torch.where(Dm3[k_] != 0, Dm3[k_] - m_, Dm3[k_])
                    del v_
            # gen 126 experiment (TNO_FSTK_DIL, default 0 = off): 3x3 max filter on the coarse co-adds (not Dm3), so a
            # track misaligned by up to 1 pooled px in some co-adds keeps its signal (probes show the coarse stack only
            # survives within ~0.04 au of the true distance).  Untested in full runs; see NOTES.
            if os.environ.get("TNO_FSTK_DIL", "0") != "0":
                for k_ in range(K):
                    Dm[k_] = F.max_pool2d(Dm[k_][None, None].float(), 3, 1, 1)[0, 0].half()
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
            if XLO is not None:
                ok &= ~((sp_ >= XLO * vcs - dv) & (sp_ <= XHI * vcs + dv) & (ang_ <= XANG + dv / max(vcs, 1e-9)))
            if not ok.any():
                log(f"fstk d={d:.2f}: no hypotheses in this prior ring")
                Dm = None; Dm3 = None
                continue
            VX, VY = VX[ok], VY[ok]
            IXo, IYo = IX.ravel()[ok], IY.ravel()[ok]
            o_ = np.argsort((VX - vcx) ** 2 + (VY - vcy) ** 2)
            VX, VY = VX[o_], VY[o_]
            IXo, IYo = IXo[o_], IYo[o_]
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
                Dqf = torch.round(Dm.float().mul_(qs))
                # gen 129: round() can add up to 0.5 per co-add, so the rounded maxima may sum past 127 and the
                # strongest hypothesis wraps to -128 in int8.  Re-quantise only then (identical Dq/qs otherwise).
                if sum(float(Dqf[k_].abs().max()) for k_ in range(K)) > 127.0:
                    del Dqf
                    qs = (127.0 - 0.5 * K) / max(sum(amax), 1e-6)
                    use8 = qs >= 1.5
                    Dqf = torch.round(Dm.float().mul_(qs)) if use8 else None
            if use8:
                Dq = Dqf.to(torch.int8).reshape(K, -1); del Dqf
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
            done_idx = None
            TREE = os.environ.get("TNO_FSTK_TREE", "1") != "0" and use8 and K >= 3
            if TREE:
                # gen 128 (TNO_FSTK_TREE, default 1; 0 = gen127 loop): share partial sums of time-contiguous co-add
                # subsets between nearby velocities (velocity tiles of TB x TB grid cells).  int8 sums are exact and
                # cannot overflow (the rounded per-co-add |max| sum to <= 127, gen 129), so `best` is bit-identical to the flat loop when
                # all hypotheses finish; per hypothesis the top level costs 1 add + 1 max instead of K-1 adds + 1 max.
                try:
                    TB = int(os.environ.get("TNO_FSTK_TREEB", "8") or 8)
                    kb = (IXo // TB) * 100003 + (IYo // TB)
                    _, bid = np.unique(kb, return_inverse=True); bid = bid.ravel()
                    brank = np.full(int(bid.max()) + 1, 1 << 40, np.int64)
                    np.minimum.at(brank, bid, np.arange(len(VX)))
                    S_all = [int(x) for x in np.argsort(dts_c, kind="stable")]
                    SA, SB = _split(S_all, dts_c)
                    done = np.zeros(len(VX), bool)
                    npass = 0
                    for b_ in np.argsort(brank):
                        if time.time() - T0 > deadline:
                            break
                        hs = np.nonzero(bid == b_)[0]
                        OFFb = OFF[hs]
                        cnt = [0]
                        pa, ia = _build(torch, Dq, OFFb, SA, dts_c, L, cnt)
                        pb, ib = _build(torch, Dq, OFFb, SB, dts_c, L, cnt)
                        for j_, h in enumerate(hs):
                            ta, ba = pa[ia[j_]]; tb, bb = pb[ib[j_]]
                            oa = int(OFFb[j_, SA[0]]) - ba; ob = int(OFFb[j_, SB[0]]) - bb
                            torch.add(ta[oa:oa + L], tb[ob:ob + L], out=acc)
                            torch.maximum(best, acc, out=best)
                        npass += cnt[0] + 2 * len(hs)
                        done[hs] = True
                        del pa, pb
                    ta = tb = None
                    done_idx = np.nonzero(done)[0]
                    nh = len(done_idx)
                    tree_info = f", tree {npass / max(nh, 1):.2f} passes/hyp vs {K}"
                except Exception as ex:
                    log("fstk tree failed, flat loop:", repr(ex)[:200])
                    try:
                        if dev == "cuda":
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    best.fill_(-127); done_idx = None; nh = 0; tree_info = ", tree failed"
            else:
                tree_info = ""
            if done_idx is None:
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
                done_idx = np.arange(nh)
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
            fin_ = torch.isfinite(sc) & (sc < 1e3)   # gen 115: drop NaN/overflow peaks
            if not bool(fin_.all()):
                yy, xx, sc = yy[fin_], xx[fin_], sc[fin_]
            nab = [int((sc > z_).sum()) for z_ in (5.0, 6.0, 7.0)]
            NPK_ = int(NPK * RPOOL) if Dm3 is not None else NPK
            sc_full = sc if probes else None
            if sc.numel() > NPK_:
                top = torch.topk(sc, NPK_).indices
                yy, xx, sc = yy[top], xx[top], sc[top]
            # best hypothesis at each kept peak: re-evaluate every searched hypothesis there (gather)
            hh = np.zeros(len(sc), np.int64)
            if len(sc) and nh > 0:
                pf_ = (yy * WD + xx).to(torch.int64)
                bv_ = torch.full((len(sc),), -1e9, device=dev); bi_ = torch.zeros(len(sc), dtype=torch.int64, device=dev)
                OFF_t = torch.tensor(OFF[done_idx], dtype=torch.int64, device=dev)
                for h0 in range(0, nh, 256):
                    v_ = torch.zeros((min(256, nh - h0), len(sc)), device=dev)
                    for k_ in range(K):
                        v_ += Dq[k_][OFF_t[h0:h0 + 256, k_][:, None] + pf_[None]].float()
                    m_, a_ = v_.max(0)
                    bt_ = m_ > bv_
                    bv_ = torch.where(bt_, m_, bv_); bi_ = torch.where(bt_, a_ + h0, bi_)
                hh = done_idx[bi_.cpu().numpy()]
                del OFF_t
            yy, xx, sc = yy.cpu().numpy(), xx.cpu().numpy(), sc.cpu().numpy()
            fi = fi0 + len(frames_out)
            rres = None
            t_rs0 = time.time()
            if Dm3 is not None and len(sc):
                try:
                    rres = _rescore(torch, dev, Dm3, xx.astype(float), yy.astype(float), VX[hh], VY[hh], dts_c, P, WD, HD,
                                    norm, dv, RSUB)
                except Exception as ex:
                    log("fstk rescore failed:", repr(ex)[:200]); rres = None
            if rres is not None:
                s2_, ex_, ey_, evx_, evy_, s0_ = rres
                # gen 115: guard against fp16 overflow / NaN rows (seen once on MPS: re-scored values 6e3-1.5e4)
                bad_ = ~np.isfinite(s2_) | (s2_ > 1e3)
                if bad_.any():
                    s2_ = np.where(bad_, -1e9, s2_)
                o2_ = np.argsort(-s2_)[:NPK]
                o2_ = o2_[s2_[o2_] > -1e8]
                log(f"fstk rescore d={d:.2f} ({time.time() - t_rs0:.1f}s): {len(sc)} peaks, clip{RCLIP} exact max {s0_.max():.2f}, local max {s2_.max():.2f}, "
                    f"median lift {np.median(s2_ - s0_):.2f}, kept {len(o2_)}, "
                    f"overlap with old top {len(set(o2_.tolist()) & set(range(min(NPK, len(sc)))))}")
                nat_ = None
                if NAT and NATN > 0:
                    try:
                        t_n0 = time.time()
                        # cost guard: the native pass loops over all N frames; shrink the pool for dense scenarios
                        natn_ = max(NPK, min(NATN, int(NATN * 24 / max(N, 24))))
                        on_ = np.argsort(-s2_)[:max(natn_, NPK)]
                        on_ = on_[s2_[on_] > -1e8]
                        if len(on_):
                            if nat_mu[0] is None:
                                nat_mu[0] = _nat_med(torch, maps, NCLIP)
                            fs_, fex_, fey_, fevx_, fevy_ = _natscore(
                                torch, dev, maps, obs, dts, wsel, DW, d, TD, TF0, x0, y0, PIX, SP, WD, HD, P,
                                xx[on_] + ex_[on_], yy[on_] + ey_[on_], VX[hh[on_]] + evx_[on_], VY[hh[on_]] + evy_[on_],
                                dv, NCLIP, nat_mu[0])
                            okn_ = np.isfinite(fs_) & (fs_ < 1e3)
                            if okn_.any():
                                fsn_ = np.where(okn_, fs_, -1e9)
                                # gen 139 blend (TNO_FSTK_NBLEND, default 1): a peak keeps its clip-3 score (scaled to
                                # the native scale by the ratio of the two top-NPK medians) when that is higher, with its
                                # clip-3 position / velocity, so bodies the native grid misses are not dropped.
                                if os.environ.get("TNO_FSTK_NBLEND", "1") != "0":
                                    s2o_ = s2_[on_]
                                    kk_ = float(np.median(np.sort(fsn_[okn_])[::-1][:NPK])) / max(float(np.median(np.sort(s2o_)[::-1][:NPK])), 1e-6)
                                    kk_ = kk_ if (np.isfinite(kk_) and 0.5 < kk_ < 3.0) else 1.0
                                    use2_ = (kk_ * s2o_ > fsn_) & (s2o_ > -1e8)
                                    fs_ = np.where(use2_, kk_ * s2o_, fs_); okn_ = okn_ | use2_
                                    fex_ = np.where(use2_, 0.0, fex_); fey_ = np.where(use2_, 0.0, fey_)
                                    fevx_ = np.where(use2_, 0.0, fevx_); fevy_ = np.where(use2_, 0.0, fevy_)
                                    fsn_ = np.where(okn_, fs_, -1e9)
                                o3_ = np.argsort(-fsn_)[:NPK]
                                o3_ = o3_[okn_[o3_]]
                                nat_ = (on_, fs_, fex_, fey_, fevx_, fevy_, o3_)
                                log(f"fstk native d={d:.2f} ({time.time() - t_n0:.1f}s): {len(on_)} peaks, max {fs_[o3_].max():.2f}, "
                                    f"kept {len(o3_)}, overlap with clip{RCLIP} kept {len(set(on_[o3_].tolist()) & set(o2_.tolist()))}")
                    except Exception as ex:
                        log("fstk native rescore failed:", repr(ex)[:200]); nat_ = None
                if nat_ is not None:
                    on_, fs_, fex_, fey_, fevx_, fevy_, o3_ = nat_
                    for q_ in o3_:
                        j = on_[q_]
                        px = (xx[j] + ex_[j] + fex_[q_] + P - (WD - 1) / 2) * SP; py = (yy[j] + ey_[j] + fey_[q_] + P - (HD - 1) / 2) * SP
                        rows_out.append((float(fs_[q_]), d, (VX[hh[j]] + evx_[j] + fevx_[q_]) * SP,
                                         (VY[hh[j]] + evy_[j] + fevy_[q_]) * SP, px, py, fi))
                else:
                  for j in o2_:
                    px = (xx[j] + ex_[j] + P - (WD - 1) / 2) * SP; py = (yy[j] + ey_[j] + P - (HD - 1) / 2) * SP
                    rows_out.append((float(s2_[j]), d, (VX[hh[j]] + evx_[j]) * SP, (VY[hh[j]] + evy_[j]) * SP, px, py, fi))
                s2sorted = np.sort(s2_)[::-1][:NPK]
                if probes:
                    kx_ = xx[o2_] + ex_[o2_]; ky_ = yy[o2_] + ey_[o2_]; ks_ = s2_[o2_]
                    kvx_ = VX[hh[o2_]] + evx_[o2_]; kvy_ = VY[hh[o2_]] + evy_[o2_]
            else:
              if len(sc) > NPK:
                  sc, xx, yy, hh = sc[:NPK], xx[:NPK], yy[:NPK], hh[:NPK]
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
                    if PROBE_OLD:
                        wv = Xp / rp
                        q = TD @ wv
                        ppx, ppy = q[0] / q[2], q[1] / q[2]
                        wn = math.sqrt(1 + ppx ** 2 + ppy ** 2)
                        Vperp = Vp - (Vp @ wv) * wv
                        pvx = (TD[0] @ Vperp) * wn / d / SP; pvy = (TD[1] @ Vperp) * wn / d / SP
                    else:
                        # gen 127: the gen126 probe used the body's HELIOCENTRIC direction, which is where the body sits in
                        # the d-frame only when d == r; at d != r it is off by ~|oref_perp| (1/d - 1/r) (2 px at 0.04 au,
                        # 3.7 px at 0.07 au here), which produced gen126's spurious "+-0.04 au window".  Now: the body's true
                        # d-frame position per co-add (topocentric ray from each frame's observer to radius d, exactly
                        # the inverse of the warp), linear least-squares fit over the co-adds -> position at tref, velocity.
                        pc_ = np.zeros((K, 2)); wc_ = np.zeros(K)
                        for k_, n in enumerate(sel):
                            lt_ = 0.0
                            for _ in range(2):
                                xn_ = orbit.propagate(s_[:3], s_[3:], np.array([t[n] - lt_ - pr["epoch_mjd"]]))[0]
                                lt_ = np.linalg.norm(xn_ - obs[n]) / C_AU
                            u_ = xn_ - obs[n]; u_ = u_ / np.linalg.norm(u_)
                            b_ = obs[n] @ u_; sr_ = -b_ + math.sqrt(b_ * b_ - obs[n] @ obs[n] + d * d)
                            q_ = TD @ ((obs[n] + sr_ * u_) / d)
                            g_ = int(gid_[k_]); pc_[g_] += [q_[0] / q_[2], q_[1] / q_[2]]; wc_[g_] += 1
                        pc_ /= np.maximum(wc_, 1)[:, None]
                        A_ = np.stack([np.ones(K), dts_c], 1)
                        cf_, *_ = np.linalg.lstsq(A_, pc_, rcond=None)
                        ppx, ppy = cf_[0]
                        pvx, pvy = cf_[1] / SP
                        res_ = pc_ - A_ @ cf_
                        lin_res = float(np.hypot(res_[:, 0], res_[:, 1]).max() / SP)
                    xi = int(round(ppx / SP + (WD - 1) / 2 - P)); yi = int(round(ppy / SP + (HD - 1) / 2 - P))
                    if not (5 <= xi < Wc - 5 and 5 <= yi < Hc - 5):
                        log(f"fstk probe {pr.get('tag', '?')}: off map"); continue
                    bestv = -1e9; bh_ = None
                    for ddx in (-1, 0, 1):
                        for ddy in (-1, 0, 1):
                            vx_ = (round(pvx / dv) + ddx) * dv; vy_ = (round(pvy / dv) + ddy) * dv
                            for ey in (-1, 0, 1):
                                for ex in (-1, 0, 1):
                                    a_ = 0.0
                                    for k_ in range(K):
                                        sxx = int(round(vx_ * dts_c[k_])); syy = int(round(vy_ * dts_c[k_]))
                                        a_ += float(Dq[k_, (P + syy + yi + ey) * WD + P + sxx + xi + ex])
                                    if a_ * qn > bestv:
                                        bestv = a_ * qn; bh_ = (xi + ex, yi + ey, vx_, vy_)
                    rsv = ""
                    if Dm3 is not None and bh_ is not None:
                        try:
                            r_ = _rescore(torch, dev, Dm3, np.array([bh_[0]], float), np.array([bh_[1]], float),
                                          np.array([bh_[2]]), np.array([bh_[3]]), dts_c, P, WD, HD, norm, dv, RSUB)
                            # also a continuous-truth evaluation (true sub-px position & velocity, no search)
                            r1_ = _rescore(torch, dev, Dm3, np.array([ppx / SP + (WD - 1) / 2 - P]), np.array([ppy / SP + (HD - 1) / 2 - P]),
                                           np.array([pvx]), np.array([pvy]), dts_c, P, WD, HD, norm, dv, 0)
                            if rres is not None:
                                rk2 = int((s2sorted > r_[0][0]).sum())
                                dd_ = np.hypot(kx_ - bh_[0], ky_ - bh_[1])
                                jn_ = int(np.argmin(dd_))
                                rsv = (f", nearest kept row {dd_[jn_]:.1f}px s {ks_[jn_]:.2f} dv ({(kvx_[jn_] - pvx) / dv:.2f},{(kvy_[jn_] - pvy) / dv:.2f})"
                                       f" krank {int((ks_ > ks_[jn_]).sum())}")
                                rsv = rsv + (f", RESCORE clip{RCLIP} exact {r_[5][0]:.2f} local {r_[0][0]:.2f} truth-cont {r1_[5][0]:.2f} "
                                       f"rank {rk2} slice-rescored max {s2sorted.max():.2f} 10th {s2sorted[min(9, len(s2sorted) - 1)]:.2f}")
                        except Exception as ex:
                            rsv = f", rescore probe failed {repr(ex)[:80]}"
                    bl = float(bs[yi - 2:yi + 3, xi - 2:xi + 3].max())
                    rk = int((sc > bl).sum()) if len(sc) else 0
                    if sc_full is not None:
                        rsv = f", full-list rank {int((sc_full > bl).sum())}/{sc_full.numel()}" + rsv
                    inprior = (VLO * vcs <= math.hypot(pvx, pvy) <= VHI * vcs) and \
                        abs(math.atan2(pvx * vcy - pvy * vcx, pvx * vcx + pvy * vcy)) <= ANG
                    pstat.append((pr.get("tag", "?"), d, bestv, bl, rk))
                    log(f"fstk probe {pr.get('tag', '?')}: d={d:.2f} (r {rp:.2f}, lin-res {lin_res if not PROBE_OLD else -1:.2f}px), |v|/vc {math.hypot(pvx, pvy) / vcs:.3f}, "
                        f"in prior {inprior}, exact-hyp local max {bestv:.2f}, best-map 5x5 max {bl:.2f}, "
                        f"rank among kept peaks {rk}, map max {float(sc.max()) if len(sc) else 0:.2f}{rsv}")
                except Exception as ex:
                    log("fstk probe failed:", repr(ex)[:120])
            log(f"fstk d={d:.2f}: {nh}/{nprior} hyps in {dtc:.1f}s ({nh / max(dtc, 1e-6):.0f} hyp/s, K {K}, "
                f"{'int8 q%.2f' % qs if use8 else 'fp16'}{tree_info}), warp {t_warp:.1f}s, max {float(sc.max()) if len(sc) else 0:.2f}, "
                f"peaks>5/6/7 {nab}, kept {len(sc)}; slice {di + 1}/{len(dg)} at {time.time() - T0:.0f}s of deadline {deadline:.0f}s; mx {mx} (old {mx_old})")
            sl_t.append((time.time() - t_d0, t_warp, dtc, nh)); done_d.append(d)
            del Dq, bs, pk
            Dm = None; Dm3 = None
        except Exception as ex:
            log(f"fstk failed at d={d:.1f}:", repr(ex)[:300])
            Dm = None; Dm3 = None
            try:
                if dev == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            nfail += 1
            if nfail >= 3:
                break
            continue
        nfail = 0
    smaps = smaps1 = None
    log(f"fstk done: {ntot} hypotheses, kernel {tconv:.1f}s ({ntot / max(tconv, 1e-6):.0f} hyp/s), {len(rows_out)} rows, "
        f"{time.time() - t_all0:.1f}s total")
    return rows_out, frames_out


def _split(S, tc):
    """gen 128: split a time-ordered co-add list in two contiguous parts minimising spreadA^2 + spreadB^2."""
    bst = None
    for m in range(1, len(S)):
        c = (tc[S[m - 1]] - tc[S[0]]) ** 2 + (tc[S[-1]] - tc[S[m]]) ** 2
        if bst is None or c < bst[0]:
            bst = (c, m)
    return S[:bst[1]], S[bst[1]:]


def _build(torch, Dq, OFFb, S, tc, L, cnt):
    """gen 128: partial sums of co-add subset S for the block's hypotheses OFFb (nb, K).  Returns (parts, inv):
    parts[u] = (tensor, base) with tensor[j - base] = sum_{k in S} Dq[k][j + OFF[h,k] - OFF[h,S0]] for every hypothesis
    h with inv[h] == u, valid for j in [min_h OFF[h,S0], max_h OFF[h,S0] + L)."""
    if len(S) == 1:
        return [(Dq[S[0]], 0)], np.zeros(len(OFFb), np.int64)
    A, B = _split(S, tc)
    pa, ia = _build(torch, Dq, OFFb, A, tc, L, cnt)
    pb, ib = _build(torch, Dq, OFFb, B, tc, L, cnt)
    o0 = OFFb[:, S[0]]; dl = OFFb[:, B[0]] - o0
    uq, inv = np.unique(np.stack([ia, ib, dl], 1), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    parts = []
    for u in range(len(uq)):
        a_, b_, d_ = (int(x) for x in uq[u])
        hs = inv == u
        lo = int(o0[hs].min()); n = int(o0[hs].max()) - lo + L
        ta, ba = pa[a_]; tb, bb = pb[b_]
        buf = torch.empty(n, dtype=Dq.dtype, device=Dq.device)
        torch.add(ta[lo - ba: lo - ba + n], tb[lo + d_ - bb: lo + d_ - bb + n], out=buf)
        cnt[0] += 1
        parts.append((buf, lo))
    return parts, inv


def _rescore(torch, dev, Dm3, xx, yy, vx, vy, dts_c, P, WD, HD, norm, dv, RSUB):
    """gen 109 (arm H): for each peak (kernel coords xx, yy; velocity vx, vy in px/day), the best stack over a local grid
    of position offsets (+-0.5 px) and velocity offsets (+-0.5 dv), steps 0.5/RSUB (RSUB 0: the exact point only); the
    track is continuous and rounded per co-add.  Returns (best, ex, ey, evx, evy, exact) as numpy arrays."""
    K = Dm3.shape[0]
    n = len(xx)
    if RSUB > 0:
        g = np.arange(-RSUB, RSUB + 1) * (0.5 / RSUB)
    else:
        g = np.zeros(1)
    EX, EY, EVX, EVY = [a.ravel() for a in np.meshgrid(g, g, g * dv, g * dv, indexing="ij")]
    G = len(EX)
    i0 = int(np.argmin(np.abs(EX) + np.abs(EY) + np.abs(EVX) + np.abs(EVY)))
    flat = Dm3.reshape(K, -1)
    NT = flat.shape[1]
    xx_t = torch.tensor(np.asarray(xx, np.float32), device=dev)[:, None]
    yy_t = torch.tensor(np.asarray(yy, np.float32), device=dev)[:, None]
    vx_t = torch.tensor(np.asarray(vx, np.float32), device=dev)[:, None]
    vy_t = torch.tensor(np.asarray(vy, np.float32), device=dev)[:, None]
    EX_t = torch.tensor(EX.astype(np.float32), device=dev)[None]; EY_t = torch.tensor(EY.astype(np.float32), device=dev)[None]
    EVX_t = torch.tensor(EVX.astype(np.float32), device=dev)[None]; EVY_t = torch.tensor(EVY.astype(np.float32), device=dev)[None]
    best = np.full(n, -1e9); arg = np.zeros(n, np.int64); exact = np.zeros(n)
    CH = max(1, int(2_000_000 // max(G, 1)))
    for c0 in range(0, n, CH):
        sl = slice(c0, min(n, c0 + CH))
        acc = torch.zeros((sl.stop - sl.start, G), device=dev)
        for k in range(K):
            dt = float(dts_c[k])
            col = torch.floor(xx_t[sl] + EX_t + P + (vx_t[sl] + EVX_t) * dt + 0.5)
            row = torch.floor(yy_t[sl] + EY_t + P + (vy_t[sl] + EVY_t) * dt + 0.5)
            col = col.clamp(0, WD - 1); row = row.clamp(0, HD - 1)
            # gen 129: int64 before the multiply (float32 row*WD+col is inexact once HD*WD > 2^24, ~4.1k x 4.1k canvas)
            idx = (row.to(torch.int64) * WD + col.to(torch.int64)).clamp_(0, NT - 1)
            acc += flat[k][idx].float()
        acc *= norm
        m_, a_ = acc.max(1)
        best[sl] = m_.cpu().numpy(); arg[sl] = a_.cpu().numpy(); exact[sl] = acc[:, i0].cpu().numpy()
        del acc
    return best, EX[arg], EY[arg], EVX[arg], EVY[arg], exact


def _nat_med(torch, maps, clip):
    """gen 139: per-frame median of the clipped 1" S/N map over covered (non-zero) pixels (subsampled)."""
    mu = np.zeros(maps.shape[0])
    for n in range(maps.shape[0]):
        try:
            v = maps[n, ::7, ::7].float()
            v = v[v != 0]
            if v.numel() > 100:
                mu[n] = float(torch.median(v.clamp(-clip, clip)))
        except Exception:
            mu[n] = 0.0
    return mu


def _natscore(torch, dev, maps, obs, dts, wsel, DW, d, TD, TF0, x0, y0, PIX, SP, WD, HD, P, xx, yy, vx, vy, dv, clip, mu):
    """gen 139 (TNO_FSTK_NAT): native-resolution re-score of fstk peaks.  xx, yy: kernel coords (canvas col/row - P,
    sub-cell), vx, vy: velocity in canvas cells/day, at time tref (dts = frame time - tref).  Each frame n is sampled on the
    unpooled 1" map at the track's exact position (ray from the frame's observer to radius d, as the warp does; linearised
    per peak and frame around the track point), nearest 1" pixel, clipped at +-clip, minus the frame median, depth-weighted
    as the fstk co-adds.  Two local grids: (1) position +-PR cells in PS steps x velocity +-VR dv in VS steps (the coarse
    and clip-3 velocity is often ~1 dv off, which the 2" pooled statistic cannot see but the 1" one can); (2) around the best,
    position +-0.25 cell x velocity +-0.125 dv, 3 steps each.  Returns (best score, ex, ey, evx, evy) numpy arrays."""
    PR = float(os.environ.get("TNO_FSTK_NPR", "1.0")); PS = float(os.environ.get("TNO_FSTK_NPS", "0.5"))
    VR = float(os.environ.get("TNO_FSTK_NVR", "1.0")); VS = float(os.environ.get("TNO_FSTK_NVS", "0.25"))
    gp = np.arange(-round(PR / PS), round(PR / PS) + 1) * PS
    gv = np.arange(-round(VR / VS), round(VR / VS) + 1) * VS * dv
    xx = np.asarray(xx, float); yy = np.asarray(yy, float); vx = np.asarray(vx, float); vy = np.asarray(vy, float)
    b1, ex1, ey1, evx1, evy1 = _natgrid(torch, dev, maps, obs, dts, wsel, DW, d, TD, TF0, x0, y0, PIX, SP, WD, HD, P,
                                        xx, yy, vx, vy, clip, mu, gp, gv)
    g2p = np.array([-0.25, 0.0, 0.25]) * (PS / 0.5); g2v = np.array([-0.125, 0.0, 0.125]) * (VS / 0.25) * dv
    b2, ex2, ey2, evx2, evy2 = _natgrid(torch, dev, maps, obs, dts, wsel, DW, d, TD, TF0, x0, y0, PIX, SP, WD, HD, P,
                                        xx + ex1, yy + ey1, vx + evx1, vy + evy1, clip, mu, g2p, g2v)
    return b2, ex1 + ex2, ey1 + ey2, evx1 + evx2, evy1 + evy2


def _natgrid(torch, dev, maps, obs, dts, wsel, DW, d, TD, TF0, x0, y0, PIX, SP, WD, HD, P, xx, yy, vx, vy, clip, mu, gp, gv):
    """gen 139: one local-grid pass of _natscore (position offsets gp cells, velocity offsets gv cells/day)."""
    N, H, W = maps.shape
    n = len(xx)
    EX, EY, EVX, EVY = [a.ravel() for a in np.meshgrid(gp, gp, gv, gv, indexing="ij")]
    G = len(EX)
    f32 = torch.float32
    w = np.asarray(wsel, float) if DW else np.ones(N)
    norm = 1.0 / math.sqrt(max(float((w ** 2).sum()), 1e-9))
    gx0 = (xx + P - (WD - 1) / 2) * SP
    gy0 = (yy + P - (HD - 1) / 2) * SP
    TDt = torch.tensor(np.asarray(TD, float), dtype=torch.float64)
    TFt = torch.tensor(np.asarray(TF0, float), dtype=torch.float64)
    EX_t = torch.tensor(EX, dtype=f32, device=dev)[None]; EY_t = torch.tensor(EY, dtype=f32, device=dev)[None]
    EVX_t = torch.tensor(EVX, dtype=f32, device=dev)[None]; EVY_t = torch.tensor(EVY, dtype=f32, device=dev)[None]
    best = np.full(n, -1e9); arg = np.zeros(n, np.int64)
    CH = max(1, int(2_000_000 // G))
    flat = maps.reshape(N, -1)
    for c0 in range(0, n, CH):
        sl = slice(c0, min(n, c0 + CH)); m = sl.stop - sl.start
        acc = torch.zeros((m, G), dtype=f32, device=dev)
        for k in range(N):
            if w[k] <= 0:
                continue
            dt = float(dts[k])
            gx = gx0[sl] + vx[sl] * SP * dt; gy = gy0[sl] + vy[sl] * SP * dt
            # (3m,) points: track centre, +1 cell in x, +1 cell in y (float64 on the CPU; m is small)
            GX = torch.tensor(np.r_[gx, gx + SP, gx], dtype=torch.float64)
            GY = torch.tensor(np.r_[gy, gy, gy + SP], dtype=torch.float64)
            Wd = TDt[2][None] + GX[:, None] * TDt[0][None] + GY[:, None] * TDt[1][None]
            Wd = Wd / torch.linalg.norm(Wd, dim=1, keepdim=True)
            X = d * Wd - torch.tensor(np.asarray(obs[k], float), dtype=torch.float64)[None]
            q = X @ TFt.T
            u = ((q[:, 0] / q[:, 2]) - x0) / PIX; v = ((q[:, 1] / q[:, 2]) - y0) / PIX
            u0, ux, uy = u[:m], u[m:2 * m] - u[:m], u[2 * m:] - u[:m]
            v0, vx_, vy_ = v[:m], v[m:2 * m] - v[:m], v[2 * m:] - v[:m]
            u0, ux, uy, v0, vx_, vy_ = [a_.to(f32).to(dev)[:, None] for a_ in (u0, ux, uy, v0, vx_, vy_)]
            dX = EX_t + EVX_t * dt; dY = EY_t + EVY_t * dt          # (1, G) offsets in cells
            cu = torch.floor(u0 + ux * dX + uy * dY + 0.5)
            cv = torch.floor(v0 + vx_ * dX + vy_ * dY + 0.5)
            ins = (cu >= 0) & (cu <= W - 1) & (cv >= 0) & (cv <= H - 1)
            idx = (cv.clamp(0, H - 1).to(torch.int64) * W + cu.clamp(0, W - 1).to(torch.int64))
            val = flat[k][idx].float()
            val = torch.where(ins & (val != 0), val.clamp(-clip, clip) - float(mu[k]), torch.zeros_like(val))
            acc.add_(val, alpha=float(w[k]))
            del cu, cv, ins, idx, val, dX, dY
        acc *= norm
        mx_, am_ = acc.max(1)
        best[sl] = mx_.cpu().numpy(); arg[sl] = am_.cpu().numpy()
        del acc
    return best, EX[arg], EY[arg], EVX[arg], EVY[arg]
