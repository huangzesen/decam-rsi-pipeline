"""Static-source removal, tracklet-less linking in the BK frame, orbit fit, scoring."""
import math, os, time
import numpy as np
from scipy.spatial import cKDTree
from orbit import (GM, C_AU, ARC, T0, log, left, radec_to_unit, tangent_frame, predict_dirs, fit_orbit)

OFF = 1 << 15


def keys_of(a, ad, half, cell, shift):
    q = np.floor(np.concatenate([a, ad * half], 1) / cell + shift).astype(np.int64) + OFF
    q = np.clip(q, 0, 2 * OFF - 1)
    return (q[:, 0] << 48) | (q[:, 1] << 32) | (q[:, 2] << 16) | q[:, 3]


def build(rows, results):
    ne = len(rows)
    et = np.array([float(r["mjd_mid"]) for r in rows])
    eobs = np.array([[float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])] for r in rows])
    enight = np.floor(et - 0.5).astype(int)
    emz = np.array([float(r["magzero"]) if r.get("magzero") not in (None, "") else np.nan for r in rows])
    eband = [r.get("band", "") for r in rows]
    D, E = [], []
    for ei, o in results.items():
        d = o["det"]
        if len(d):
            D.append(d); E.append(np.full(len(d), ei))
    D = np.concatenate(D); E = np.concatenate(E)
    return dict(et=et, eobs=eobs, enight=enight, emz=emz, eband=eband, D=D, E=E, done=np.array(sorted(results)))


def static_mask(u, night, sn):
    """True for stationary detections: a detection within 0.7" on another night, or within 1.5" on >= 2 other
    nights.  Per-night k=1 queries (linear in the number of exposures, unlike all-pairs)."""
    st = np.zeros(len(u), bool)
    un = np.unique(night)
    if len(un) < 2:
        return st
    idx = [np.flatnonzero(night == n_) for n_ in un]
    trees = [cKDTree(u[ix]) for ix in idx]
    cnt = np.zeros(len(u), np.int16)
    K = 10         # compare with at most K other nights (spread over the run) to stay linear in big scenarios
    for a_, ia in enumerate(idx):
        others = [b_ for b_ in range(len(trees)) if b_ != a_]
        if len(others) > K:
            others = [others[int(round(x))] for x in np.linspace(0, len(others) - 1, K)]
        for b_ in others:
            tb = trees[b_]
            d, _ = tb.query(u[ia], k=1, distance_upper_bound=1.5 * ARC, workers=int(os.environ.get("TNO_NCPU", "4")))
            hit = np.isfinite(d)
            cnt[ia[hit]] += 1
            st[ia[d < 0.7 * ARC]] = True
    st |= cnt >= 2
    return st


def observer_velocity(et, eobs):
    """Observer heliocentric velocity (au/day) at each exposure from finite differences over the run."""
    o = np.argsort(et)
    t, x = et[o], eobs[o]
    keep = np.r_[True, np.diff(t) > 1e-4]
    t, x = t[keep], x[keep]
    v = np.zeros((len(et), 3))
    if len(t) < 2:
        return v
    g = np.stack([np.gradient(x[:, k], t) for k in range(3)], 1)
    for k in range(3):
        v[:, k] = np.interp(et, t, g[:, k])
    return v


def motion_ok(ui, uj, ti, tj, vo):
    """Same-night pair with a gap >= 0.5 h: its motion must look like a body beyond ~25 au (rate within the range set
    by the observer's reflex motion, direction within the reflex direction +- what a bound orbit can add)."""
    dt = tj - ti
    ok = np.ones(len(ui), bool)
    long_ = np.abs(dt) >= 0.5 / 24
    if not long_.any():
        return ok
    ui, uj, dt, vo = ui[long_], uj[long_], dt[long_], vo[long_]
    vp = vo - np.sum(vo * ui, 1)[:, None] * ui            # observer velocity perpendicular to the line of sight
    vpn = np.linalg.norm(vp, axis=1)
    du = (uj - ui) / dt[:, None]
    du = du - np.sum(du * ui, 1)[:, None] * ui              # sky motion, rad/day
    rate = np.linalg.norm(du, axis=1)
    noise = 0.5 * ARC / np.abs(dt)                          # rate error from centroids
    lo = np.maximum(vpn / 150.0 * 0.3 - noise, 0)
    hi = vpn / 25.0 * 1.6 + math.sqrt(2 * GM / 25.0) / 25.0 + noise
    good_ = (rate >= lo) & (rate <= hi)
    # direction: opposite to the observer's perpendicular velocity, when the reflex dominates bound orbital motion
    vmax_own = math.sqrt(2 * GM / 25.0)
    strong = vpn > 1.5 * vmax_own
    cosang = -np.sum(du * vp, 1) / np.maximum(rate * vpn, 1e-30)
    tol = np.arcsin(np.clip(vmax_own / np.maximum(vpn, 1e-9), 0, 1)) + np.radians(15) + np.arctan(noise / np.maximum(rate, 1e-30))
    dir_ok = (~strong) | (cosang >= np.cos(np.minimum(tol, np.pi)))
    ok[np.flatnonzero(long_)] = good_ & dir_ok
    return ok


def run(args, rows, meta, results, cands, write, budget):
    B = build(rows, results)
    D, E = B["D"], B["E"]
    et, eobs, enight = B["et"], B["eobs"], B["enight"]
    vobs = observer_velocity(et, eobs)
    ra, de = np.radians(D[:, 0]), np.radians(D[:, 1])
    u = radec_to_unit(ra, de)
    night = enight[E]
    sn, sh = D[:, 2], D[:, 3]
    st = static_mask(u, night, sn)
    good = (~st) & (sh > 0.4) & (sh < 1.25 + 3.0 / np.sqrt(np.maximum(sn, 1))) & (sn < 1500)
    # halos / spikes / ghosts around bright stationary sources
    bright = np.flatnonzero(st & (sn > 80))
    if len(bright):
        rr = np.clip(2.5 * np.sqrt(sn[bright] / 80.0), 2.5, 45.0) * ARC
        gidx = np.flatnonzero(good)
        tr = cKDTree(u[gidx])
        for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 32), (32, 46)):
            sel_ = (rr >= lo * ARC) & (rr < hi * ARC)
            if not sel_.any():
                continue
            tb = cKDTree(u[bright[sel_]])
            sdm = tb.sparse_distance_matrix(tr, hi * ARC, output_type="ndarray")
            if len(sdm):
                kill = sdm["v"] < rr[sel_][sdm["i"]]
                good[gidx[sdm["j"][kill]]] = False
    log(f"{len(D)} detections, {st.sum()} stationary, {good.sum()} usable")
    # usable catalogue
    # streaks / clumps: >= 2 other usable S/N >= 4 detections of the same exposure within 10"
    g4 = np.flatnonzero(good & (sn >= 4))
    if len(g4):
        t4 = cKDTree(u[g4])
        P = t4.query_pairs(10 * ARC, output_type="ndarray")
        if len(P):
            se = E[g4[P[:, 0]]] == E[g4[P[:, 1]]]
            nbe = np.zeros(len(g4), int)
            np.add.at(nbe, P[se, 0], 1); np.add.at(nbe, P[se, 1], 1)
            good[g4[nbe >= 2]] = False
    gi = np.flatnonzero(good)
    log(f"{len(gi)} usable after clump filter")
    nights_avail = np.unique(enight[B["done"]])
    if len(nights_avail) < 3:
        log("fewer than 3 nights")
        return
    # frame
    if "ra" in meta:
        ra0, de0 = math.radians(meta["ra"]), math.radians(meta["dec"])
    else:
        ra0 = math.atan2(np.mean(u[:, 1]), np.mean(u[:, 0])); de0 = math.asin(np.mean(u[:, 2]))
    F = tangent_frame(ra0, de0)
    obs_f = eobs @ F.T
    done = B["done"]
    t_ref = 0.5 * (et[done].min() + et[done].max())
    half = max(0.5 * (et[done].max() - et[done].min()), 1.0)
    ug = u[gi] @ F.T
    core = float(np.percentile(np.hypot(ug[:, 0] / ug[:, 2], ug[:, 1] / ug[:, 2]), 99.5)) + 0.02
    # gamma grid: Earth path curvature over the window sets the step
    tt = np.linspace(-half, half, 41)
    ob = np.array([np.interp(tt, et[done] - t_ref, obs_f[done, k]) for k in range(3)]).T
    A = np.c_[np.ones_like(tt), tt]
    resid = ob[:, :2] - A @ np.linalg.lstsq(A, ob[:, :2], rcond=None)[0]
    curv = max(float(np.abs(resid).max()), 1e-4)
    tol_ph = 2.5 * ARC
    dg = min(max(tol_ph / curv, 5e-5), 2e-3)
    gmin, gmax = 1 / 150.0, 1 / 25.0
    grid = np.arange(gmin, gmax + dg / 2, dg)
    # coarse-to-fine order
    order = []
    for stride in (8, 4, 2, 1):
        for k in range(0, len(grid), stride):
            if k not in order:
                order.append(k)
    log(f"curvature {curv:.4f} au, dgamma {dg:.2e}, {len(grid)} gammas, core {math.degrees(core):.2f} deg")

    # --------------------------------------------------------------------- orbit fit + extension + score
    uall = u
    gtree = cKDTree(uall[gi])
    etrees = [cKDTree(uall[E == e]) for e in done]
    ext_r = 1.2 * ARC
    results_ = []
    # process linkages: best (more nights, more S/N) first
    seen_sets = []
    owner = {}
    seen_mem = set()
    from forced import Forced
    FP = Forced(results)
    log(f'forced maps for {len(FP.maps)} exposures')
    REASON = {}
    def finish(members, st2):
        """Final fit, sanity, coverage/miss counts, magnitudes -> candidate dict (None if implausible)."""
        st2, res2 = fit_orbit(st2, t_ref, et[E[members]], eobs[E[members]], uall[members])
        r_h = float(np.linalg.norm(st2[:3]))
        if not (20 < r_h < 300):
            return None
        vv = float(np.linalg.norm(st2[3:]))
        if vv > 1.5 * math.sqrt(2 * GM / r_h):
            return None
        # coverage: exposures where the prediction lands on data (>= 2 detections of any kind within 15")
        pd = predict_dirs(st2, t_ref, et[done], eobs[done])
        memexp = set(E[members].tolist())
        ncov, nblend, misses = 0, 0, []
        for k, e in enumerate(done):
            if e in memexp:
                ncov += 1
                continue
            if etrees[k].query_ball_point(pd[k], 30 * ARC, return_length=True) < 4:
                continue
            if etrees[k].query_ball_point(pd[k], 1.5 * ARC, return_length=True) > 0:
                nblend += 1           # something (star / flagged source) sits there: cannot tell
                continue
            ncov += 1
            misses.append(e)
        ncov = max(ncov, 1)
        snm = sn[members]
        score_sum = float(np.sum(np.minimum(snm, 50.0))) / math.sqrt(ncov)
        nn = len(np.unique(night[members]))
        nfrac = len(members) / ncov
        mags = []
        for x in members:
            mz = B["emz"][E[x]]
            if np.isfinite(mz) and D[x, 4] > 0:
                mags.append(mz - 2.5 * math.log10(D[x, 4]))
        mag = float(np.median(mags)) if mags else None
        mstd = float(np.std(mags)) if len(mags) >= 2 else 0.0
        cand = dict(state=st2, members=members, nn=nn, score=score_sum, rms=float(np.sqrt(np.mean(res2 ** 2))),
                    nmiss=len(misses), nblend=nblend, ncov=ncov, medsn=float(np.median(snm)), mstd=mstd,
                    nfrac=nfrac, r=r_h, mag=mag, nseed_n=0, maxsn=float(snm.max()),
                    sumsn=float(snm.sum()))
        cand.update(pd=pd, memexp=memexp)
        return cand

    fitted = set()
    # gen 2: within-night catalogue shift-and-stack (hough.py) -> extra seeds for faint multi-exposure tracklets
    hseeds = {}
    try:
        import hough
        th0 = time.time()
        hbud = 0.08 * budget
        nexp_n = {nn: len(np.unique(E[gi[night[gi] == nn]])) for nn in nights_avail}
        for nn in sorted(nights_avail, key=lambda q: -nexp_n[q]):
            if nexp_n[nn] < 3:
                continue
            left_t = hbud - (time.time() - th0)
            if left_t < 1.0:
                break
            m = gi[night[gi] == nn]
            em = E[m]
            ue = np.unique(em)
            if len(ue) > 64:
                ue = ue[np.argsort(et[ue])][np.linspace(0, len(ue) - 1, 64).astype(int)]
                m = m[np.isin(em, ue)]; em = E[m]
            lidx = np.searchsorted(ue, em)
            tm = 0.5 * (et[ue].min() + et[ue].max())
            uf = u[m] @ F.T
            xx = uf[:, 0] / uf[:, 2] * 206264.806
            yy = uf[:, 1] / uf[:, 2] * 206264.806
            vof = F @ vobs[ue].mean(0)
            hi_, hs_ = hough.night_tracklets(xx, yy, (et[em] - tm) * 24.0, lidx, sn[m], vof, tlimit=left_t)
            hseeds[nn] = m[hi_]
            log(f"hough night {nn}: {len(ue)} exp, {len(m)} pts -> {len(hi_)} tracklet seeds "
                f"(top S/N {hs_[:3].round(1).tolist()}) {time.time() - th0:.1f}s")
    except Exception as ex:
        log("hough failed", ex)
    PASSES = [(3.5, 5.5, 5.0), (3.0, 5.0, 4.6)]
    for pass_i, (PE, PC, S1) in enumerate(PASSES):
        if pass_i > 0 and time.time() - T0 > 0.45 * budget:
            break
        log(f"pass {pass_i}: pair S/N >= {PE}, combined >= {PC}, singles >= {S1}")
        # seeds.  Nights with >= 2 exposures: a detection needs a same-night partner in another exposure (a body moves
        # < ~5"/h); one representative (highest S/N) per partner group.  Single-exposure nights: S/N >= 5.5.
        seeds = []
        for nn in nights_avail:
            m = gi[night[gi] == nn]
            if len(m) == 0:
                continue
            nexp = len(np.unique(E[m]))
            if nexp >= 2:
                # pair exposures of the night that are close in time (each with its 3 nearest in time, <= 1.5 h)
                m4 = m[sn[m] >= PE]
                ex = np.unique(E[m4])
                ex = ex[np.argsort(et[ex])]
                ix = {e: m4[E[m4] == e] for e in ex}
                tr = {e: cKDTree(u[ix[e]]) for e in ex}
                I_, J_ = [], []
                paired = set()
                for a_ in range(len(ex)):
                    for b_ in range(a_ + 1, min(a_ + 4, len(ex))):
                        dth = abs(et[ex[b_]] - et[ex[a_]]) * 24
                        if dth > 4.0:
                            break
                        paired.add(ex[a_]); paired.add(ex[b_])
                        sdm = tr[ex[a_]].sparse_distance_matrix(tr[ex[b_]], (1.2 + 6.0 * dth) * ARC, output_type="ndarray")
                        if len(sdm):
                            I_.append(ix[ex[a_]][sdm["i"]]); J_.append(ix[ex[b_]][sdm["j"]])
                # exposures without a partner exposure in the pairing window are treated as single-exposure nights
                singles = m[(sn[m] >= S1) & ~np.isin(E[m], list(paired))]
                if not I_:
                    sel = singles
                    I_, J_ = [np.zeros(0, int)], [np.zeros(0, int)]
                i, j = np.concatenate(I_), np.concatenate(J_)
                ok = np.hypot(sn[i], sn[j]) >= PC
                i, j = i[ok], j[ok]
                if len(i):
                    ok = motion_ok(u[i], u[j], et[E[i]], et[E[j]], vobs[E[i]])
                    i, j = i[ok], j[ok]
                if len(i) == 0:
                    sel = singles
                else:
                    # every detection with a valid partner is a seed; for near-simultaneous pairs (same object
                    # almost certainly, < 2" apart) keep only the brighter.  (Connected components chained
                    # through random partners when the time gap, hence the radius, is large.)
                    nodes = np.unique(np.r_[i, j])
                    dd = np.linalg.norm(u[i] - u[j], axis=1)
                    cl = dd < 2.0 * ARC
                    lower = np.where(sn[i[cl]] >= sn[j[cl]], j[cl], i[cl])
                    sel = np.setdiff1d(nodes, lower)
                    cap = 4000
                    if len(sel) < cap:
                        extra = np.setdiff1d(singles, nodes)
                        extra = extra[np.argsort(-sn[extra])[:cap - len(sel)]]
                        sel = np.r_[sel, extra]
            else:
                sel = m[sn[m] >= S1]
            cap = 4000
            if len(sel) > cap:
                sel = sel[np.argsort(-sn[sel])[:cap]]
            if nn in hseeds and len(hseeds[nn]):
                sel = np.union1d(sel, hseeds[nn][:1500 if pass_i == 0 else 3000])
            seeds.append(sel)
        seeds = np.concatenate(seeds) if seeds else np.zeros(0, int)
        inj = np.flatnonzero(D[:, 5] == 99)
        if len(inj):
            log(f"DEBUG injected {len(inj)}: static {st[inj].sum()} usable {good[inj].sum()} seeds {np.isin(inj, seeds).sum()}")
        if os.environ.get("TNO_DEBUG"):
            np.save(os.environ["TNO_DEBUG"] + f".seeds{pass_i}.npy", seeds)
            np.save(os.environ["TNO_DEBUG"] + ".gi.npy", gi); np.save(os.environ["TNO_DEBUG"] + ".st.npy", st)
        log(f"{len(seeds)} seeds; per night {[int((night[seeds] == nn).sum()) for nn in nights_avail]}")
        if len(np.unique(night[seeds])) < 3:
            break

        us = u[seeds] @ F.T
        theta = us[:, :2] / us[:, 2:3]
        ts = et[E[seeds]] - t_ref
        obs_f = eobs @ F.T
        os_ = obs_f[E[seeds]]
        ns = night[seeds]

        cell = 4.0 * ARC
        un = np.unique(ns)
        links = {}
        link_deadline = budget * 0.80
        tg0 = time.time()
        for kk, gidx in enumerate(order):
            if time.time() - T0 > link_deadline:
                log(f"linking stopped at {kk}/{len(order)} gammas")
                break
            g = grid[gidx]
            tp = ts - 1.0 / (C_AU * g)
            k_ = 0.5 * GM * g ** 3
            f = 1.0 - k_ * tp * tp - g * os_[:, 2]
            phi = theta * f[:, None] + g * os_[:, :2]
            uu = 1.0 - k_ * tp * tp
            npara = math.sqrt(2 * GM * g ** 3)
            idx_n = [np.flatnonzero(ns == n_) for n_ in un]
            trees = [cKDTree(phi[ix]) for ix in idx_n]
            K0, K1, I, J = [], [], [], []
            for p in range(len(un)):
                for q in range(p + 1, len(un)):
                    dt = un[q] - un[p]
                    if dt < 0.5:
                        continue
                    sd = trees[p].sparse_distance_matrix(trees[q], npara * (dt + 0.5) + 2 * ARC, output_type="ndarray")
                    if len(sd) == 0:
                        continue
                    i = idx_n[p][sd["i"]]; j = idx_n[q][sd["j"]]
                    Dd = uu[i] * tp[j] - uu[j] * tp[i]
                    al = (phi[i] * tp[j, None] - phi[j] * tp[i, None]) / Dd[:, None]
                    ad = (uu[i, None] * phi[j] - uu[j, None] * phi[i]) / Dd[:, None]
                    ok = (np.hypot(al[:, 0], al[:, 1]) <= core) & (np.hypot(ad[:, 0], ad[:, 1]) <= npara)
                    i, j, al, ad = i[ok], j[ok], al[ok], ad[ok]
                    K0.append(keys_of(al, ad, half, cell, 0.0)); K1.append(keys_of(al, ad, half, cell, 0.5))
                    I.append(i.astype(np.int32)); J.append(j.astype(np.int32))
            if not I:
                continue
            I = np.concatenate(I); J = np.concatenate(J)
            npairs = len(I)
            nl0 = len(links)
            for K in (K0, K1):
                k = np.concatenate(K)
                o = np.argsort(k, kind="stable")
                ks = k[o]
                brk = np.flatnonzero(np.diff(ks)) + 1
                st_ = np.r_[0, brk]; en_ = np.r_[brk, len(ks)]
                big = np.flatnonzero(en_ - st_ >= 2)
                if len(un) <= 62 and len(big):
                    # cheap pre-filter: distinct nights among the cell's pair endpoints via OR-ed bit masks
                    nidx = np.searchsorted(un, ns)
                    pm = (np.left_shift(np.int64(1), nidx[I]) | np.left_shift(np.int64(1), nidx[J]))[o]
                    cm = np.bitwise_or.reduceat(pm, st_[big])
                    pc = np.zeros(len(cm), np.int64)
                    x = cm.copy()
                    while np.any(x):
                        pc += x & 1
                        x >>= 1
                    big = big[pc >= 3]
                for b in big:
                    m = o[st_[b]:en_[b]]
                    c = np.unique(np.r_[I[m], J[m]])
                    if len(np.unique(ns[c])) < 3:
                        continue
                    # line fit with clipping
                    for _ in range(4):
                        Am = np.stack([uu[c], tp[c]], 1)
                        sol, *_ = np.linalg.lstsq(Am, phi[c], rcond=None)
                        r = np.hypot(*(phi[c] - Am @ sol).T)
                        if r.max() <= 1.5 * ARC:
                            break
                        keep = r < max(1.5 * ARC, np.sort(r)[len(r) // 2] * 2.5)
                        if keep.all():
                            keep = r < r.max()
                        c = c[keep]
                        if len(c) < 3:
                            break
                    if len(c) < 3 or len(np.unique(ns[c])) < 3 or r.max() > 1.5 * ARC:
                        continue
                    # one detection per exposure
                    key = tuple(sorted(seeds[c].tolist()))
                    rms = float(np.sqrt(np.mean(r ** 2)) / ARC)
                    if key not in links or links[key][0] > rms:
                        links[key] = (rms, g, sol.copy())
            if kk % 5 == 0 or kk < 4:
                log(f"gamma {gidx} (R {1 / g:.1f}): {npairs} pairs, {len(links) - nl0} new, {len(links)} linkages")
        log(f"linking done: {len(links)} linkages in {time.time() - tg0:.0f}s")
        if len(inj):
            injs = set(inj.tolist())
            nl = [k for k in links if sum(1 for x in k if x in injs) >= 3]
            log(f"DEBUG linkages with >=3 injected seeds: {len(nl)}; pure {sum(1 for k in nl if all(x in injs for x in k))}")

        items = sorted(links.items(), key=lambda kv: (-len(np.unique(night[list(kv[0])])), -len(kv[0]), kv[1][0]))
        log(f"fitting {len(items)} linkages")
        for n_it, (key, (rms, g, sol)) in enumerate(items):
            if time.time() - T0 > budget * 0.93:
                log("fit stage stopped at", n_it)
                break
            c = np.array(key, int)
            if key in fitted:
                continue
            fitted.add(key)
            pure = len(inj) > 0 and all(x in inj for x in c)
            # skip if mostly owned by an earlier accepted candidate
            ov = sum(1 for x in c if x in owner)
            if ov >= len(c):
                if pure:
                    REASON["owned"] = REASON.get("owned", 0) + 1
                continue
            al, ad = sol[0], sol[1]
            rf = np.array([al[0], al[1], 1.0]) / g
            vf = np.array([ad[0], ad[1], 0.0]) / g
            state0 = np.r_[F.T @ rf, F.T @ vf]
            try:
                st_, res = fit_orbit(state0, t_ref, et[E[c]], eobs[E[c]], uall[c])
            except Exception:
                continue
            if not np.all(np.isfinite(st_)) or res.max() > 1.5:
                if len(inj) and all(x in inj for x in c):
                    log("DEBUG pure injected linkage failed initial fit", res.round(2), len(c), 1 / g)
                continue
            # extend to every processed exposure, tightening the radius as the orbit improves
            members = c.copy()
            st2 = st_
            res2 = res
            okfit = True
            for rad_ in (4.0, 2.0, 1.2):
                pd = predict_dirs(st2, t_ref, et[done], eobs[done])
                hits = gtree.query_ball_point(pd, rad_ * ARC)
                add = []
                for k, e in enumerate(done):
                    h = [gi[x] for x in hits[k] if E[gi[x]] == e]
                    if h:
                        dd = [np.linalg.norm(uall[x] - pd[k]) for x in h]
                        add.append(h[int(np.argmin(dd))])
                mem = np.array(sorted(set(add) | set(members.tolist())), int)
                s_try, r_try = fit_orbit(st2, t_ref, et[E[mem]], eobs[E[mem]], uall[mem])
                # clip worst outliers one at a time
                while len(mem) > 3 and r_try.max() > 1.0:
                    w = np.argmax(r_try)
                    mem = np.delete(mem, w)
                    s_try, r_try = fit_orbit(s_try, t_ref, et[E[mem]], eobs[E[mem]], uall[mem])
                if r_try.max() > 1.0 or not np.all(np.isfinite(s_try)):
                    if rad_ == 4.0:
                        res2 = res
                        if res.max() > 1.0:
                            okfit = False
                    continue
                members, st2, res2 = mem, s_try, r_try
                # one detection per exposure: the closest to the fit
                o_ = np.lexsort((res2, E[members]))
                members, res2 = members[o_], res2[o_]
                _, first = np.unique(E[members], return_index=True)
                members, res2 = members[first], res2[first]
            if not okfit:
                REASON['okfit'] = REASON.get('okfit', 0) + pure
                continue
            if len(members) < 3:
                continue
            if len(np.unique(night[members])) < 3:
                REASON['nights'] = REASON.get('nights', 0) + pure
                continue
            cand = finish(members, st2)
            if cand is None:
                continue
            mkey = tuple(members.tolist())
            if mkey in seen_mem:
                continue
            seen_mem.add(mkey)
            results_.append(cand)
            for x in members:
                owner[x] = len(results_) - 1
            if len(results_) % 300 == 0:
                add_forced(results_, FP, done, enight)
                publish(select(results_), cands, t_ref, write)
    add_forced(results_, FP, done, enight)
    results_ = select(results_)
    publish(results_, cands, t_ref, write)
    # refinement: a short-arc orbit can miss a distant night by several arcsec; for the best candidates try every
    # detection within 12" of the prediction on exposures without a member, keep additions that fit to < 1"
    nref = 0
    for cnd in sorted(results_, key=lambda c: -c["pp"])[:2000]:
        if time.time() - T0 > budget * 0.95:
            break
        members, st2 = cnd["members"].copy(), cnd["state"].copy()
        changed = False
        for _ in range(2):
            pd = predict_dirs(st2, t_ref, et[done], eobs[done])
            memexp = set(E[members].tolist())
            order_k = sorted([k for k, e in enumerate(done) if e not in memexp],
                             key=lambda k: -np.min(np.abs(et[done[k]] - et[E[members]])))
            for k in order_k:
                e = done[k]
                if e in memexp:
                    continue
                h = [gi[x] for x in gtree.query_ball_point(pd[k], 12 * ARC) if E[gi[x]] == e]
                h = sorted(h, key=lambda x: np.linalg.norm(uall[x] - pd[k]))[:5]
                for x in h:
                    mem = np.r_[members, x]
                    s_, r_ = fit_orbit(st2, t_ref, et[E[mem]], eobs[E[mem]], uall[mem])
                    if np.all(np.isfinite(s_)) and r_.max() < 1.0:
                        members, st2, changed = mem, s_, True
                        memexp.add(e)
                        pd = predict_dirs(st2, t_ref, et[done], eobs[done])
                        break
        if changed:
            nc = finish(np.array(sorted(members.tolist()), int), st2)
            if nc is not None:
                results_.append(nc)
                nref += 1
    if nref:
        add_forced(results_, FP, done, enight)
        results_ = select(results_)
        publish(results_, cands, t_ref, write)
    log(f"refinement extended {nref} candidates")
    log('DEBUG reasons', REASON)
    from collections import Counter
    log("(nmem, nn) counts", sorted(Counter((len(c["members"]), c["nn"]) for c in results_).items()))
    for cnd in sorted(results_, key=lambda c: -c["pp"])[:40]:
        log(f"cand p={cnd['pp']:.3f} nn={cnd['nn']} nmem={len(cnd['members'])} ncov={cnd['ncov']} miss={cnd['nmiss']} bl={cnd['nblend']} msn={cnd['medsn']:.1f} mstd={cnd['mstd']:.2f} sumsn={cnd['sumsn']:.1f} rms={cnd['rms']:.2f} r={cnd['r']:.1f} mag={cnd['mag']}")


def p_of(c):
    """Heuristic confidence. 3 detections = zero degrees of freedom for an orbit: weak by construction."""
    nm = len(c["members"])
    z = -3.2 + 1.3 * min(nm - 3, 6) + 0.7 * (c["nn"] - 3) + 0.04 * min(c["score"], 60)
    wmiss = 1.0 if c["medsn"] >= 8 else 0.5
    z -= wmiss * 1.3 * c["nmiss"]
    z -= 3.0 * max(c["rms"] - 0.4, 0)
    # residuals large for the S/N (noise-peak linkages sit at ~0.5" rms; real faint bodies ~0.2")
    z -= 2.0 * max(c["rms"] / (0.8 / max(c["medsn"], 3.0) + 0.08) - 1.2, 0)
    z -= 1.5 * max(c["mstd"] - 0.5, 0)
    if c.get("n_on", 0) >= 3:
        # forced photometry along the orbit: a real body is positive in (nearly) every exposure it lands on
        z += 3.0 * (c["fpos"] - 0.7)
        z += 0.5 * float(np.clip(c["fnights"] - c["nn"], -2, 2))
        # evidence beyond the 3 detections an orbit can always fit
        z -= 0.8 * max(0, 3 - (len(c["members"]) - 3 + c["nm_pos"]))
        if c["n_on"] < 0.5 * c.get("ndone", c["n_on"]):
            z -= 1.0
    return 1 / (1 + math.exp(-z))


def add_forced(rs, FP, done, enight):
    """Forced S/N along each candidate's orbit on every exposure with a stored map (batched per exposure)."""
    todo = [c for c in rs if "fpos" not in c]
    if not todo:
        return
    P = np.stack([c["pd"] for c in todo])
    fsa = np.full((len(todo), len(done)), np.nan)
    for k, e in enumerate(done):
        if FP.has(e):
            fsa[:, k] = FP.sample(e, P[:, k, :])
    en = enight[done]
    for c, fs in zip(todo, fsa):
        on = np.isfinite(fs)
        n_on = int(on.sum())
        if n_on:
            stack = float(np.clip(fs[on], -5, 30).sum() / math.sqrt(n_on))
            nonmem = on & ~np.isin(done, list(c["memexp"]))
            nm_pos = int((fs[nonmem] > 2.0).sum()); nm_n = int(nonmem.sum())
            nm_mean = float(np.mean(np.clip(fs[nonmem], -5, 30))) if nm_n else 0.0
            ngood = 0
            for nn_ in np.unique(en[on]):
                mm = on & (en == nn_)
                if fs[mm].sum() / math.sqrt(mm.sum()) > 3.0:
                    ngood += 1
            fpos = float(np.mean(fs[on] > 2.0)); fmed = float(np.median(fs[on]))
        else:
            stack, nm_pos, nm_n, nm_mean, ngood, fpos, fmed = 0.0, 0, 0, 0.0, 0, 0.0, 0.0
        c.update(stack=stack, n_on=n_on, nm_pos=nm_pos, nm_n=nm_n, nm_mean=nm_mean, fnights=ngood, fpos=fpos,
                 fmed=fmed, ndone=len(done))


def select(results_):
    """Greedy by p: drop candidates sharing >= 2 detections (or > half) with a better one."""
    for c in results_:
        c["pp"] = p_of(c)
    out, used = [], {}
    for c in sorted(results_, key=lambda c: -c["pp"]):
        m = c["members"].tolist()
        sh = sum(1 for x in m if x in used)
        if sh >= 2 or sh > len(m) / 2:
            continue
        out.append(c)
        for x in m:
            used[x] = 1
    return out


def publish(results_, cands, t_ref, write):
    import os, json
    cands.clear()
    if os.environ.get("TNO_DEBUG"):
        json.dump([dict(p=p_of(c), nm=len(c["members"]), nn=c["nn"], ncov=c["ncov"], miss=c["nmiss"], msn=c["medsn"],
                        rms=c["rms"], mstd=c["mstd"], r=c["r"], stack=c.get("stack"), n_on=c.get("n_on"), nm_pos=c.get("nm_pos"),
                        nm_n=c.get("nm_n"), nm_mean=c.get("nm_mean"), fpos=c.get("fpos"), fmed=c.get("fmed"), fnights=c.get("fnights"), state=[float(x) for x in c["state"]]) for c in results_],
                  open(os.environ["TNO_DEBUG"], "w"))
    for c in results_:
        c["pp"] = p_of(c)
        d = dict(epoch_mjd=float(t_ref), state=[float(x) for x in c["state"]], p=round(float(c["pp"]), 5))
        if c["mag"] is not None and np.isfinite(c["mag"]):
            d["mag"] = round(c["mag"], 2)
        cands.append(d)
    write()
