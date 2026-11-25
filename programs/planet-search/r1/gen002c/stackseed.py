"""Pair seeding verified by a forced stack (generation 2c).

The hash linker needs >= 3 seeds (S/N >= 4.6 on single-exposure nights) on 3 nights.  This stage needs only two
detections of S/N >= ~4 on two different nights: for every assumed inverse distance gamma (BK frame, same grid as
the linker) each cross-night pair fixes (alpha, alpha-dot); the path is predicted on every other exposure and the
matched-filter S/N maps are sampled there (forced photometry).  Pairs whose forced stack over the other exposures
(and at least one third night) is significant are extended with the full >= 3 sigma catalogue, orbit-fitted and
scored by the same p_of as the linker's candidates.
"""
import math, time
import numpy as np
from scipy.spatial import cKDTree
from orbit import GM, C_AU, ARC, T0, log, fit_orbit, predict_dirs


def run_stack(ctx, deadline):
    (u, F, gi, sn, E, et, eobs, enight, done, t_ref, half, core, grid, order, FP, extend, finish, owner,
     results_, seen_mem) = ctx
    night = enight[E]
    maps_e = [e for e in done if FP.has(e)]
    if len(maps_e) < 3:
        log("stackseed: too few S/N maps")
        return 0
    # anchors: usable detections S/N >= 4 (cap per exposure, brightest first), not already owned by a candidate
    A_SN = 4.0
    cand = gi[(sn[gi] >= A_SN) & (sn[gi] < 60)]
    cand = np.array([x for x in cand if x not in owner], int)
    if len(cand) == 0:
        return 0
    keep = []
    for e in np.unique(E[cand]):
        m = cand[E[cand] == e]
        keep.append(m[np.argsort(-sn[m])[:2500]])
    anc = np.concatenate(keep)
    ns = night[anc]
    un = np.unique(ns)
    if len(un) < 3:
        return 0
    log(f"stackseed: {len(anc)} anchors over {len(un)} nights, {len(maps_e)} maps")
    us = u[anc] @ F.T
    theta = us[:, :2] / us[:, 2:3]
    obs_f = eobs @ F.T
    os_ = obs_f[E[anc]]
    ts = et[E[anc]] - t_ref
    me = np.array(maps_e)
    tme = et[me] - t_ref
    ome = obs_f[me]
    nme = enight[me]
    best = {}
    t0 = time.time()
    ZTHR = 4.0
    for kk, gidx in enumerate(order):
        if time.time() - T0 > deadline:
            log(f"stackseed: stopped at {kk}/{len(order)} gammas")
            break
        g = grid[gidx]
        k_ = 0.5 * GM * g ** 3
        tp = ts - 1.0 / (C_AU * g)
        f = 1.0 - k_ * tp * tp - g * os_[:, 2]
        phi = theta * f[:, None] + g * os_[:, :2]
        uu = 1.0 - k_ * tp * tp
        npara = math.sqrt(2 * GM * g ** 3)
        idx_n = [np.flatnonzero(ns == n_) for n_ in un]
        trees = [cKDTree(phi[ix]) for ix in idx_n]
        I, J, AL, AD = [], [], [], []
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
                I.append(i[ok]); J.append(j[ok]); AL.append(al[ok]); AD.append(ad[ok])
        if not I:
            continue
        I = np.concatenate(I); J = np.concatenate(J); AL = np.concatenate(AL); AD = np.concatenate(AD)
        npair = len(I)
        if npair == 0:
            continue
        # predicted positions on each exposure with a map -> forced S/N
        tpe = tme - 1.0 / (C_AU * g)
        uue = 1.0 - k_ * tpe * tpe
        fe = uue - g * ome[:, 2]
        FS = np.full((npair, len(me)), np.nan, np.float32)
        for k in range(len(me)):
            ph = AL * uue[k] + AD * tpe[k]
            th = (ph - g * ome[k, :2]) / fe[k]
            v = np.c_[th, np.ones(npair)]
            v /= np.linalg.norm(v, axis=1)[:, None]
            FS[:, k] = FP.sample(me[k], v @ F)
        ea, eb = E[anc[I]], E[anc[J]]
        na, nb = ns[I], ns[J]
        excl = (me[None, :] == ea[:, None]) | (me[None, :] == eb[:, None])
        on = np.isfinite(FS) & ~excl
        third = on & (nme[None, :] != na[:, None]) & (nme[None, :] != nb[:, None])
        n_on = on.sum(1)
        z = np.where(on, np.clip(np.nan_to_num(FS), -3, 8), 0).sum(1) / np.sqrt(np.maximum(n_on, 1))
        z3 = np.where(third, np.clip(np.nan_to_num(FS), -3, 8), 0).sum(1) / np.sqrt(np.maximum(third.sum(1), 1))
        sel = np.flatnonzero((third.sum(1) >= 1) & (z >= ZTHR) & (z3 >= 2.5))
        if len(sel) > 400:
            sel = sel[np.argsort(-z[sel])[:400]]
        for s in sel:
            key = (int(anc[I[s]]), int(anc[J[s]]))
            if key not in best or best[key][0] < z[s]:
                best[key] = (float(z[s]), g, AL[s].copy(), AD[s].copy())
        if kk % 6 == 0:
            log(f"stackseed gamma {gidx} (R {1 / g:.1f}): {npair} pairs, {len(sel)} pass, {len(best)} total")
    log(f"stackseed: {len(best)} verified pairs in {time.time() - t0:.0f}s")
    # extension/fit, strongest first; skip pairs whose anchors are already used
    nadd = 0
    used = set(owner.keys())
    for key, (z, g, al, ad) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        if time.time() - T0 > deadline + 0.05 * (deadline / 0.85):
            log("stackseed: fit stage stopped")
            break
        if key[0] in used or key[1] in used:
            continue
        rf = np.array([al[0], al[1], 1.0]) / g
        vf = np.array([ad[0], ad[1], 0.0]) / g
        state0 = np.r_[F.T @ rf, F.T @ vf]
        r = extend(np.array(key, int), state0)
        if r is None:
            continue
        members, st2 = r
        if len(members) < 3 or len(np.unique(night[members])) < 3:
            continue
        c = finish(members, st2)
        if c is None:
            continue
        mkey = tuple(members.tolist())
        if mkey in seen_mem:
            continue
        seen_mem.add(mkey)
        c["src"] = "stack"
        results_.append(c)
        nadd += 1
        for x in members:
            used.add(x)
            owner[x] = len(results_) - 1
    log(f"stackseed: added {nadd} candidates")
    return nadd
