"""Vanilla CPU tracklet-less linker: Bernstein-Khushalani transform over a distance grid, then pair clustering.

Usage: python3 link_cpu.py DATA.npz [--season 2] [--core 1.5] [--rmin 30] [--rmax 150] [--dgamma 1e-4]
                           [--cell 3] [--dtmin 10] [--minpairs 3] [--minnights 4] [--gammas K] [--out OUT.json]

Per sky patch and time window (one observing season), for each asserted distance 1/gamma (gamma-dot = 0):
  1. transform every detection to the frame where a body at that distance moves on a straight line
     (Holman, Napier, Payne & Kurlander 2025, arXiv:2506.02144, eqs. for phi_x, phi_y; z axis at the patch centre;
     radial gravity in f, transverse gravity in the model u(t') = 1 - GM gamma^3 t'^2 / 2, light time 1/(c gamma)),
  2. pair every detection with every detection on a later night (dtmin <= dt <= window) that a bound orbit can
     reach, n_para = sqrt(2 GM gamma^3) (no same-night tracklets are needed),
  3. turn each pair into the state (alpha, beta, alpha-dot, beta-dot) at the window mid-time; keep states whose
     position lies in the patch core,
  4. hash states into 4-D cells (size `cell` arcsec in position and in velocity x half-window), two grids offset by
     half a cell; cells with >= minpairs pairs are candidates,
  5. fit a straight line to each candidate's detections, clip, extend to every exposure in the window, keep linkages
     with >= minnights distinct nights.
This is the HelioLinC idea (Holman et al. 2018) with single detections in place of tracklets, in the PS1 Part 1 frame.
"""
import argparse, json, sys, time
import numpy as np
from scipy.spatial import cKDTree
from orbits import GM, C_AU, ARCSEC, radec_to_unit, tangent_frame, elements_to_rv, ecl_to_eq

OFF = 1 << 15


def load(path):
    d = dict(np.load(path))
    meta = json.load(open(path.replace(".npz", ".json")))
    return d, meta


def window(d, meta, season, core_deg, span=1):
    """Detections of `span` consecutive seasons from `season`; frame and reference time."""
    s0, s1 = meta["seasons_mjd"][season][0], meta["seasons_mjd"][season + span - 1][1]
    et = d["exp_t"]
    ew = np.flatnonzero((et >= s0) & (et <= s1 + 1))
    sel = np.flatnonzero(np.isin(d["det_exp"], ew))
    F = tangent_frame(meta["ra0"], meta["dec0"])
    t_ref = 0.5 * (et[ew[0]] + et[ew[-1]])
    half = 0.5 * (et[ew[-1]] - et[ew[0]])
    u = radec_to_unit(d["det_ra"][sel], d["det_dec"][sel]) @ F.T
    obs = d["exp_obs"] @ F.T
    W = dict(sel=sel, exp=d["det_exp"][sel], t=et[d["det_exp"][sel]] - t_ref, night=np.floor(et[d["det_exp"][sel]]).astype(np.int64),
             theta=u[:, :2] / u[:, 2:3], obs=obs[d["det_exp"][sel]], t_ref=t_ref, half=half, F=F,
             core=np.radians(core_deg), exps=ew, exp_t=et - t_ref, exp_obs=obs, exp_night=np.floor(et).astype(np.int64))
    return W


def transform(W, g, gdot=0.0):
    """phi = theta f + gamma x_E; returns phi (N,2), t' (N,), u(t') (N,)."""
    tp = W["t"] - 1.0 / (C_AU * g)
    k = 0.5 * GM * g ** 3
    f = 1.0 + gdot * tp - k * tp * tp - g * W["obs"][:, 2]
    phi = W["theta"] * f[:, None] + g * W["obs"][:, :2]
    return phi, tp, 1.0 - k * tp * tp


def pair_states(phi, tp, u, i, j):
    D = u[i] * tp[j] - u[j] * tp[i]
    a = (phi[i] * tp[j, None] - phi[j] * tp[i, None]) / D[:, None]
    ad = (u[i, None] * phi[j] - u[j, None] * phi[i]) / D[:, None]
    return a, ad


def keys_of(a, ad, half, cell, shift):
    q = np.floor(np.concatenate([a, ad * half], 1) / cell + shift).astype(np.int64) + OFF
    q = np.clip(q, 0, 2 * OFF - 1)
    return (q[:, 0] << 48) | (q[:, 1] << 32) | (q[:, 2] << 16) | q[:, 3]


def fit_line(phi, tp, u, w=None):
    A = np.stack([u, tp], 1)
    sol, *_ = np.linalg.lstsq(A, phi, rcond=None)
    res = phi - A @ sol
    return sol, np.sqrt((res ** 2).sum(1))


def link_gamma(W, g, gd, a, timing):
    t0 = time.perf_counter()
    phi, tp, u = transform(W, g, gd)
    npara = np.sqrt(2 * GM * g ** 3)
    reach = W["core"] + npara * (W["half"] + 1) + 5 * a.cell * ARCSEC
    live = np.flatnonzero(np.hypot(phi[:, 0], phi[:, 1]) <= reach)
    nights = W["night"][live]
    o = np.argsort(nights, kind="stable")
    live, nights = live[o], nights[o]
    un, start = np.unique(nights, return_index=True)
    stop = np.append(start[1:], len(live))
    trees = [cKDTree(phi[live[s:e]]) for s, e in zip(start, stop)]
    timing["transform"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    cell = a.cell * ARCSEC
    tolpos = 2 * a.sigma * ARCSEC
    K, I, J = [[], []], [], []
    npairs = 0
    for p in range(len(un)):
        for q in range(p + 1, len(un)):
            dt = un[q] - un[p]
            if dt < a.dtmin:
                continue
            if a.window and dt > a.window:
                break
            sd = trees[p].sparse_distance_matrix(trees[q], npara * (dt + 1) + tolpos, output_type="ndarray")
            if len(sd) == 0:
                continue
            i = live[start[p] + sd["i"]]
            j = live[start[q] + sd["j"]]
            npairs += len(i)
            al, ad = pair_states(phi, tp, u, i, j)
            ok = (np.hypot(al[:, 0], al[:, 1]) <= W["core"]) & (np.abs(ad) <= npara).all(1)
            i, j, al, ad = i[ok], j[ok], al[ok], ad[ok]
            for s, sh in enumerate((0.0, 0.5)):
                K[s].append(keys_of(al, ad, W["half"], cell, sh))
            I.append(i.astype(np.int32)); J.append(j.astype(np.int32))
    timing["pairs"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    if not I:
        return [], npairs, 0
    I = np.concatenate(I); J = np.concatenate(J)
    cands = []
    for s in range(2):
        k = np.concatenate(K[s])
        o = np.argsort(k, kind="stable")
        ks = k[o]
        brk = np.flatnonzero(np.diff(ks)) + 1
        st = np.concatenate([[0], brk]); en = np.append(brk, len(ks))
        big = np.flatnonzero(en - st >= a.minpairs)
        for b in big:
            m = o[st[b]:en[b]]
            cands.append(np.unique(np.concatenate([I[m], J[m]])))
    timing["cluster"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    links = verify(W, phi, tp, u, cands, a)
    timing["verify"] += time.perf_counter() - t0
    return links, npairs, len(I)


def verify(W, phi, tp, u, cands, a):
    tol = a.fit_tol * ARCSEC
    tree = None
    out = []
    seen = set()
    for c in cands:
        key = tuple(c)
        if key in seen:
            continue
        seen.add(key)
        for _ in range(3):
            if len(c) < 3:
                break
            sol, r = fit_line(phi[c], tp[c], u[c])
            if r.max() <= tol:
                break
            c = c[r <= max(tol, np.sort(r)[len(r) // 2] * 3)] if r.max() > tol else c
        if len(c) < 3 or len(np.unique(W["night"][c])) < 3:
            continue
        sol, r = fit_line(phi[c], tp[c], u[c])
        if r.max() > tol:
            continue
        # extend to every exposure in the window
        if tree is None:
            tree = cKDTree(phi)
        te = W["exp_t"][W["exps"]] - 1.0 / (C_AU * a._g)
        ue = 1.0 - 0.5 * GM * a._g ** 3 * te * te
        pred = np.stack([ue, te], 1) @ sol
        hits = tree.query_ball_point(pred, tol)
        add = []
        for k, (e, h) in enumerate(zip(W["exps"], hits)):
            h = np.array([x for x in h if W["exp"][x] == e], int)
            if len(h):
                add.append(h[np.argmin(np.hypot(*(phi[h] - pred[k]).T))])
        c2 = np.unique(np.concatenate([c, np.array(add, int)])) if add else c
        sol, r = fit_line(phi[c2], tp[c2], u[c2])
        c2 = c2[r <= tol]
        nn = len(np.unique(W["night"][c2]))
        if nn >= a.minnights:
            sol, r = fit_line(phi[c2], tp[c2], u[c2])
            out.append(dict(det=c2, nights=nn, rms=float(np.sqrt((r ** 2).mean()) / ARCSEC), gamma=a._g,
                            state=sol.T.ravel().tolist()))
    return out


def merge(links):
    """Drop duplicates and subsets; keep the longest linkage per shared-detection group."""
    links = sorted(links, key=lambda L: (-len(L["det"]), L["rms"]))
    owner = {}
    keep = []
    for L in links:
        ids = set(L["det"].tolist())
        olap = {}
        for x in ids:
            if x in owner:
                olap[owner[x]] = olap.get(owner[x], 0) + 1
        if olap and max(olap.values()) >= 0.5 * len(ids):
            continue
        k = len(keep)
        keep.append(L)
        for x in ids:
            owner.setdefault(x, k)
    return keep


def truth_alpha(d, W):
    """True (alpha, beta) of each body at t_ref (heliocentric position in the patch frame) and its gamma."""
    n = np.sqrt(GM / d["obj_a"] ** 3)
    M = d["obj_M0"] + n * (W["t_ref"] - d["obj_t0"])
    r, _ = elements_to_rv(d["obj_a"], d["obj_e"], d["obj_inc"], d["obj_Om"], d["obj_om"], M)
    x = ecl_to_eq(r) @ W["F"].T
    return x[:, :2] / x[:, 2:3], 1.0 / x[:, 2]


def evaluate(d, W, links, a):
    obj = d["det_obj"][W["sel"]]
    al, gam = truth_alpha(d, W)
    incore = np.hypot(al[:, 0], al[:, 1]) <= W["core"]
    nights = {}
    for b in np.unique(obj[obj >= 0]):
        nights[b] = len(np.unique(W["night"][obj == b]))
    inrange = (1 / gam >= a.rmin) & (1 / gam <= a.rmax)
    findable = [b for b, n in nights.items() if n >= a.minnights and incore[b] and inrange[b]]
    found, false = set(), 0
    for L in links:
        ob = obj[L["det"]]
        vals, cnt = np.unique(ob, return_counts=True)
        top = vals[np.argmax(cnt)]
        if top >= 0 and cnt.max() >= 0.8 * len(ob):
            if len(np.unique(W["night"][L["det"]][ob == top])) >= a.minnights:
                found.add(int(top))
        else:
            false += 1
    fset = set(int(b) for b in findable)
    missed = sorted(fset - found)
    return dict(findable=len(fset), found=len(found & fset), found_other=len(found - fset), false=false,
                missed=[dict(id=b, r=float(1 / gam[b]), nights=nights[b]) for b in missed][:40])


def hypotheses(a, half):
    """(gamma, gamma-dot) grid. gamma uniform with step dgamma; gamma-dot uniform over |gamma-dot| <= vr_frac * n_para
    with the step at which the neglected term alpha-dot * dgdot * t'^2 stays below gdot_tol over the window."""
    H = []
    for g in np.arange(1 / a.rmax, 1 / a.rmin + a.dgamma / 2, a.dgamma):
        npara = np.sqrt(2 * GM * g ** 3)
        n = 1
        if a.gdot_tol > 0:
            step = a.gdot_tol * ARCSEC / (npara * (2 / 3) * half ** 2)
            n = max(1, int(np.ceil(2 * a.vr_frac * npara / step)))
        span = 2 * a.vr_frac * npara
        H += [(g, (k - (n - 1) / 2) * span / n if n > 1 else 0.0) for k in range(n)]
    if a.sample:
        H = [H[k] for k in np.linspace(0, len(H) - 1, a.sample).round().astype(int)]
    return H


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--season", type=int, default=2)
    ap.add_argument("--span", type=int, default=1, help="number of consecutive seasons in one window")
    ap.add_argument("--core", type=float, default=1.5)
    ap.add_argument("--rmin", type=float, default=30.0)
    ap.add_argument("--rmax", type=float, default=150.0)
    ap.add_argument("--dgamma", type=float, default=1e-4)
    ap.add_argument("--gdot-tol", type=float, default=1.0, help="arcsec; 0 = gamma-dot fixed at 0 as in PS1 Part 1")
    ap.add_argument("--vr-frac", type=float, default=1.0, help="gamma-dot range as a fraction of n_para")
    ap.add_argument("--sample", type=int, default=0, help="run only K hypotheses spread over the grid, for timing")
    ap.add_argument("--cell", type=float, default=3.0, help="arcsec")
    ap.add_argument("--fit-tol", type=float, default=1.0, help="arcsec, max residual of a linkage")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--dtmin", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=0.0,
                    help="max pair separation, days; 0 = no limit inside the analysis window (multi-season windows need it)")
    ap.add_argument("--minpairs", type=int, default=3)
    ap.add_argument("--minnights", type=int, default=4)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--budget", type=float, default=0, help="seconds; stop starting new hypotheses after this (0 = none)")
    ap.add_argument("--out", default=None)
    return ap


_G = {}


def _init(a):
    d, meta = load(a.data)
    _G.update(a=a, W=window(d, meta, a.season, a.core, a.span))


def _run(h):
    a, W = _G["a"], _G["W"]
    g, gd = h
    a._g = g
    timing = dict(transform=0.0, pairs=0.0, cluster=0.0, verify=0.0)
    t1 = time.perf_counter()
    L, npairs, nkept = link_gamma(W, g, gd, a, timing)
    return L, timing, dict(r=float(1 / g), gdot=gd, pairs=int(npairs), kept=int(nkept), links=len(L),
                           sec=time.perf_counter() - t1)


def main():
    a = parser().parse_args()
    d, meta = load(a.data)
    W = window(d, meta, a.season, a.core, a.span)
    grid = hypotheses(a, W["half"])
    timing = dict(transform=0.0, pairs=0.0, cluster=0.0, verify=0.0)
    links, per = [], []
    T0 = time.perf_counter()
    if a.procs > 1:
        import multiprocessing as mp
        # longest hypotheses first so the pool drains evenly
        order = sorted(grid, key=lambda h: -h[0])
        res = []
        # fork after numpy/BLAS threads start can deadlock on macOS; spawn there
        ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
        with ctx.Pool(a.procs, _init, (a,)) as pool:
            for r in pool.imap_unordered(_run, order, chunksize=1):
                res.append(r)
                if a.budget and time.perf_counter() - T0 > a.budget:
                    pool.terminate()
                    break
    else:
        _G.update(a=a, W=W)
        res = []
        for h in grid:
            res.append(_run(h))
            if a.budget and time.perf_counter() - T0 > a.budget:
                break
    for L, tm, p in res:
        links += L
        per.append(p)
        for k in timing:
            timing[k] += tm[k]
    wall = time.perf_counter() - T0
    links = merge(links)
    ev = evaluate(d, W, links, a)
    import copy
    a_full = copy.copy(a); a_full.sample = 0
    n_full = len(hypotheses(a_full, W["half"]))
    res = dict(args={k: v for k, v in vars(a).items() if not k.startswith("_")}, n_det=int(len(W["sel"])), procs=a.procs,
               n_hyp=len(grid), n_hyp_full=n_full, half=W["half"], n_done=len(per), n_gamma=len(set(g for g, _ in grid)), wall_s=wall, timing=timing, pairs=int(sum(p["pairs"] for p in per)),
               pairs_per_s=sum(p["pairs"] for p in per) / max(1e-9, timing["pairs"]),
               cpu_s=sum(p["sec"] for p in per), eval=ev, per_gamma=per)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_gamma", "args")}, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
