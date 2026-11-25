"""Gen 5a: forced-photometry pair linking ("pair + pixel stack").

The hash linker needs a seed detection on >= 3 nights.  A body near the per-exposure detection limit often has
seeds on only 2 nights.  Here every cross-night seed pair (at each assumed inverse distance gamma, BK frame, as in
link.py) defines a linear BK trajectory; its predicted position on every other exposure with a stored S/N map is
sampled (forced S/N, 3x3 max on the 2x-decimated matched-filter map) and summed.  Pairs whose trajectory shows
flux on other nights become linkages for the normal orbit-fit / catalogue-extension stage in link.py.
All numpy, batched over pairs and per CCD.
"""
import math, time
import numpy as np
from scipy.spatial import cKDTree
from detect import gnomonic, poly2
from orbit import GM, C_AU, ARC, T0, log


class FastForced:
    """Forced S/N lookup for points given in tangent-frame coordinates theta (frame F)."""

    def __init__(self, results, F):
        self.F = F
        self.ex = {}
        for e, o in results.items():
            ms = o.get("maps") or []
            if not ms:
                continue
            cen, geo = [], []
            for g, m in ms:
                cu = np.array([math.cos(g["dc"]) * math.cos(g["rc"]), math.cos(g["dc"]) * math.sin(g["rc"]), math.sin(g["dc"])])
                cf = F @ cu
                cen.append(cf[:2] / cf[2])
                geo.append((g, m))
            cen = np.array(cen)
            st = 30 * ARC
            lo = cen.min(0) - 0.25 * math.pi / 180
            n = np.ceil((cen.max(0) + 0.25 * math.pi / 180 - lo) / st).astype(int) + 1
            gx, gy = np.meshgrid(lo[0] + (np.arange(n[0]) + 0.5) * st, lo[1] + (np.arange(n[1]) + 0.5) * st, indexing="ij")
            k = min(2, len(geo))
            _, nb = cKDTree(cen).query(np.c_[gx.ravel(), gy.ravel()], k=k)
            self.ex[e] = ((lo, st, n, nb.reshape(n[0], n[1], k).astype(np.int16)), geo)

    def set_static(self, th, snr):
        """Bitmap (1" cells) of stationary sources (dilated ~2", bright ones by their halo radius): forced values
        there are set to nan, else every trajectory crossing a star looks like a detection."""
        c = 1.0 * ARC
        lo = th.min(0) - 60 * ARC
        n = np.ceil((th.max(0) + 60 * ARC - lo) / c).astype(int) + 1
        bm = np.zeros((n[0], n[1]), bool)
        ix = ((th[:, 0] - lo[0]) / c).astype(np.int64); iy = ((th[:, 1] - lo[1]) / c).astype(np.int64)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                if dx * dx + dy * dy <= 5:
                    bm[np.clip(ix + dx, 0, n[0] - 1), np.clip(iy + dy, 0, n[1] - 1)] = True
        b = np.flatnonzero(snr > 80)
        for q in b[np.argsort(-snr[b])][:20000]:
            r = int(min(max(2.5 * math.sqrt(snr[q] / 80.0), 2.5), 45.0)) + 1
            x0, y0 = ix[q], iy[q]
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            disc = xx * xx + yy * yy <= r * r
            xa, xb = max(x0 - r, 0), min(x0 + r + 1, n[0]); ya, yb = max(y0 - r, 0), min(y0 + r + 1, n[1])
            if xa >= xb or ya >= yb:
                continue
            bm[xa:xb, ya:yb] |= disc[xa - x0 + r:xb - x0 + r, ya - y0 + r:yb - y0 + r]
        self.static = (lo, c, n, bm)
        log(f"fpair static bitmap {n.tolist()}, {bm.mean() * 100:.1f}% masked")

    def is_static(self, th):
        if getattr(self, "static", None) is None:
            return np.zeros(len(th), bool)
        lo, c, n, bm = self.static
        ix = ((th[:, 0] - lo[0]) / c).astype(np.int64); iy = ((th[:, 1] - lo[1]) / c).astype(np.int64)
        ok = (ix >= 0) & (ix < n[0]) & (iy >= 0) & (iy < n[1])
        out = np.zeros(len(th), bool)
        out[ok] = bm[ix[ok], iy[ok]]
        return out

    def has(self, e):
        return e in self.ex

    def sample(self, e, th, mx=True):
        """th (N,2) tangent coords -> forced S/N (nan off-chip).  mx: max over the 3x3 decimated pixels around."""
        out = np.full(len(th), np.nan, np.float32)
        if e not in self.ex or len(th) == 0:
            return out
        (lo, st, n, lut), geo = self.ex[e]
        k = lut.shape[2]
        ix = np.clip(((th[:, 0] - lo[0]) / st).astype(np.int64), 0, n[0] - 1)
        iy = np.clip(((th[:, 1] - lo[1]) / st).astype(np.int64), 0, n[1] - 1)
        nb = lut[ix, iy]
        uf = np.c_[th, np.ones(len(th))]
        u = uf @ self.F
        u /= np.linalg.norm(u, axis=1)[:, None]
        ra = np.arctan2(u[:, 1], u[:, 0]); de = np.arcsin(np.clip(u[:, 2], -1, 1))
        todo = np.ones(len(th), bool)
        for kk in range(k):
            cand = np.flatnonzero(todo)
            if len(cand) == 0:
                break
            ids = nb[cand, kk]
            o = np.argsort(ids, kind="stable")
            cs = ids[o]
            brk = np.flatnonzero(np.diff(cs)) + 1
            for a0, a1 in zip(np.r_[0, brk], np.r_[brk, len(cs)]):
                c = int(cs[a0])
                sel = cand[o[a0:a1]]
                g, m = geo[c]
                xi, eta = gnomonic(ra[sel], de[sel], g["rc"], g["dc"])
                A = poly2(xi, eta)
                x = A @ g["cx"]; y = A @ g["cy"]
                inside = (x >= 4) & (x <= g["nx"] - 5) & (y >= 4) & (y <= g["ny"] - 5)
                if not inside.any():
                    continue
                sel, x, y = sel[inside], x[inside] / 2.0, y[inside] / 2.0
                if mx:
                    xi_ = np.clip(np.rint(x).astype(int), 1, m.shape[1] - 2)
                    yi_ = np.clip(np.rint(y).astype(int), 1, m.shape[0] - 2)
                    v = np.full(len(sel), -np.inf, np.float32)
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            v = np.maximum(v, m[yi_ + dy, xi_ + dx].astype(np.float32))
                else:
                    x0 = np.clip(np.floor(x).astype(int), 0, m.shape[1] - 2); y0 = np.clip(np.floor(y).astype(int), 0, m.shape[0] - 2)
                    fx, fy = x - x0, y - y0
                    v = ((1 - fx) * (1 - fy) * m[y0, x0].astype(np.float32) + fx * (1 - fy) * m[y0, x0 + 1] +
                         (1 - fx) * fy * m[y0 + 1, x0] + fx * fy * m[y0 + 1, x0 + 1])
                out[sel] = v
                todo[sel] = False
        return out


def search(seeds, theta, ts, os_, ns, E, sn, et, eobs_f, enight, done, t_ref, grid, order, core, FF,
           deadline, max_links=6000, zmin=4.5):
    """Returns links {key: (rms, gamma, sol)} like link.py's hash linker (key = sorted detection indices)."""
    un = np.unique(ns)
    mexp = np.array([e for e in done if FF.has(e)], int)
    if len(mexp) < 3 or len(un) < 3:
        return {}
    men = enight[mexp]
    nidx_m = np.searchsorted(un, men)          # nights of map exposures (may be outside un -> clipped)
    best = {}
    t0 = time.time()
    idx_n = [np.flatnonzero(ns == n_) for n_ in un]
    ngam = 0
    for gidx in order:
        if time.time() - T0 > deadline:
            log(f"fpair stopped after {ngam} gammas")
            break
        g = grid[gidx]
        ngam += 1
        tp = ts - 1.0 / (C_AU * g)
        k_ = 0.5 * GM * g ** 3
        f = 1.0 - k_ * tp * tp - g * os_[:, 2]
        phi = theta * f[:, None] + g * os_[:, :2]
        uu = 1.0 - k_ * tp * tp
        npara = math.sqrt(2 * GM * g ** 3)
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
        M = len(I)
        if M == 0:
            continue
        # forced S/N on every map exposure not belonging to the pair's own exposures
        tpe = et[mexp] - t_ref - 1.0 / (C_AU * g)
        uue = 1.0 - k_ * tpe * tpe
        fe = uue - g * eobs_f[mexp, 2]
        # forced values standardised per exposure over all pairs (almost all are noise): removes the bias of the
        # 3x3 max; cascade: after half of the exposures drop pairs whose partial sum is negative
        S = np.zeros(M, np.float32); NON = np.zeros(M, np.int16)
        nng = np.zeros((M, len(un)), np.float32); nnc = np.zeros((M, len(un)), np.int16)
        live = np.arange(M)
        eo = np.random.default_rng(0).permutation(len(mexp))
        for kn, k in enumerate(eo):
            if kn == len(eo) // 2 and kn >= 2:
                live = live[S[live] > 0]
            e = mexp[k]
            th = (AL[live] * uue[k] + AD[live] * tpe[k] - g * eobs_f[e, :2]) / fe[k]
            v = FF.sample(e, th)
            v[FF.is_static(th)] = np.nan
            v[(E[seeds[I[live]]] == e) | (E[seeds[J[live]]] == e)] = np.nan
            on = np.isfinite(v)
            if on.sum() < 50:
                continue
            mu = float(np.median(v[on])); sg = float(np.percentile(v[on], 84) - np.percentile(v[on], 16)) / 2
            vc = np.where(on, np.clip((v - mu) / max(sg, 0.3), -3, 3.5), 0).astype(np.float32)
            S[live] += vc; NON[live] += on
            if men[k] in un:
                nng[live, nidx_m[k]] += vc; nnc[live, nidx_m[k]] += on
        zf = S / np.sqrt(np.maximum(NON, 1))
        ztot = zf
        nst = nng / np.sqrt(np.maximum(nnc, 1))
        pi_ = np.searchsorted(un, ns[I]); qi_ = np.searchsorted(un, ns[J])
        nst[np.arange(M), pi_] = 0; nst[np.arange(M), qi_] = 0
        nother = (nst >= 2.5).sum(1)
        # evidence from the other nights only (same-night exposures re-measure the seeds themselves)
        nnc[np.arange(M), pi_] = 0; nnc[np.arange(M), qi_] = 0
        no_ = nnc.sum(1)
        zo = (nst * np.sqrt(nnc)).sum(1) / np.sqrt(np.maximum(no_, 1))
        ztot = zo
        ok = (no_ >= 2) & (nother >= 1) & (zo >= zmin)
        sel = np.flatnonzero(ok)
        sel = sel[np.argsort(-ztot[sel])[:2000]]
        for s in sel:
            a, b = int(seeds[I[s]]), int(seeds[J[s]])
            key = (a, b) if a < b else (b, a)
            z = float(ztot[s])
            if key not in best or best[key][0] < z:
                best[key] = (z, g, np.stack([AL[s], AD[s]]))
        if ngam % 5 == 1:
            log(f"fpair gamma R {1 / g:.1f}: {M} pairs, {len(sel)} pass, {len(best)} total ({time.time() - t0:.0f}s)")
    items = sorted(best.items(), key=lambda kv: -kv[1][0])[:max_links]
    log(f"fpair: {len(best)} pair linkages, keeping {len(items)}; top z {[round(v[0], 1) for _, v in items[:5]]}")
    return {k: (-v[0], v[1], v[2]) for k, v in items}
