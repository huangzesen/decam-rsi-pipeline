"""gen087 armB: truth-guided DIAGNOSIS only (env TNO_ARMB_DIAG=truth.json, default off; never used for ranking)."""
import json, math, os
import numpy as np

ARCR = math.pi / 180 / 3600
_S = {}


def on():
    return bool(os.environ.get("TNO_ARMB_DIAG"))


def setup(fm, t_all, log):
    try:
        tr = json.load(open(os.environ["TNO_ARMB_DIAG"]))
        B = tr["bodies"]
        U = []   # (B, N, 3) per frame
        P = []   # present per frame
        for b in B:
            ra = np.radians(np.array(b["ra"], float)); de = np.radians(np.array(b["dec"], float))
            u = np.stack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)], 1)
            ex = set(int(x) for x in b["exps"])
            uf, pf = [], []
            for m in fm:
                v = u[m].mean(0); uf.append(v / np.linalg.norm(v)); pf.append(any(i in ex for i in m))
            U.append(uf); P.append(pf)
        _S.update(B=B, U=np.array(U), P=np.array(P), cls=np.array([b["cls"] for b in B]),
                  r=np.array([b["r_au"] for b in B]))
        log(f"armB diag: {len(B)} bodies loaded")
    except Exception as ex:
        log("armB diag setup failed", repr(ex)[:200]); _S.clear()


def pix(TF0, x0, y0, PIX, U):
    q = U @ TF0.T
    return np.rint((q[..., 0] / q[..., 2] - x0) / PIX).astype(int), np.rint((q[..., 1] / q[..., 2] - y0) / PIX).astype(int)


def maps_stage(maps, TF0, x0, y0, PIX, W, H, log):
    if not _S:
        return
    try:
        ix, iy = pix(TF0, x0, y0, PIX, _S["U"])
        Bn, N = ix.shape
        V = np.full((Bn, N), np.nan)
        for j in range(Bn):
            for n in range(N):
                if _S["P"][j, n] and 1 <= ix[j, n] < W - 1 and 1 <= iy[j, n] < H - 1:
                    V[j, n] = float(maps[n, iy[j, n] - 1:iy[j, n] + 2, ix[j, n] - 1:ix[j, n] + 2].float().max())
        _S["mapv"] = V
        for c in "ABC":
            s = _S["cls"] == c
            n5 = (V[s] >= 5).sum(1); n4 = (V[s] >= 4).sum(1)
            log(f"armB diag maps {c}: per-body median 3x3max z {np.round(np.nanmedian(V[s], 1), 1).tolist()}; "
                f"frames>=4 {n4.tolist()}; frames>=5 {n5.tolist()}")
    except Exception as ex:
        log("armB diag maps failed", repr(ex)[:200])


def night_stage(NMs, GC, GI, PK, ZK, nu, night, TF0, x0, y0, PIX, W, H, NZ, log):
    if not _S:
        return
    try:
        Bn = len(_S["B"])
        nN = len(nu)
        val = np.full((Bn, nN), np.nan); pkz = np.full((Bn, nN), np.nan); rk = np.full((Bn, nN), np.nan)
        for k, u in enumerate(nu):
            idx = np.flatnonzero(night == u)
            for j in range(Bn):
                if not _S["P"][j, idx].any():
                    continue
                uu = _S["U"][j, idx].mean(0); uu /= np.linalg.norm(uu)
                ix, iy = pix(TF0, x0, y0, PIX, uu[None])
                ix, iy = int(ix[0]), int(iy[0])
                if not (0 <= ix < W and 0 <= iy < H):
                    continue
                g = GI[k][int(np.argmin(np.abs(GC[k] - 1 / _S["r"][j])))]
                val[j, k] = float(NMs[g][iy * W + ix])
                P = PK[g]
                if len(P):
                    dd = np.hypot(P[:, 0] - ix, P[:, 1] - iy)
                    i0 = int(np.argmin(dd))
                    if dd[i0] <= 3.5:
                        pkz[j, k] = ZK[g][i0]
                        rk[j, k] = float((ZK[g] > ZK[g][i0]).sum())
        _S["nval"] = val; _S["npk"] = pkz
        for c in "ABC":
            s = _S["cls"] == c
            ng = (val[s] >= 2.5).sum(1); nz = (val[s] >= NZ).sum(1); npk = np.isfinite(pkz[s]).sum(1)
            log(f"armB diag night {c}: nights co-add>=2.5 {ng.tolist()}; >=NZ {nz.tolist()}; kept peak (cap) {npk.tolist()}; "
                f"bodies with >=3 kept-peak nights {int((npk >= 3).sum())}/{int(s.sum())}, >=2 {int((npk >= 2).sum())}")
        log("armB diag night B coadd values:", [np.round(v, 1).tolist() for v in val[_S['cls'] == 'B']])
    except Exception as ex:
        log("armB diag night failed", repr(ex)[:200])


def seeds_stage(name, S, t_ep, t, obs, log, kinds=None, tol=3.0):
    """S (K,6) states at t_ep; per body: seeds within tol arcsec on >= 3 nights (frames)."""
    if not _S or S is None or len(S) == 0:
        return
    try:
        import orbit
        night = np.floor(t - 0.5).astype(int)
        hit = {}
        for c0 in range(0, len(S), 2000):
            u = orbit.predict_dirs_b(np.asarray(S[c0:c0 + 2000]), t_ep, t, obs)   # (K,N,3)
            sep = np.arccos(np.clip(np.einsum("bni,kni->bkn", _S["U"], u), -1, 1)) / ARCR
            for j in range(len(_S["B"])):
                ok = (sep[j] < tol) & _S["P"][j][None]
                nn = np.array([len(set(night[ok[k]].tolist())) for k in range(ok.shape[0])])
                for k in np.flatnonzero(nn >= 3):
                    hit.setdefault(j, []).append(c0 + int(k))
        out = []
        for c in "ABC":
            js = [j for j in range(len(_S["B"])) if _S["cls"][j] == c]
            got = [j for j in js if j in hit]
            kd = [sorted(set(int(kinds[i]) for i in hit[j]))[:3] if kinds is not None else [] for j in got]
            out.append(f"{c} {len(got)}/{len(js)} first-idx {[hit[j][0] for j in got]} kinds {kd}")
        log(f"armB diag seeds[{name}] (K={len(S)}, tol {tol}\"): " + "; ".join(out))
    except Exception as ex:
        log("armB diag seeds failed", repr(ex)[:200])
