import pickle, numpy as np, sys
sys.path.insert(0, "$RSI_SANDBOX/r2/gen012/workspace/submission")
from finder import _terms
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
res = pickle.load(open("cats.pkl", "rb"))
refa = np.concatenate([c for _, c in res[6] if c is not None])
ref = refa[refa[:, 2] >= 30]
tr = cKDTree(ref[:, :2])
good = {cc["ext"]: cc["ccdnum"] for cc, _ in res[2]}
def fit(X, Y, snr):
    o = np.argsort(-snr)[:250]; o = o[snr[o] >= 15]
    L = 150.0; bs = 4.0
    P = np.stack([X[o], Y[o]], 1)
    idx = tr.query_ball_point(P, L * 1.42)
    dx = np.concatenate([ref[j, 0] - P[k, 0] for k, j in enumerate(idx)]); dy = np.concatenate([ref[j, 1] - P[k, 1] for k, j in enumerate(idx)])
    H, xe, ye = np.histogram2d(dx, dy, bins=int(2 * L / bs), range=[[-L, L], [-L, L]])
    Hs = ndi.uniform_filter(H, 3) * 9
    ij = np.unravel_index(np.argmax(Hs), Hs.shape); med = np.median(Hs)
    if Hs.max() < max(15, med + 8 * np.sqrt(med + 1)): return None
    sx = xe[ij[0]] + bs / 2; sy = ye[ij[1]] + bs / 2
    m = snr >= 8
    x = X[m]; y = Y[m]; xc = x.mean(); yc = y.mean()
    M = np.array([[sx, 1, 0], [sy, 0, 1.0]])   # on centred coords, output = offset from (xc,yc)... 
    A = np.stack([np.ones(len(x)), x - xc, y - yc], 1)
    for tol in (8.0, 4.0, 2.0, 1.2, 0.8):
        q = np.stack([xc + A @ M[0], yc + A @ M[1]], 1)
        d, j = tr.query(q)
        k = d < tol
        if k.sum() < 15: return None
        M = np.stack([np.linalg.lstsq(A[k], ref[j[k], 0] - xc, rcond=None)[0], np.linalg.lstsq(A[k], ref[j[k], 1] - yc, rcond=None)[0]])
    q = np.stack([xc + A @ M[0], yc + A @ M[1]], 1)
    d, j = tr.query(q); k = d < 0.8
    r = q[k] - ref[j[k], :2]
    # quadratic test
    A2 = np.stack([np.ones(len(x)), x - xc, y - yc, (x - xc) ** 2 / 500, (x - xc) * (y - yc) / 500, (y - yc) ** 2 / 500], 1)
    M2 = np.stack([np.linalg.lstsq(A2[k], ref[j[k], 0] - xc, rcond=None)[0], np.linalg.lstsq(A2[k], ref[j[k], 1] - yc, rcond=None)[0]])
    r2 = np.stack([xc + A2[k] @ M2[0], yc + A2[k] @ M2[1]], 1) - ref[j[k], :2]
    return M, k.sum(), m.sum(), np.sqrt((r ** 2).mean(0)), np.sqrt((r2 ** 2).mean(0)), Hs.max(), med
for e in (0, 1, 2):
    bynum = {cc["ccdnum"]: cc for cc, _ in res[e]}
    for cc, cat in res[e][:61:4]:
        ny, nx = cc["shape"]
        T = _terms((cat[:, 0] - cc["X0"]) / cc["SX"], (cat[:, 1] - cc["Y0"]) / cc["SX"])
        px = T @ cc["cpx"]; py = T @ cc["cpy"]
        w = bynum.get(good[cc["ext"]])
        T = _terms((px - nx / 2) / (nx / 2), (py - ny / 2) / (ny / 2))
        X = T @ w["cX"]; Y = T @ w["cY"]
        f = fit(X, Y, cat[:, 2])
        if f is None: print(e, cc["ext"], "FAIL"); continue
        M, n, nt, r, r2, pk, med = f
        print(e, cc["ext"], "shift %.2f %.2f lin %.5f %.5f %.5f %.5f" % (M[0, 0], M[1, 0], M[0, 1] - 1, M[0, 2], M[1, 1], M[1, 2] - 1), "n", n, "of", nt, "rms", r.round(3), "quad", r2.round(3), "peak", pk, med)
