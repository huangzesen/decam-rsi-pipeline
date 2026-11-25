import sys, csv, math, os, numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r2/gen016/workspace/submission")
import finder as F
GM = F.GM; ASEC = F.ASEC
scen = os.environ["RSI_DEV_DIR"] + "/scen02"
rows = list(csv.DictReader(open(scen + "/exposures.csv")))
t = np.array([float(r["mjd_mid"]) for r in rows]); t -= t.mean()
Oeq = np.array([[float(r[c]) for c in ("obs_x", "obs_y", "obs_z")] for r in rows])
ra0 = math.radians(float(rows[0]["ra_center"])); dec0 = math.radians(float(rows[0]["dec_center"]))
zc = np.array([math.cos(dec0) * math.cos(ra0), math.cos(dec0) * math.sin(ra0), math.sin(dec0)])
xc = np.cross([0, 0, 1.0], zc); xc /= np.linalg.norm(xc); yc = np.cross(zc, xc)
R = np.array([xc, yc, zc]); O = Oeq @ R.T
E = len(t); night = np.round(t).astype(int); nts = np.unique(night)
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
TOLS = float(os.environ.get("TOLS", 1.0)); MAXOFF = 60.0
res_all = []
for trial in range(40):
    r = float(np.clip(rng.normal(42, 5), 30, 60)); kecl = R @ np.array([0, -math.sin(F.EPS), math.cos(F.EPS)])
    u = np.array([rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), 1.0]); u /= np.linalg.norm(u)
    b = O[6] @ u; rho = -b + math.sqrt(b * b - O[6] @ O[6] + r * r); P = O[6] + rho * u; ph = P / r
    e1 = np.cross(kecl, ph); e1 /= np.linalg.norm(e1); e2 = np.cross(ph, e1); beta = math.radians(rng.normal(0, 10))
    vc = math.sqrt(GM / r); V = vc * math.sqrt(rng.uniform(0.8, 1.2)) * (math.cos(beta) * e1 + math.sin(beta) * e2) + rng.uniform(-0.2, 0.2) * vc * ph
    t0 = t[6] - rho / F.C_AU
    xy_true = F.predict_xy(P, V, t0, t, O)
    # supported: three consecutive nights; a partial fit to noisy positions there
    k0 = rng.integers(0, len(nts) - 2); supn = nts[k0:k0 + 3]; sup = np.isin(night, supn); uns = ~sup
    obs = xy_true + rng.normal(0, float(os.environ.get("NOISE", 0.3)), xy_true.shape)
    ds = np.nonzero(sup)[0]
    # start from a wrong distance, fit to the supported detections
    best = None
    for rg in (32.0, 38.0, 44.0, 52.0):
        sc_ = rg / r
        P1, V1, resid = F.fit_orbit(P * sc_, V * math.sqrt(1 / sc_), t0, t[ds], O[ds], obs[ds], np.full(len(ds), 0.3))
        xy1 = F.predict_xy(P1, V1, t0, t, O); rr = np.hypot(*(xy1[ds] - obs[ds]).T).max()
        if best is None or rr < best[0]: best = (rr, P1, V1, xy1)
    _, P1, V1, xy1 = best
    off0 = np.hypot(*(xy1 - xy_true).T)
    # --- the scan of extend_tracks (numpy replica)
    r0 = np.linalg.norm(P1); vc1 = math.sqrt(GM / r0)
    J = np.zeros((E, 2, 6))
    for a in range(6):
        P2 = P1.copy(); V2 = V1.copy()
        if a < 3: P2[a] += 1e-5 * r0; J[..., a] = (F.predict_xy(P2, V2, t0, t, O) - xy1) / 1e-5
        else: V2[a - 3] += 1e-3 * vc1; J[..., a] = (F.predict_xy(P2, V2, t0, t, O) - xy1) / 1e-3
    JS = (J * sup[:, None, None]).reshape(2 * E, 6)
    U, S, Vt = np.linalg.svd(JS, full_matrices=False)
    sig = S / math.sqrt(2 * sup.sum()); mot = np.linalg.norm(np.einsum("exp,kp->ekx", J, Vt), axis=2)
    mk = np.max(np.where(uns[:, None], mot, 0), axis=0)
    amp = np.minimum(TOLS / np.maximum(sig, 1e-9), MAXOFF / np.maximum(mk, 1e-9)); reach = amp * mk
    o2 = np.argsort(-reach)[:2]; k1, k2 = o2; re1 = reach[k1]; re2 = reach[k2] if reach[k2] >= 1.2 else 0.0
    h1, h2 = 75, 7
    st1 = re1 / h1 / mk[k1]; st2 = re2 / h2 / mk[k2] if re2 > 0 else 0.0
    g1, g2 = np.meshgrid(np.arange(-h1, h1 + 1), np.arange(-h2, h2 + 1), indexing="ij"); g1 = g1.ravel().astype(float); g2 = g2.ravel().astype(float)
    strong = sig >= 5.0; strong[k1] = False
    if re2 > 0: strong[k2] = False
    pinv = np.einsum("kp,k,zk->pz", Vt, np.where(strong, 1 / np.maximum(S, 1e-9), 0), U)
    dq = st1 * g1[:, None] * Vt[k1][None] + st2 * g2[:, None] * Vt[k2][None]
    lin = xy1[None] + np.einsum("exp,tp->tex", J, dq); T_ = len(dq)
    supf = np.repeat(sup, 2).astype(float)
    for it in range(3):
        Pq = P1[None] + r0 * dq[:, :3]; Vq = V1[None] + vc1 * dq[:, 3:]
        xyt = F.predict_xy_b(Pq, Vq, np.full(T_, t0), t, O)
        if it < 2:
            res = (xyt.reshape(T_, 2 * E) - lin.reshape(T_, 2 * E)) * supf[None]
            dq = dq - res @ pinv.T
    d = np.hypot(xyt[..., 0] - xy_true[None, :, 0], xyt[..., 1] - xy_true[None, :, 1])      # [T,E]
    dmax = d.max(axis=1); ib = int(np.argmin(dmax))
    res_all.append((off0[uns].max(), dmax[ib], d[ib][sup].max(), re1, re2, sig[k1], sig[k2], off0[sup].max()))
    print("r %.0f sup nights %s  start: far off %.1f\" sup off %.2f\" | reach %.0f\" / %.0f\" (sig %.2f %.2f) | best trial: max off %.2f\" (sup %.2f\")" % (r, (supn - nts[0]).tolist(), off0[uns].max(), off0[sup].max(), re1, re2, sig[k1], sig[k2], dmax[ib], d[ib][sup].max()))
a = np.array(res_all)
print("start far off median %.1f; recovered to < 1.0\": %d of %d; < 1.5\": %d" % (np.median(a[:, 0]), (a[:, 1] < 1.0).sum(), len(a), (a[:, 1] < 1.5).sum()))
