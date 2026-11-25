import sys, pickle, math, json, numpy as np
GM = 2.959122082855911e-4
D = pickle.load(open(sys.argv[1], "rb"))
C = [c for c in D["cands"] if c["score"] is not None]
kecl = D["kecl"]
def feat(c):
    P, V = c["P"], c["V"]; r = np.linalg.norm(P); h = np.cross(P, V); hn = np.linalg.norm(h)
    inc = math.degrees(math.acos(np.clip(h @ kecl / hn, -1, 1))); w = hn * hn / (GM * r); f = (P @ V / r) / math.sqrt(GM / r)
    ecc = float(np.linalg.norm(np.cross(V, h) / GM - P / r))
    beta = math.degrees(math.asin(P @ kecl / r))
    return r, w, inc, f, ecc, beta
F = np.array([feat(c) for c in C]); S = np.array([c["score"] for c in C]); pos = np.array([c["pos"] for c in C])
inj = np.array([-1 if c["inj"] is None else c["inj"] for c in C]); fit = np.array([c["fit"] for c in C])
# identify injected by track too
ixy = D["inj_xy"]
if ixy is not None:
    XY = np.array([c["xy"] for c in C])                      # [M, E, 2]
    for b in range(ixy.shape[1]):
        d = np.hypot(XY[:, :, 0] - ixy[None, :, b, 0], XY[:, :, 1] - ixy[None, :, b, 1])
        m = (d < 2.0).sum(1) >= max(3, d.shape[1] // 2)
        inj[m & (inj < 0)] = b
print("cands", len(C), "fitted", fit.sum(), "injected-matching", (inj >= 0).sum(), "distinct inj", len(set(inj[inj >= 0])))
ch = inj < 0
print("score quantiles of chance cands:", np.round(np.quantile(S[ch], [0.5, 0.9, 0.99, 0.999]), 1), "top non-inj scores", np.round(np.sort(S[ch])[::-1][:12], 1))
for lo in (25, 28, 31, 34, 37, 40):
    print(f"  N(chance score>={lo}) = {(S[ch] >= lo).sum()}")
names = ["r", "w", "inc", "f", "ecc", "beta"]
for sel, nm in ((ch & (S > 28), "chance S>28"), (ch & (S > 28) & fit, "chance fitted"), (ch & (S > 28) & ~fit, "chance grid"), (inj >= 0, "injected")):
    print(nm, sel.sum())
    for k, n in enumerate(names):
        print(f"   {n:5s}", np.round(np.quantile(F[sel, k], [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]), 2))
if len(sys.argv) > 2:
    bodies = json.load(open(sys.argv[2]))
    print("injected: best cand per body: truth (r, w, beta_dir) vs recovered (r, w, inc, f), score, fit")
    for b in sorted(set(inj[inj >= 0])):
        m = np.nonzero(inj == b)[0]; q = m[np.argmax(S[m])]
        bd = bodies[b]; st = np.array(bd["state"]); P, V = st[:3], st[3:]
        h = np.cross(P, V); kq = np.array([0, -math.sin(math.radians(23.43928)), math.cos(math.radians(23.43928))])
        ti = math.degrees(math.acos(h @ kq / np.linalg.norm(h)))
        print(f"  {b:2d} mag {bd['mag']:.2f} truth r={bd['r']:.1f} w={bd['w']:.2f} i={ti:.1f} f={bd['f']:+.2f} | rec r={F[q,0]:.1f} w={F[q,1]:.2f} i={F[q,2]:.1f} f={F[q,3]:+.2f} e={F[q,4]:.2f} score={S[q]:.1f} pos={pos[q]:.1f} fit={fit[q]}")
np.save(sys.argv[1] + ".feat.npy", np.column_stack([F, S, pos, inj, fit]))
