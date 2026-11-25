import sys, pickle, numpy as np
D = pickle.load(open(sys.argv[1] + "/cands.pkl", "rb"))
C = [c for c in D["cands"] if c.get("alt") is not None and c.get("xy") is not None and c.get("score") is not None and len(c["alt"]) == 4]
night, ixy, wS = D["night"], D["inj_xy"], D["wS"]; E = len(night)
print("cands", len(C), "E", E)
def hit(c):
    d = np.hypot(c["xy"][:, None, 0] - ixy[:, :, 0], c["xy"][:, None, 1] - ixy[:, :, 1])
    for b in range(d.shape[1]):
        if len(set(night[d[:, b] < 2.0])) >= 3: return b
    return -1
H = np.array([hit(c) for c in C]); bic = np.array([c["alt"][2] for c in C]); bil = np.array([c["alt"][3] for c in C])
two = np.array([bool(c.get("two")) for c in C]); deep = np.array([bool(c.get("deep")) for c in C])
d = bic - bil
print("score change bic - bil: injected-hit cands: mean %.2f median %.2f (n=%d) | others: mean %.2f median %.2f, top-200 others by bil: mean %.2f" % (
    d[H >= 0].mean(), np.median(d[H >= 0]), (H >= 0).sum(), d[H < 0].mean(), np.median(d[H < 0]), d[H < 0][np.argsort(-bil[H < 0])[:200]].mean()))
# crude de-dup of the non-injected list: one per (rounded first/last position)
def dedup(sc):
    o = np.argsort(-sc); seen = set(); out = []
    for i in o:
        if H[i] >= 0: continue
        k = tuple(np.round(C[i]["xy"][[0, -1]].ravel() / 3.0).astype(int))
        if k in seen: continue
        seen.add(k); out.append(sc[i])
        if len(out) >= 100: break
    return np.array(out)
nb, nl = dedup(bic), dedup(bil)
print("chance level (non-injected, de-duplicated) rank 5/15/25/40:  bil", np.round(nl[[4, 14, 24, 39]], 1), " bic", np.round(nb[[4, 14, 24, 39]], 1))
rows = []
for b in sorted(set(H[H >= 0])):
    m = H == b
    sb, sl = bic[m].max(), bil[m].max()
    rows.append((b, sl, sb, int((nl > sl).sum()), int((nb > sb).sum()), int(np.isfinite(C[np.nonzero(m)[0][np.argmax(bic[m])]]["sv"]).sum())))
print("body  score_bil score_bic  #chance above (bil -> bic)  n_exp")
for r in sorted(rows, key=lambda r: -r[1]):
    print("%4d  %7.1f  %7.1f   %3d -> %3d   %d" % r)
r = np.array(rows)
for K in (10, 15, 20, 25):
    print("bodies with < %d chance candidates above: bil %d  bic %d" % (K, (r[:, 3] < K).sum(), (r[:, 4] < K).sum()))
if D.get("inj_sv_bil") is not None:
    a, b_ = D["inj_sv"], D["inj_sv_bil"]; ok = np.isfinite(a) & np.isfinite(b_) & (b_ > 2.5)
    print("S/N at the true positions of injected bodies (bil > 2.5): mean bic/bil = %.3f, n = %d" % (np.mean(a[ok] / b_[ok]), ok.sum()))
