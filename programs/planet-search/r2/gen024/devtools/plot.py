import sys, pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
res = pickle.load(open(sys.argv[1], "rb")); e = int(sys.argv[2]); oth = [int(x) for x in sys.argv[3].split(",")]; lo, hi = float(sys.argv[4]), float(sys.argv[5])
cat, st = res[e]; st = st.astype(np.float32)
m = np.zeros(len(cat), bool)
for o in oth:
    d, _ = cKDTree(res[o][0][:, :2]).query(cat[:, :2], distance_upper_bound=0.8); m |= np.isfinite(d)
sel = (cat[:, 2] >= lo) & (cat[:, 2] < hi)
print("n in range", sel.sum(), "matched", (sel & m).sum(), "unmatched", (sel & ~m).sum())
rng = np.random.default_rng(0)
for name, mm in (("unmatched", sel & ~m), ("matched", sel & m)):
    idx = np.nonzero(mm)[0]; idx = rng.permutation(idx)[:100]
    fig, ax = plt.subplots(10, 10, figsize=(10, 10))
    for a, i in zip(ax.ravel(), idx):
        s = st[i].copy(); s[s < -50] = np.nan
        a.imshow(s, vmin=-2, vmax=max(6, np.nanmax(s) * 0.8), cmap="gray"); a.set_title(f"{cat[i,2]:.0f}", fontsize=6, pad=0)
    for a in ax.ravel(): a.axis("off")
    fig.subplots_adjust(0, 0, 1, 0.98, 0.05, 0.12); fig.savefig(f"{sys.argv[6]}_{name}.png", dpi=70); plt.close(fig)
