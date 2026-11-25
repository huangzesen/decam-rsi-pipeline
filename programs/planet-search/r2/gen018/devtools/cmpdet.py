import sys, time, csv, os, math, numpy as np, torch
sys.path.insert(0, sys.argv[1])
import finder
scen = sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
r = rows[int(sys.argv[3]) if len(sys.argv) > 3 else 0]
ra0 = math.radians(float(r["ra_center"])); dec0 = math.radians(float(r["dec_center"]))
ez = np.array([math.cos(dec0)*math.cos(ra0), math.cos(dec0)*math.sin(ra0), math.sin(dec0)]); ex = np.array([-math.sin(ra0), math.cos(ra0), 0.0]); ey = np.cross(ez, ex)
hd = [3, 20, 41]
task = (os.path.join(scen, r["image"]), os.path.join(scen, r["mask"]), hd, 3.0, 1.2, np.stack([ex, ey, ez]), None)
t = time.time(); cat, pieces = finder.detect_chunk(task); t_cpu = time.time() - t
dev = "cuda" if torch.cuda.is_available() else "mps"
t = time.time(); ch = finder.load_chunk(task); t_load = time.time() - t
finder.gpu_ccd(torch, dev, ch[0], 3.0)
t = time.time(); out = [finder.gpu_ccd(torch, dev, c, 3.0) for c in ch]; t_gpu = time.time() - t
print(f"cpu {t_cpu:.2f}s  load {t_load:.2f}s  gpu {t_gpu:.2f}s for {len(hd)} CCDs")
cat2 = np.concatenate([o[0] for o in out])
print("n", len(cat), len(cat2))
from scipy.spatial import cKDTree
for lo in (3, 4, 5, 8):
    a = cat[cat[:, 2] >= lo]; b = cat2[cat2[:, 2] >= lo - 0.05]
    dd, ii = cKDTree(b[:, :2]).query(a[:, :2])
    m = dd < 0.05
    print(f"snr>={lo}: cpu {len(a)} matched<0.05\" {m.sum()}  max dpos(matched) {dd[m].max():.4f}  median |dsnr| {np.median(np.abs(a[m,2]-b[ii[m],2])):.4f} max {np.abs(a[m,2]-b[ii[m],2]).max():.3f} flux rel {np.max(np.abs(a[m,3]/b[ii[m],3]-1)):.4f}")
for (y0, x0, m), o in zip(pieces, out):
    y1, x1, m2 = o[1]
    print("piece", (y0, x0, m.shape), (y1, x1, m2.shape), "nodata", (m == -128).sum(), (m2 == -128).sum(), "nodata mismatch", ((m == -128) != (m2 == -128)).sum(),
          "diff>1:", (np.abs(m.astype(int) - m2.astype(int)) > 1).sum(), "diff>3:", (np.abs(m.astype(int) - m2.astype(int)) > 3).sum())
