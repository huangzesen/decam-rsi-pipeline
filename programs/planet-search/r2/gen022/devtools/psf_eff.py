import sys, numpy as np, glob, os
from astropy.io import fits
from scipy import ndimage as ndi
S = "$DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev/"
def run(img, ccds):
    h = fits.open(img)
    hd = fits.open(img.replace("_ooi_", "_ood_"))
    for ci in ccds:
        d = h[ci].data.astype(np.float64); m = hd[ci].data != 0
        hdr = h[ci].header; fw = float(hdr["FWHM"]); s0 = fw / 2.3548
        med = np.median(d[::7, ::7]); sig = 1.4826 * np.median(np.abs(d[::7, ::7] - med))
        d = d - med
        sm = ndi.gaussian_filter(d, s0)
        mx = ndi.maximum_filter(sm, 15)
        H = 12
        ys, xs = np.nonzero((sm == mx) & (d > 150 * sig) & (d < 25000))
        effs = {k: [] for k in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.45, 'dg', 'mof')}
        fw_meas = []
        for y, x in zip(ys, xs):
            if y < H + 2 or x < H + 2 or y > d.shape[0] - H - 3 or x > d.shape[1] - H - 3: continue
            st = d[y - H:y + H + 1, x - H:x + H + 1]
            if m[y - H:y + H + 1, x - H:x + H + 1].any(): continue
            # local background from the border
            b = np.median(np.concatenate([st[0], st[-1], st[:, 0], st[:, -1]])); st = st - b
            yy, xx = np.mgrid[-H:H + 1, -H:H + 1]
            core = (np.hypot(yy, xx) <= 4)
            cy = (st * yy)[core].sum() / st[core].sum(); cx = (st * xx)[core].sum() / st[core].sum()
            if abs(cy) > 1 or abs(cx) > 1: continue
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            ap = np.hypot(yy - cy, xx - cx) <= 9
            # isolation: outer annulus flux small
            if np.abs(st[~ap]).max() > 0.03 * st.max(): continue
            P2 = (st[ap] ** 2).sum() - ap.sum() * sig ** 2
            def eff(K):
                return (K[ap] * st[ap]).sum() ** 2 / ((K[ap] ** 2).sum() * P2)
            for k in effs:
                if k == 'dg':
                    K = np.exp(-0.5 * r2 / s0 ** 2) + 0.12 * np.exp(-0.5 * r2 / (2.2 * s0) ** 2)
                elif k == 'mof':
                    al = fw / (2 * np.sqrt(2 ** (1 / 3.0) - 1)); K = (1 + r2 / al ** 2) ** -3.0
                else:
                    K = np.exp(-0.5 * r2 / (k * s0) ** 2)
                effs[k].append(eff(K))
            # measured FWHM from second moments of the core
            hm = st.max() / 2; fw_meas.append(2 * np.sqrt((st > hm).sum() / np.pi))
        print(os.path.basename(img)[:22], "ccd", ci, "hdrFWHM %.2f px" % fw, "meas ~%.2f" % np.median(fw_meas) if fw_meas else "", "n", len(fw_meas),
              " eff(S/N ratio)=", {k: round(float(np.sqrt(np.median(v))), 3) for k, v in effs.items() if v})
for a in sys.argv[1:]:
    run(a, [5, 20, 33, 50])
