"""Dense pure-noise copy of a scenario: each exposure -> K copies at +j*DT min, every image HDU replaced by fresh
Gaussian noise N(0, 1/sqrt(w)) (independent per copy), weight/mask symlinked, observer advanced linearly."""
import csv, io, json, math, os, sys
import numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen058/workspace/submission/linker")
import fastfits
src, dst, K = sys.argv[1], sys.argv[2], int(sys.argv[3]); DT = 6.0 / 1440

def hdu_bytes(h, k, raw):
    fi = h.fileinfo(k); return raw[fi["hdrLoc"]: fi["datLoc"] + fi["datSpan"]]

def job(a):
    i, j, row, newimg = a
    import astropy; astropy.log.setLevel("ERROR")
    from astropy.io import fits
    rng = np.random.default_rng(1000 * i + j)
    raw = open(os.path.join(src, row["image"]), "rb").read()
    hb = fits.open(io.BytesIO(raw), disable_image_compression=True)
    hc = fits.open(io.BytesIO(raw))
    hw = fits.open(os.path.join(src, row["weight"]), disable_image_compression=True)
    wext = {hw[k].header.get("EXTNAME", k): k for k in range(1, len(hw))}
    parts = [hdu_bytes(hb, 0, raw)]
    for k in range(1, len(hb)):
        h = hc[k].header
        if "CTYPE1" not in h:
            parts.append(hdu_bytes(hb, k, raw)); continue
        nx, ny = int(h["NAXIS1"]), int(h["NAXIS2"])
        wt = fastfits.read_tiled(hw[wext.get(h.get("EXTNAME"), k)])
        wt = np.nan_to_num(np.asarray(wt, np.float32))
        sd = np.where(wt > 0, 1 / np.sqrt(np.maximum(wt, 1e-30)), 0).astype(np.float32)
        img = (rng.standard_normal((ny, nx), dtype=np.float32) * sd)
        comp = fits.CompImageHDU(img, header=h, compression_type="RICE_1", tile_shape=(1, nx),
                                 quantize_level=16, quantize_method=-1)
        bio = io.BytesIO(); fits.HDUList([fits.PrimaryHDU(), comp]).writeto(bio); rb = bio.getvalue()
        hh = fits.open(io.BytesIO(rb), disable_image_compression=True)
        parts.append(hdu_bytes(hh, 1, rb)); hh.close()
    with open(os.path.join(dst, newimg), "wb") as f:
        for p in parts: f.write(p)
    return newimg

def main():
    rows = list(csv.DictReader(open(os.path.join(src, "exposures.csv"))))
    t = np.array([float(r["mjd_mid"]) for r in rows]); o = np.array([[float(r[k]) for k in ("obs_x", "obs_y", "obs_z")] for r in rows])
    A = np.c_[np.ones_like(t), t - t.mean(), (t - t.mean()) ** 2]
    coef = np.linalg.lstsq(A, o, rcond=None)[0]
    vel = lambda tt: coef[1] + 2 * coef[2] * (tt - t.mean())
    os.makedirs(os.path.join(dst, "exposures"), exist_ok=True)
    out, tasks = [], []
    for i, r in enumerate(rows):
        for j in range(K):
            d = dict(r); dt = j * DT
            d["exp_id"] = f"{r['exp_id']}{j:02d}"
            d["image"] = r["image"].replace(".fits.fz", f"_c{j:02d}.fits.fz")
            for key in ("mjd_start", "mjd_mid"): d[key] = f"{float(r[key]) + dt:.8f}"
            ob = o[i] + vel(t[i]) * dt
            d["obs_x"], d["obs_y"], d["obs_z"] = (f"{x:.9f}" for x in ob)
            for key in ("weight", "mask"):
                p = os.path.join(dst, r[key])
                if not os.path.exists(p): os.symlink(os.path.abspath(os.path.join(src, r[key])), p)
            out.append(d); tasks.append((i, j, r, d["image"]))
    with open(os.path.join(dst, "exposures.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(out)
    m = json.load(open(os.path.join(src, "meta.json"))); m["n_exposures"] = len(out)
    m["last_mjd"] = max(float(d["mjd_mid"]) for d in out); json.dump(m, open(os.path.join(dst, "meta.json"), "w"))
    from multiprocessing import Pool
    with Pool(4) as p:
        for n in p.imap_unordered(job, tasks): print(n, flush=True)

if __name__ == '__main__':
    main()
