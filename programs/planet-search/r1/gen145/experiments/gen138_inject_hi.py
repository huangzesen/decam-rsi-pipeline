"""Injection of synthetic TNOs into a copy of a dev scenario.

python tools/inject.py SRC_SCEN DST_SCEN [--n 40] [--seed 1] [--procs 4]

Bodies: heliocentric states at the mid time (r 30-60 au, near-circular prograde, inclination up to ~20 deg),
placed inside the field at the mid time.  Per-exposure pixel positions from two-body + light time
(submission/linker/orbit.py predict_dirs) and each CCD's TPV WCS.  Gaussian PSF (CCD FWHM header), flux set per
exposure so that S/N = F*sqrt(sum(w*psf^2)) equals the body's target S/N (w = weight map).
Classes: A (S/N 7-12, weight-1 region), B (S/N 5/sqrt(n_per_night)..4.8, weight-2 region), C (1.5-3, fainter).
Only CCD HDUs that receive a source are re-encoded (RICE_1, row tiles, NO_DITHER, q=16); all other HDUs are copied
byte for byte.  Weight and mask files are symlinked.  Truth -> DST/truth.json.
"""
import argparse, csv, io, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "$RSI_SANDBOX/r1/gen138/workspace/submission/linker")
import orbit  # noqa: E402
import fastfits  # noqa: E402

OBL = math.radians(23.4392911)
R_EQ_ECL = np.array([[1, 0, 0], [0, math.cos(OBL), -math.sin(OBL)], [0, math.sin(OBL), math.cos(OBL)]])  # ecl->eq


def night_of(mjd):
    return np.floor(np.asarray(mjd) + 0.3).astype(int)


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def make_bodies(rows, meta, n, rng, ccd_boxes):
    t = np.array([float(r["mjd_mid"]) for r in rows])
    obs = np.array([[float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])] for r in rows])
    tmid = 0.5 * (t.min() + t.max())
    im = int(np.argmin(np.abs(t - tmid)))
    o = obs[im] + (obs[min(im + 1, len(t) - 1)] - obs[im]) * 0  # observer near mid
    ra0, de0 = math.radians(meta["ra"]), math.radians(meta["dec"])
    nA = nB = int(os.environ.get("INJ_NAB", round(n * 15 / 40)))
    classes = ["A"] * nA + ["B"] * nB + ["C"] * (n - nA - nB)
    nights = night_of(t)
    nper = np.median(np.unique(nights, return_counts=True)[1])
    bodies = []
    for cls in classes:
        for _try in range(200):
            # random direction within 0.9 deg of centre, on a CCD at the mid time (approx: any CCD box)
            rr = 0.9 * math.sqrt(rng.uniform()) ; th = rng.uniform(0, 2 * math.pi)
            ra = ra0 + math.radians(rr * math.cos(th)) / math.cos(de0)
            de = de0 + math.radians(rr * math.sin(th))
            u = orbit.radec_to_unit(np.array(ra), np.array(de))
            R = rng.uniform(float(os.environ.get('INJ_RLO', '30')), float(os.environ.get('INJ_RHI', '60')))
            # |o + rho u| = R
            b = o @ u; c = o @ o - R * R
            rho = -b + math.sqrt(b * b - c)
            x = o + rho * u
            xe = R_EQ_ECL.T @ x
            rh = xe / np.linalg.norm(xe)
            east = np.cross([0, 0, 1.0], rh); east /= np.linalg.norm(east)
            north = np.cross(rh, east)
            a = math.radians(rng.uniform(float(os.environ.get("INJ_ALO", "-20")), float(os.environ.get("INJ_AHI", "20")))) * (1 if rng.uniform() < 0.5 else -1)
            vdir = math.cos(a) * east + math.sin(a) * north
            vc = math.sqrt(orbit.GM / R) * rng.uniform(float(os.environ.get("INJ_VLO", "0.92")), float(os.environ.get("INJ_VHI", "1.05")))
            ve = vc * vdir + rh * vc * rng.uniform(-0.05, 0.05)
            v = R_EQ_ECL @ ve
            st = np.r_[x, v]
            if cls == "A":
                snr = rng.uniform(7, 12)
            elif cls == "B":
                snr = rng.uniform(min(5 / math.sqrt(max(nper, 1)) + 0.05, 3.5), 4.8)
            else:
                snr = rng.uniform(1.5, 3.0)
            snr *= float(os.environ.get("INJ_SNR_SCALE", "1"))   # gen097: 1/0.7 puts the MEASURED S/N at the class targets
            bodies.append(dict(cls=cls, state=st.tolist(), epoch_mjd=float(tmid), snr=float(snr), r_au=R))
            break
    return bodies


def body_radec(bodies, rows):
    t = np.array([float(r["mjd_mid"]) for r in rows])
    obs = np.array([[float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])] for r in rows])
    out = []
    for b in bodies:
        d = orbit.predict_dirs(b["state"], b["epoch_mjd"], t, obs)
        ra = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
        de = np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
        out.append((ra, de))
    return out


def hdu_bytes(hdul_raw, k, raw):
    fi = hdul_raw.fileinfo(k)
    return raw[fi["hdrLoc"]: fi["datLoc"] + fi["datSpan"]]


INJ_COL = {"g": 0.50, "r": 1.0, "i": 1.32, "z": 1.5, "Y": 1.6, "VR": 0.85}


def process_exposure(args):
    ei, src, dst, row, bodies, radec = args
    import astropy
    astropy.log.setLevel("ERROR")
    from astropy.io import fits
    from astropy.wcs import WCS
    ipath = os.path.join(src, row["image"]); wpath = os.path.join(src, row["weight"])
    raw = open(ipath, "rb").read()
    hb = fits.open(io.BytesIO(raw), disable_image_compression=True)
    hc = fits.open(io.BytesIO(raw))
    hw = fits.open(wpath, disable_image_compression=True)
    wext = {hw[k].header.get("EXTNAME", k): k for k in range(1, len(hw))}
    ra = np.array([r[0][ei] for r in radec]); de = np.array([r[1][ei] for r in radec])
    recs = []
    parts = [hdu_bytes(hb, 0, raw)]
    for k in range(1, len(hb)):
        h = hc[k].header
        newb = None
        if "CTYPE1" in h:
            w = WCS(h)
            nx, ny = int(h["NAXIS1"]), int(h["NAXIS2"])
            px, py = w.all_world2pix(ra, de, 0)
            inside = np.where((px > 5) & (px < nx - 6) & (py > 5) & (py < ny - 6))[0]
            if len(inside):
                img = fastfits.read_tiled(hb[k])
                if img is None:
                    img = hc[k].data.astype(np.float32)
                img = np.array(img, np.float32)
                kw = wext.get(h.get("EXTNAME"), k)
                wt = fastfits.read_tiled(hw[kw])
                if wt is None:
                    wt = fits.open(wpath)[kw].data
                wt = np.nan_to_num(np.asarray(wt, np.float32))
                fwhm = float(h.get("FWHM") or 4.0)
                sig = fwhm / 2.3548
                hr = int(math.ceil(4 * sig))
                for j in inside:
                    x0, y0 = px[j], py[j]
                    ix, iy = int(round(x0)), int(round(y0))
                    xs = slice(max(ix - hr, 0), min(ix + hr + 1, nx)); ys = slice(max(iy - hr, 0), min(iy + hr + 1, ny))
                    yy, xx = np.mgrid[ys, xs]
                    psf = np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sig ** 2)
                    psf /= 2 * math.pi * sig * sig
                    q = float(np.sum(wt[ys, xs] * psf ** 2))
                    snr = bodies[j]["snr"]
                    if os.environ.get("INJ_COLOR", "0") == "1":   # gen124: red TNO colours, mean over exposures kept
                        snr *= INJ_COL.get(row["band"].strip(), 1.0) / bodies[j]["cn"]
                    F = snr / math.sqrt(q) if q > 0 else 0.0
                    if F > 0:
                        img[ys, xs] += (F * psf).astype(np.float32)
                    recs.append(dict(body=int(j), exp=ei, ccd=h.get("EXTNAME"), x=float(x0), y=float(y0), flux=F,
                                     snr=snr if F > 0 else 0.0, fwhm=fwhm,
                                     skysig=float(h.get("SKYSIGMA") or 0), wmed=float(np.median(wt[ys, xs]))))
                comp = fits.CompImageHDU(img, header=h, compression_type="RICE_1", tile_shape=(1, nx),
                                         quantize_level=16, quantize_method=-1)
                bio = io.BytesIO()
                fits.HDUList([fits.PrimaryHDU(), comp]).writeto(bio)
                rb = bio.getvalue()
                hh = fits.open(io.BytesIO(rb), disable_image_compression=True)
                newb = hdu_bytes(hh, 1, rb)
                hh.close()
        parts.append(newb if newb is not None else hdu_bytes(hb, k, raw))
    out = os.path.join(dst, row["image"])
    with open(out, "wb") as f:
        for p in parts:
            f.write(p)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    rows = read_rows(os.path.join(a.src, "exposures.csv"))
    meta = json.load(open(os.path.join(a.src, "meta.json")))
    os.makedirs(os.path.join(a.dst, "exposures"), exist_ok=True)
    for fn in ("exposures.csv", "meta.json"):
        open(os.path.join(a.dst, fn), "wb").write(open(os.path.join(a.src, fn), "rb").read())
    for r in rows:
        for key in ("weight", "mask"):
            d = os.path.join(a.dst, r[key])
            if not os.path.exists(d):
                os.symlink(os.path.abspath(os.path.join(a.src, r[key])), d)
    rng = np.random.default_rng(a.seed)
    bodies = make_bodies(rows, meta, a.n, rng, None)
    radec = body_radec(bodies, rows)
    cn = float(np.mean([INJ_COL.get(r["band"].strip(), 1.0) for r in rows if (r.get("magzero") or "").strip()]))
    for b in bodies:
        b["cn"] = cn
    tasks = []
    for i, r in enumerate(rows):
        if (r.get("magzero") or "").strip() == "":   # gen105: bad-WCS / uncalibrated exposures get no bodies
            d = os.path.join(a.dst, r["image"])
            if not os.path.exists(d):
                os.symlink(os.path.abspath(os.path.join(a.src, r["image"])), d)
            continue
        tasks.append((i, a.src, a.dst, r, bodies, radec))
    from multiprocessing import Pool
    with Pool(a.procs) as pool:
        allrecs = sum(pool.map(process_exposure, tasks), [])
    t = np.array([float(r["mjd_mid"]) for r in rows])
    nights = night_of(t)
    for j, b in enumerate(bodies):
        ex = sorted({r["exp"] for r in allrecs if r["body"] == j})
        b["exps"] = ex
        b["nights"] = sorted({int(nights[e]) for e in ex})
        b["ra"] = radec[j][0].tolist(); b["dec"] = radec[j][1].tolist()
    json.dump(dict(bodies=bodies, injections=allrecs, src=a.src), open(os.path.join(a.dst, "truth.json"), "w"))
    cnt = {}
    for b in bodies:
        cnt.setdefault(b["cls"], []).append(len(b["nights"]))
    print("bodies per class, nights covered:", {k: v for k, v in cnt.items()})
    print("injections:", len(allrecs))


if __name__ == "__main__":
    main()
