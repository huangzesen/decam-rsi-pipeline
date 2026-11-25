"""Shared by slice_run.py and slice_score.py: where a candidate's orbit lies on the sky, and forced photometry."""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orbits import rv_to_elements, elements_to_rv, GM, C_AU

PIX = 0.263                 # arcsec per DECam pixel
MAX_PER_WINDOW = 200        # candidates read per window, as the task grader (--max-per-scen)
RMIN_AU = 25.0              # candidates inside this distance are ignored, as the task grader


def night_of(mjd):
    return int(math.floor(mjd - 0.5))


def positions(c, mjds, obs):
    """RA, Dec (deg) of candidate c at each exposure: two-body about the Sun with light time (rsi/harness/grade.py)."""
    r0, v0 = np.array(c["state"][:3]), np.array(c["state"][3:])
    a, e, i, Om, om, M0 = rv_to_elements(r0[None], v0[None])
    if not (e[0] < 1 and a[0] > 0):
        return None
    n = math.sqrt(GM / a[0] ** 3)
    ra, de = [], []
    for t, o in zip(mjds, obs):
        tau = 0.0
        for _ in range(3):
            r, _ = elements_to_rv(a, e, i, Om, om, M0 + n * (t - tau - c["epoch"]))
            g = r[0] - o
            tau = np.linalg.norm(g) / C_AU
        ra.append(math.degrees(math.atan2(g[1], g[0])) % 360); de.append(math.degrees(math.asin(g[2] / np.linalg.norm(g))))
    return np.array(ra), np.array(de)


def sep(ra1, de1, ra2, de2):
    """Angular separation in arcsec."""
    d1, d2 = np.radians(de1), np.radians(de2)
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(np.radians(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600


def stack(f, e):
    """S/N of the inverse-variance stack of fluxes f with errors e."""
    f, e = np.array(f, float), np.array(e, float)
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    if not ok.any():
        return 0.0
    w = 1 / e[ok] ** 2
    return float(np.sum(f[ok] * w) / np.sqrt(np.sum(w)))


def load_cands(f):
    """A finder's output as a list sorted by falling p, each with its rank in the window (1 = best) and distance r."""
    out = []
    try:
        d = json.load(open(f))
        for c in (d.get("candidates", []) if isinstance(d, dict) else d):
            s = [float(x) for x in c["state"]]
            if len(s) == 6 and all(map(math.isfinite, s)) and math.isfinite(float(c["epoch_mjd"])):
                out.append(dict(epoch=float(c["epoch_mjd"]), state=s, p=float(c.get("p", 0.0)), mag=c.get("mag"),
                                r=float(np.linalg.norm(s[:3]))))
    except Exception:
        pass
    out = sorted(out, key=lambda c: -c["p"])[:MAX_PER_WINDOW]
    for k, c in enumerate(out):
        c["rank"] = k + 1
    return out


def read_exposures(csv_path):
    """exposures.csv of a window -> stamps, mid times, observer positions, night numbers."""
    import csv
    E = list(csv.DictReader(open(csv_path)))
    stamps = [os.path.basename(e["image"])[:17] for e in E]
    mjds = [float(e["mjd_mid"]) for e in E]
    obs = [np.array([float(e["obs_x"]), float(e["obs_y"]), float(e["obs_z"])]) for e in E]
    return stamps, mjds, obs, [night_of(m) for m in mjds]


class Exposure:
    """One exposure's image, weight map and mask. Forced photometry at a sky position, as rsi/bin/make_truth.py:
    aperture 0.7 FWHM (at least 1.5 px), sky from the median of a 3-5 FWHM annulus. Only the rows needed are read
    when astropy can read a section of a compressed image; otherwise one CCD at a time is held in memory."""

    def __init__(self, scen, row):
        from astropy.io import fits
        from astropy.wcs import WCS
        self.h = [fits.open(os.path.join(scen, "exposures", os.path.basename(row[k])), memmap=False) for k in ("image", "weight", "mask")]
        self.ccd = []
        for k, x in enumerate(self.h[0]):
            if k and "CTYPE1" in x.header:
                w = WCS(x.header)
                nx, ny = x.header.get("NAXIS1", 2046), x.header.get("NAXIS2", 4094)
                ra0, de0 = w.all_pix2world(nx / 2, ny / 2, 0)
                self.ccd.append((k, w, nx, ny, x.header.get("FWHM") or 4.0, float(ra0), float(de0)))
        self.section = True
        self.held = None                     # (hdu number, image, weight, mask) when sections cannot be read

    def close(self):
        self.held = None
        for h in self.h:
            h.close()

    def locate(self, ra, dec):
        for k, w, nx, ny, fwhm, ra0, de0 in self.ccd:
            if abs(dec - de0) > 0.2 or abs(((ra - ra0 + 180) % 360 - 180) * math.cos(math.radians(dec))) > 0.2:
                continue
            px, py = w.all_world2pix(ra, dec, 0)
            if 0 <= px < nx and 0 <= py < ny:
                return k, float(px), float(py), nx, ny, fwhm
        return None

    def cut(self, k, y0, y1, x0, x1):
        if self.section:
            try:
                return [np.asarray(h[k].section[y0:y1, x0:x1]) for h in self.h]
            except Exception:
                self.section = False
        if self.held is None or self.held[0] != k:
            self.held = None
            self.held = (k,) + tuple(np.array(h[k].data) for h in self.h)
            for h in self.h:
                del h[k].data                 # drop astropy's own copy: one CCD of an exposure in memory at a time
        return [a[y0:y1, x0:x1] for a in self.held[1:]]

    def measure(self, ra, dec, loc=None):
        """{} when the position is on no CCD; else hdu, x, y and flag ('' good, 'bad' masked pixel in the aperture,
        'edge' too little sky), with flux, err, snr when measured."""
        loc = loc or self.locate(ra, dec)
        if loc is None:
            return {}
        k, px, py, nx, ny, fwhm = loc
        ra_, r1, r2 = max(1.5, 0.7 * fwhm), 3 * fwhm, 5 * fwhm
        x0, x1 = int(max(0, np.floor(px - r2))), int(min(nx, np.ceil(px + r2) + 1))
        y0, y1 = int(max(0, np.floor(py - r2))), int(min(ny, np.ceil(py + r2) + 1))
        I, Wt, M = self.cut(k, y0, y1, x0, x1)
        I, Wt = I.astype(np.float32), Wt.astype(np.float32)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - px, yy - py)
        ap, an = d <= ra_, (d >= r1) & (d <= r2) & (M == 0) & (Wt > 0)
        rec = dict(hdu=k, x=round(px, 2), y=round(py, 2))
        if ap.sum() == 0 or an.sum() < 20:
            rec.update(flag="edge")
            return rec
        sky = np.median(I[an])
        f = float(np.sum(I[ap] - sky))
        var = float(np.sum(1.0 / np.where(Wt[ap] > 0, Wt[ap], np.inf))) + ap.sum() ** 2 * np.pi / 2 * np.var(I[an]) / an.sum()
        err = float(np.sqrt(var)) if var > 0 else None
        rec.update(flux=f, err=err, flag="bad" if ((M[ap] != 0) | (Wt[ap] <= 0)).any() else "", snr=(f / err) if err else None)
        return rec


def measure_many(exps, points):
    """points: [(exposure index, ra, dec)]. Returns one record per point, measured exposure by exposure and CCD by CCD."""
    out = [None] * len(points)
    by = {}
    for j, (k, ra, dec) in enumerate(points):
        by.setdefault(k, []).append(j)
    for k, js in by.items():
        locs = [(exps[k].locate(points[j][1], points[j][2]), j) for j in js]
        for loc, j in sorted(locs, key=lambda x: (x[0] is None, x[0][0] if x[0] else 0)):
            out[j] = exps[k].measure(points[j][1], points[j][2], loc) if loc else {}
        exps[k].held = None
    return out


def good(m):
    return bool(m) and m.get("flag") == "" and bool(m.get("err"))
