"""Exposure table for the archive estimate: NSC DR2 exposures (fetch_exposures.py output) with mid-time, night,
heliocentric observer position (au, equatorial), galactic latitude and coverage (number of exposures of the same
instrument with centres within 1 deg). Writes OUT.npz.

Usage: python -I footprint/prep.py NSCDIR OUT.npz
"""
import csv, os, sys
import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree


def main():
    src, out = sys.argv[1], sys.argv[2]
    rows = [r for f in sorted(os.listdir(os.path.join(src, "slices"))) if f.endswith(".csv")
            for r in csv.DictReader(open(os.path.join(src, "slices", f)))]
    col = lambda k, t=float: np.array([t(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]) if t is float else np.array([r[k] for r in rows])
    ra, dec, mjd, texp = col("ra"), col("dec"), col("mjd"), col("exptime")
    inst, filt, name = col("instrument", str), col("filter", str), col("exposure", str)
    nmeas, depth, fwhm = col("nmeas"), col("depth95"), col("fwhm")
    tmid = mjd + texp / 2 / 86400
    t = Time(tmid, format="mjd", scale="utc")
    xe = (get_body_barycentric("earth", t).xyz - get_body_barycentric("sun", t).xyz).to(u.au).value.T
    b = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic.b.deg
    uv = np.c_[np.cos(np.radians(dec)) * np.cos(np.radians(ra)), np.cos(np.radians(dec)) * np.sin(np.radians(ra)), np.sin(np.radians(dec))]
    cover = np.zeros(len(ra), int)
    for i in np.unique(inst):
        m = np.flatnonzero(inst == i)
        tr = cKDTree(uv[m])
        cover[m] = tr.query_ball_point(uv[m], 2 * np.sin(np.radians(1.0) / 2), return_length=True)
    np.savez_compressed(out, name=name, inst=inst, filt=filt, ra=ra, dec=dec, tmid=tmid, night=np.floor(tmid).astype(np.int64),
                        exptime=texp, nmeas=nmeas, depth95=depth, fwhm=fwhm, xe=xe, b=b, cover=cover)
    print(len(ra), "exposures ->", out)


if __name__ == "__main__":
    main()
