"""Real-object test: 2026 ED138's own DECam (W84) detections, 2016-2023, from MPEC 2026-U85, planted among synthetic
false detections, in the linker's input format.

Usage: python ed138.py MPEC.txt OUT.npz [--rho 500] [--visits 8] [--radius 3] [--seed 7] [--fit]
Observer: heliocentric CTIO (astropy built-in ephemeris + geodetic site), so the frame matches the heliocentric MPC
elements used as truth. Seasons: Feb 1 - Sep 30 of 2016-2019 (opposition near June 10). The real exposures are the
44 W84 epochs (ED138 placed within 0.8 deg of each exposure centre); `visits` adds random exposures per position per
season in which ED138 is absent (not reported). --fit also prints the straight-line residual of each season's
detections in the transformed frame at the best (gamma, gamma-dot).
"""
import argparse, json, re
import numpy as np
from astropy.time import Time
from astropy.utils import iers
from astropy.coordinates import get_body_barycentric, EarthLocation
import astropy.units as u
from orbits import ARCSEC, radec_to_unit, unit_to_radec, tangent_frame, GM

PAT = re.compile(r"K26ED8D\s+\S?C(\d{4}) (\d{2}) (\d{2}\.\d{6})(\d{2}) (\d{2}) (\d{2}\.\d{3})([-+]\d{2}) (\d{2}) (\d{2}\.\d{2})\s+"
                 r"(\d{2}\.\d{2})(\w)\w{3}\d{3}(W84|X05)")
# MPEC 2026-U85 orbit (heliocentric ecliptic J2000), epoch JD 2461200.5 TT
EL = dict(a=42.9486367, e=0.1417431, inc=24.47104, Om=322.50088, om=167.95217, M=128.33566, mjd=61200.0)
iers.conf.auto_download = False          # the bundled IERS-B table covers 2016-2023
CTIO = EarthLocation.from_geodetic(lon=-70.80652 * u.deg, lat=-30.16967 * u.deg, height=2207 * u.m)


def parse(path):
    rows = []
    for m in PAT.finditer(open(path).read()):
        y, mo, d, rh, rm, rs, dd, dm, ds, mag, band, site = m.groups()
        if site != "W84":
            continue
        day = float(d)
        t = Time(f"{y}-{mo}-{int(day):02d}", scale="utc").mjd + (day - int(day))
        ra = np.radians(15 * (int(rh) + int(rm) / 60 + float(rs) / 3600))
        sgn = -1 if dd.startswith("-") else 1
        dec = sgn * np.radians(abs(int(dd)) + int(dm) / 60 + float(ds) / 3600)
        rows.append((t, ra, dec, float(mag), band))
    return rows


def observer(mjd):
    t = Time(mjd, format="mjd", scale="utc")
    e = get_body_barycentric("earth", t).xyz.to(u.au).value.T
    s = get_body_barycentric("sun", t).xyz.to(u.au).value.T
    g = CTIO.get_gcrs(t).cartesian.xyz.to(u.au).value.T
    return e - s + g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mpec")
    ap.add_argument("out")
    ap.add_argument("--rho", type=float, default=500.0)
    ap.add_argument("--visits", type=float, default=8.0)
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = parse(a.mpec)
    t_obj = np.array([r[0] for r in rows])
    u_obj = radec_to_unit(np.array([r[1] for r in rows]), np.array([r[2] for r in rows]))
    seasons = [(Time(f"{y}-02-01").mjd, Time(f"{y}-09-30").mjd) for y in (2016, 2017, 2018, 2019)]
    s16 = (t_obj >= seasons[0][0]) & (t_obj <= seasons[0][1])
    c = u_obj[s16].mean(0); c /= np.linalg.norm(c)
    ra0, dec0 = unit_to_radec(c)
    F = tangent_frame(ra0, dec0)
    R = np.radians(a.radius)
    er = np.radians(1.1)
    # exposures: the real epochs, plus random ones
    et, ec, eobj = [], [], []
    for k in range(len(rows)):
        off = rng.normal(0, 1, 3); off -= off @ u_obj[k] * u_obj[k]
        uc = u_obj[k] + off / np.linalg.norm(off) * np.radians(0.8) * np.sqrt(rng.uniform())
        et.append(t_obj[k]); ec.append(uc / np.linalg.norm(uc)); eobj.append(k)
    # random exposures over a field that follows ED138's mean position in each season
    n_extra = int(round(a.visits * a.radius ** 2 / 1.1 ** 2))
    for s0, s1 in seasons:
        m = (t_obj >= s0) & (t_obj <= s1)
        cs = u_obj[m].mean(0) if m.any() else c
        Fs = tangent_frame(*unit_to_radec(cs / np.linalg.norm(cs)))
        for _ in range(n_extra):
            t = np.floor(rng.uniform(s0, s1)) + rng.uniform(0.05, 0.38)
            rr, th = R * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
            uc = Fs[2] + rr * np.cos(th) * Fs[0] + rr * np.sin(th) * Fs[1]
            et.append(t); ec.append(uc / np.linalg.norm(uc)); eobj.append(-1)
    et = np.array(et); ec = np.array(ec); eobj = np.array(eobj)
    o = np.argsort(et); et, ec, eobj = et[o], ec[o], eobj[o]
    obs = observer(et)
    # false detections
    nf = rng.poisson(a.rho * np.pi * 1.1 ** 2, len(et))
    fe = np.repeat(np.arange(len(et)), nf)
    rr = er * np.sqrt(rng.uniform(size=len(fe))); th = rng.uniform(0, 2 * np.pi, len(fe))
    ex = np.cross([0, 0, 1.0], ec); ex /= np.linalg.norm(ex, axis=1)[:, None]
    ey = np.cross(ec, ex)
    fu = ec[fe] + (rr * np.cos(th))[:, None] * ex[fe] + (rr * np.sin(th))[:, None] * ey[fe]
    fu /= np.linalg.norm(fu, axis=1)[:, None]
    oe = np.flatnonzero(eobj >= 0)
    det_u = np.concatenate([fu, u_obj[eobj[oe]]])
    det_exp = np.concatenate([fe, oe]).astype(np.int32)
    det_obj = np.concatenate([-np.ones(len(fe), np.int32), np.zeros(len(oe), np.int32)])
    o = np.lexsort((det_obj, det_exp))
    ra, dec = unit_to_radec(det_u[o])
    rad = np.radians
    np.savez_compressed(a.out, det_ra=ra, det_dec=dec, det_exp=det_exp[o], det_obj=det_obj[o], exp_t=et,
                        exp_ra=unit_to_radec(ec)[0], exp_dec=unit_to_radec(ec)[1], exp_obs=obs,
                        obj_a=np.array([EL["a"]]), obj_e=np.array([EL["e"]]), obj_inc=np.array([rad(EL["inc"])]),
                        obj_Om=np.array([rad(EL["Om"])]), obj_om=np.array([rad(EL["om"])]),
                        obj_M0=np.array([rad(EL["M"])]), obj_r0=np.array([0.0]), obj_vr=np.array([0.0]),
                        obj_t0=np.array([EL["mjd"]]))
    meta = dict(vars(a), ra0=float(ra0), dec0=float(dec0), seasons_mjd=seasons, n_exp=len(et), n_false=int(len(fe)),
                n_obj=1, n_obj_det=int(len(oe)), exp_radius_deg=1.1)
    json.dump(meta, open(a.out.replace(".npz", ".json"), "w"), indent=1)
    nights = {}
    for s, (s0, s1) in enumerate(seasons):
        m = (t_obj >= s0) & (t_obj <= s1)
        nights[s] = int(len(np.unique(np.floor(t_obj[m]))))
    print(json.dumps(dict(n_exp=len(et), n_false=int(len(fe)), ed138_det=int(len(oe)), nights_per_season=nights)))
    if a.fit:
        import link_cpu as L
        from scipy.optimize import minimize
        d, meta = L.load(a.out)
        for s in range(4):
            for span in (1, 4) if s == 0 else (1,):
                W = L.window(d, meta, s, 1.5, span)
                m = np.flatnonzero(d["det_obj"][W["sel"]] == 0)
                if len(np.unique(W["night"][m])) < 3:
                    continue
                Wm = dict(W, t=W["t"][m], obs=W["obs"][m], theta=W["theta"][m])
                f = lambda p: (L.fit_line(*L.transform(Wm, p[0] / 100, p[1] * 1e-6))[1] ** 2).sum()
                best = min((minimize(f, [100 / r, 0.0], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-22, maxiter=4000))
                            for r in (40, 46, 50)), key=lambda z: z.fun)
                res = L.fit_line(*L.transform(Wm, best.x[0] / 100, best.x[1] * 1e-6))[1]
                print(f"season {s} span {span}: {len(m)} det, {len(np.unique(W['night'][m]))} nights, best r = "
                      f"{100 / best.x[0]:.2f} au, gamma-dot = {best.x[1]:.3f}e-6/day, max resid {res.max() / ARCSEC:.3f}\", "
                      f"rms {np.sqrt((res ** 2).mean()) / ARCSEC:.3f}\"")


if __name__ == "__main__":
    main()
