"""Orbit of a linked candidate, attribution of its other detections, and the archive check.

1. Take the top linkage of a search (most nights, then lowest rms) from LINKS.npz (link_gpu.py --save-links; `sel`
   indexes DATA.npz) and fit a barycentric two-body orbit (Sun + planets' mass, light time, observer = CTIO) to its
   detections. The start is the linkage's own BK state; nothing from the MPC orbit is used.
2. Attribution: predict every other exposure of DATA.npz with the fit's covariance, attach the nearest detection inside
   max(--rmin, 4 sigma), refit, repeat until nothing changes. Labelled detections (det_obj = 0) are only reported.
3. Archive check (with --mpc OBS.json, the MPC get-obs reply): predict each MPC observation from the DECam-only orbit
   and compare (offset / predicted sigma), then fit DECam + MPC non-W84 observations together and print the elements.

Usage: python orbit_fit.py DATA.npz LINKS.npz [--season 3 --span 2] [--mpc OBS.json] [--json OUT.json]
"""
import argparse, json, sys, os
import numpy as np
from scipy.optimize import least_squares
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel, EarthLocation
import astropy.units as u
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import link_cpu as L
from orbits import GM, C_AU, ARCSEC, radec_to_unit, unit_to_radec, eq_to_ecl
from ed138 import CTIO

GMT = GM * 1.0013417            # Sun + planets: barycentric two-body for a distant body
SITES = {"W84": CTIO, "T14": EarthLocation.from_geodetic(-155.46806 * u.deg, 19.82525 * u.deg, 4204 * u.m),
         "645": EarthLocation.from_geodetic(-105.82028 * u.deg, 32.78028 * u.deg, 2788 * u.m),
         "568": EarthLocation.from_geodetic(-155.47166 * u.deg, 19.82600 * u.deg, 4139 * u.m)}


def observer_bary(mjd_utc, site):
    """Barycentric ICRS position (au) of the site, and TDB MJD."""
    t = Time(mjd_utc, format="mjd", scale="utc")
    e, _ = get_body_barycentric_posvel("earth", t)
    g = SITES[site].get_gcrs(t).cartesian.xyz.to(u.au).value.T
    return e.xyz.to(u.au).value.T + g, t.tdb.mjd


def stumpff(z):
    c, s = np.empty_like(z), np.empty_like(z)
    p, n, o = z > 1e-8, z < -1e-8, np.abs(z) <= 1e-8
    sp, sn = np.sqrt(z[p]), np.sqrt(-z[n])
    c[p] = (1 - np.cos(sp)) / z[p]; s[p] = (sp - np.sin(sp)) / sp ** 3
    c[n] = (np.cosh(sn) - 1) / -z[n]; s[n] = (np.sinh(sn) - sn) / sn ** 3
    c[o] = 0.5 - z[o] / 24; s[o] = 1 / 6 - z[o] / 120
    return c, s


def propagate(r0, v0, dt, mu=GMT):
    """Universal-variable two-body propagation of one state over an array of dt (days)."""
    r0n = np.linalg.norm(r0); vr0 = r0 @ v0 / r0n; alpha = 2 / r0n - v0 @ v0 / mu; sm = np.sqrt(mu)
    x = sm * abs(alpha) * dt
    for _ in range(50):
        z = alpha * x * x; c, s = stumpff(z)
        F = r0n * vr0 / sm * x * x * c + (1 - alpha * r0n) * x ** 3 * s + r0n * x - sm * dt
        dF = r0n * vr0 / sm * x * (1 - z * s) + (1 - alpha * r0n) * x * x * c + r0n
        dx = F / dF; x = x - dx
        if np.all(np.abs(dx) < 1e-12):
            break
    z = alpha * x * x; c, s = stumpff(z)
    f = 1 - x * x / r0n * c; g = dt - x ** 3 * s / sm
    return f[:, None] * r0 + g[:, None] * v0


def directions(p, t0, t, R):
    """Astrometric unit vectors from observers R (barycentric) at TDB times t, light time included."""
    tau = np.zeros(len(t))
    for _ in range(3):
        rho = propagate(p[:3], p[3:], t - tau - t0) - R
        tau = np.linalg.norm(rho, axis=1) / C_AU
    return rho / np.linalg.norm(rho, axis=1)[:, None]


def tangent(uo):
    ra, dec = unit_to_radec(uo)
    e = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], 1)
    n = np.stack([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)], 1)
    return e, n


def residuals(p, t0, t, R, uo, sig):
    up = directions(p, t0, t, R); e, n = tangent(uo)
    d = up - uo
    return np.concatenate([(d * e).sum(1), (d * n).sum(1)]) / ARCSEC / np.concatenate([sig, sig])


def fit(p0, t0, t, R, uo, sig):
    sc = np.array([1, 1, 1, 1e-3, 1e-3, 1e-3])
    res = least_squares(lambda q: residuals(q * sc, t0, t, R, uo, sig), p0 / sc, method="lm", xtol=1e-14, ftol=1e-14)
    p = res.x * sc
    J = res.jac / sc[None, :]
    chi2 = (res.fun ** 2).sum(); dof = max(1, len(res.fun) - 6)
    U_, s_, Vt_ = np.linalg.svd(res.jac, full_matrices=False)   # covfix 2026-11-23: invert in the scaled parameters (pinv of the unscaled J.T @ J loses digits)
    cov = (Vt_.T / s_ ** 2) @ Vt_ * np.outer(sc, sc) * max(1.0, chi2 / dof)
    return p, cov, res.fun.reshape(2, -1).T * sig[:, None], chi2, dof


def predict_sigma(p, cov, t0, t, R):
    """Predicted unit vectors and the 1-sigma semi-major axis (arcsec) of the position uncertainty at each time."""
    u0 = directions(p, t0, t, R); e, n = tangent(u0)
    J = np.zeros((len(t), 2, 6))
    for k in range(6):
        h = np.zeros(6); h[k] = 1e-7 * (1 if k < 3 else 1e-3) * max(1.0, abs(p[k]) if k < 3 else 1.0)
        d = (directions(p + h, t0, t, R) - u0) / ARCSEC
        J[:, 0, k] = (d * e).sum(1) / h[k]; J[:, 1, k] = (d * n).sum(1) / h[k]
    C = J @ cov @ J.transpose(0, 2, 1)
    return u0, np.sqrt(np.linalg.eigvalsh(C)[:, -1])


def elements(p, t0, mu=GMT, helio=False):
    r, v = p[:3].copy(), p[3:].copy()
    if helio:
        s, sv = get_body_barycentric_posvel("sun", Time(t0, format="mjd", scale="tdb"))
        r -= s.xyz.to(u.au).value; v -= sv.xyz.to(u.au / u.day).value; mu = GM
    r, v = eq_to_ecl(r), eq_to_ecl(v)
    h = np.cross(r, v); rn = np.linalg.norm(r)
    ev = np.cross(v, h) / mu - r / rn; e = np.linalg.norm(ev)
    a = 1 / (2 / rn - v @ v / mu)
    return dict(a=a, e=e, q=a * (1 - e), i=np.degrees(np.arccos(h[2] / np.linalg.norm(h))), r=rn,
                Om=np.degrees(np.arctan2(h[0], -h[1])) % 360)


def mpc_obs(path):
    out = []
    for blk in json.load(open(path)):
        for l in blk["OBS80"].splitlines():
            y, mo, dd = int(l[15:19]), int(l[20:22]), float(l[23:32])
            mjd = Time(f"{y}-{mo:02d}-{int(dd):02d}", scale="utc").mjd + (dd - int(dd))
            ra = 15 * (int(l[32:34]) + int(l[35:37]) / 60 + float(l[38:44]) / 3600)
            sg = -1 if l[44] == "-" else 1
            dec = sg * (int(l[45:47]) + int(l[48:50]) / 60 + float(l[51:56]) / 3600)
            out.append((mjd, np.radians(ra), np.radians(dec), l[77:80]))
    return out


def fmt(el):
    return f"a {el['a']:8.1f} au  e {el['e']:.4f}  q {el['q']:7.3f} au  i {el['i']:6.2f} deg  Om {el['Om']:6.2f}  r {el['r']:.3f} au"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("links")
    ap.add_argument("--season", type=int, default=3); ap.add_argument("--span", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=0.1, help="arcsec per coordinate, DECam (NSC) and MPC")
    ap.add_argument("--rmin", type=float, default=2.0, help="arcsec, smallest attribution radius")
    ap.add_argument("--mpc", default=None)
    ap.add_argument("--sites", default="", help="comma list of MPC sites to use from --mpc (default all but W84)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    d, meta = L.load(a.data)
    Lk = np.load(a.links)
    q = np.lexsort((Lk["rms"], -Lk["nights"]))[0]
    det = Lk["sel"][Lk["det"][Lk["off"][q]:Lk["off"][q + 1]]]
    W = L.window(d, meta, a.season, 1.5, a.span)
    out = dict(linkage=dict(nights=int(Lk["nights"][q]), rms=float(Lk["rms"][q]), r=float(1 / Lk["gamma"][q]),
                            det=det.tolist(), labelled=int((d["det_obj"][det] == 0).sum())))
    print(f"top linkage: {len(det)} detections on {Lk['nights'][q]} nights, r {1 / Lk['gamma'][q]:.2f} au, rms {Lk['rms'][q]:.3f}\"; "
          f"labelled OF201: {(d['det_obj'][det] == 0).sum()}")
    # observers for every exposure
    Rall, tall = observer_bary(d["exp_t"], "W84")
    uall = radec_to_unit(d["det_ra"], d["det_dec"])
    # start: BK state of the linkage (heliocentric, patch frame) -> barycentric equatorial
    g = Lk["gamma"][q]; ax, adx, ay, ady = Lk["state"][q]
    F = W["F"]
    x = np.array([ax, ay, 1.0]) / g @ F; v = np.array([adx, ady, 0.0]) / g @ F
    t0 = Time(W["t_ref"], format="mjd", scale="utc").tdb.mjd
    s, sv = get_body_barycentric_posvel("sun", Time(t0, format="mjd", scale="tdb"))
    p = np.concatenate([x + s.xyz.to(u.au).value, v + sv.xyz.to(u.au / u.day).value])
    cur = np.array(sorted(det))
    for it in range(6):
        e_ = d["det_exp"][cur]
        p, cov, res, chi2, dof = fit(p, t0, tall[e_], Rall[e_], uall[cur], np.full(len(cur), a.sigma))
        print(f"fit {it}: {len(cur)} det on {len(np.unique(np.floor(d['exp_t'][e_])))} nights, rms {np.sqrt((res ** 2).mean()):.3f}\", "
              f"chi2/dof {chi2 / dof:.2f};  bary {fmt(elements(p, t0))}")
        have = set(e_.tolist())
        others = np.array([k for k in range(len(d["exp_t"])) if k not in have])
        up, smaj = predict_sigma(p, cov, t0, tall[others], Rall[others])
        add = []
        for k, uk, sk in zip(others, up, smaj):
            m = np.flatnonzero(d["det_exp"] == k)
            if not len(m):
                continue
            sep = np.arccos(np.clip(uall[m] @ uk, -1, 1)) / ARCSEC
            j = np.argmin(sep); rad = max(a.rmin, 4 * sk)
            if sep[j] <= rad:
                add.append((m[j], k, sep[j], sk))
        if not add:
            break
        for mj, k, sp, sk in add:
            print(f"   + {d['exp_name'][k]}  MJD {d['exp_t'][k]:.3f}  offset {sp:6.2f}\"  predicted sigma {sk:7.2f}\"  "
                  f"mag {d['det_mag'][mj]:.2f} {d['det_filt'][mj] if 'det_filt' in d else ''}  labelled OF201: {d['det_obj'][mj] == 0}")
        cur = np.array(sorted(set(cur.tolist()) | {x[0] for x in add}))
    e_ = d["det_exp"][cur]
    out["decam"] = dict(det=cur.tolist(), nights=int(len(np.unique(np.floor(d["exp_t"][e_])))),
                        labelled=int((d["det_obj"][cur] == 0).sum()), extra=int((d["det_obj"][cur] != 0).sum()),
                        bary=elements(p, t0), helio=elements(p, t0, helio=True), t0_tdb=t0)
    lab = np.flatnonzero(d["det_obj"] == 0)
    print(f"DECam: {len(cur)} detections, {out['decam']['labelled']} of the {len(lab)} labelled OF201 ones, {out['decam']['extra']} others")
    print(f"   barycentric {fmt(out['decam']['bary'])}\n   heliocentric {fmt(out['decam']['helio'])}")
    if a.mpc:
        obs = [o for o in mpc_obs(a.mpc) if o[3] != "W84" and (not a.sites or o[3] in a.sites.split(","))]
        R, t = [], []
        for mjd, ra, dec, site in obs:
            r_, t_ = observer_bary(np.array([mjd]), site); R.append(r_[0]); t.append(t_[0])
        R, t = np.array(R), np.array(t)
        uo = radec_to_unit(np.array([o[1] for o in obs]), np.array([o[2] for o in obs]))
        up, smaj = predict_sigma(p, cov, t0, t, R)
        off = np.arccos(np.clip((up * uo).sum(1), -1, 1)) / ARCSEC
        print("archive check: the MPC observations predicted from the DECam-only orbit")
        rows = []
        for o, of, sk in zip(obs, off, smaj):
            rows.append(dict(mjd=o[0], site=o[3], offset=float(of), sigma=float(sk)))
            print(f"   {Time(o[0], format='mjd').iso[:10]} {o[3]}  offset {of:8.2f}\"  predicted 1-sigma {sk:8.2f}\"  ({of / sk:5.2f} sigma)")
        out["archive_check"] = rows
        e_ = d["det_exp"][cur]
        tt = np.concatenate([tall[e_], t]); RR = np.concatenate([Rall[e_], R]); uu = np.concatenate([uall[cur], uo])
        sg = np.concatenate([np.full(len(cur), a.sigma), np.full(len(obs), a.sigma)])
        p2, cov2, res2, chi2, dof = fit(p, t0, tt, RR, uu, sg)
        rr = np.hypot(res2[:, 0], res2[:, 1])
        print(f"DECam + archive ({len(tt)} observations, {Time(tt.min(), format='mjd').iso[:4]}-{Time(tt.max(), format='mjd').iso[:4]}): "
              f"rms {np.sqrt((res2 ** 2).mean()):.3f}\" per coordinate, max {rr.max():.2f}\"")
        sd = np.sqrt(np.diag(cov2))
        da = []
        for k in range(200):                          # element spread from the covariance
            pk = p2 + np.linalg.cholesky(cov2 + 1e-30 * np.eye(6)) @ np.random.default_rng(k).standard_normal(6)
            da.append([elements(pk, t0)[x] for x in ("a", "q", "e", "i")])
        da = np.std(np.array(da), 0)
        out["full"] = dict(n=len(tt), bary=elements(p2, t0), helio=elements(p2, t0, helio=True),
                           sd_bary=dict(a=da[0], q=da[1], e=da[2], i=da[3]))
        print(f"   barycentric {fmt(out['full']['bary'])}  (sd a {da[0]:.1f}, q {da[1]:.4f}, e {da[2]:.5f}, i {da[3]:.4f})")
        print(f"   heliocentric {fmt(out['full']['helio'])}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
