"""cand9: n-body orbit fitting for the HSC test (as nbody_fit.py: barycentric test particle under the Sun, Jupiter, Saturn,
Uranus and Neptune from astropy's builtin ephemeris, light time, each telescope's own site). Points are
(mjd_utc, ra, dec, sigma_arcsec, site) with site in {"DECam", "CFHT", "Subaru"}.
fit(points, state0, epoch) -> dict(state, rms, chi2, resid). rXX() gives the first arc's points and a barycentric start state."""
import json, math, os
import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, get_body_barycentric, get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

C = os.path.dirname(os.path.abspath(__file__)); CAU = 173.1446326846693
GM = dict(sun=2.9591220828559115e-04, jupiter=2.825345909524226e-07, saturn=8.459715185680659e-08, uranus=1.292024916781969e-08, neptune=1.524358900784276e-08)
SITES = {"DECam": EarthLocation.from_geodetic(-70.80653 * u.deg, -30.16954 * u.deg, 2207 * u.m),
         "CFHT": EarthLocation.from_geodetic(-155.46833 * u.deg, 19.82525 * u.deg, 4204 * u.m),
         "Subaru": EarthLocation.from_geodetic(-155.47611 * u.deg, 19.82556 * u.deg, 4139 * u.m)}
GRID = np.arange(Time("2012-06-01").mjd, Time("2026-01-01").mjd, 1.0)
with solar_system_ephemeris.set("builtin"):
    _tt = Time(GRID, format="mjd", scale="tdb")
    SP = {b: CubicSpline(GRID, get_body_barycentric(b, _tt).xyz.to(u.au).value.T) for b in GM}


def observer(m, site):
    t = Time(m, format="mjd", scale="utc")
    with solar_system_ephemeris.set("builtin"):
        e = get_body_barycentric("earth", t).xyz.to(u.au).value
    return e + SITES[site].get_gcrs_posvel(t)[0].xyz.to(u.au).value


def rhs(t, y):
    r = y[:3]; acc = np.zeros(3)
    for b, gm in GM.items():
        d = SP[b](t) - r; acc += gm * d / np.linalg.norm(d) ** 3
    return np.r_[y[3:], acc]


def traj(state, ep, lo, hi):
    f = solve_ivp(rhs, (ep, hi + 2), state, method="DOP853", rtol=1e-10, atol=1e-12, dense_output=True)   # both sides always (2026-11-22: a missing side crashed light time; hid C### = 2006 TM130)
    b = solve_ivp(rhs, (ep, lo - 2), state, method="DOP853", rtol=1e-10, atol=1e-12, dense_output=True)
    return lambda t: (f.sol(t) if t >= ep else b.sol(t))[:3]


def radec(tr, m, o):
    tau = 0.
    for _ in range(3):
        g = tr(m - tau) - o; tau = np.linalg.norm(g) / CAU
    return math.degrees(math.atan2(g[1], g[0])) % 360, math.degrees(math.asin(g[2] / np.linalg.norm(g)))


_OBS = {}


def obs_cached(m, site):
    k = (round(m, 7), site)
    if k not in _OBS:
        _OBS[k] = observer(m, site)
    return _OBS[k]


def predict(state, ep, times_sites):
    ms = [m for m, _ in times_sites]; tr = traj(np.asarray(state, float), ep, min(ms + [ep]), max(ms + [ep]))
    return [radec(tr, m, obs_cached(m, s)) for m, s in times_sites]


def fit(points, state0, ep, max_nfev=60):
    sc = np.array([1e-3] * 3 + [1e-5] * 3); s0 = np.asarray(state0, float)
    ms = [p[0] for p in points]; lo, hi = min(ms + [ep]), max(ms + [ep]); w = np.array([1 / p[3] for p in points])

    def res(q):
        tr = traj(s0 + q * sc, ep, lo, hi); out = []
        for (m, a, d, sg, site) in points:
            ra, de = radec(tr, m, obs_cached(m, site))
            out += [((ra - a + 180) % 360 - 180) * math.cos(math.radians(d)) * 3600, (de - d) * 3600]
        return np.array(out)
    W = np.repeat(w, 2); H = 0.02  # central differences with a fixed step in q (2e-5 au, 2e-7 au/day): far above the integrator's noise, still linear.
    # (A relative diff_step collapses to ~1e-8 at q = 0, the start of every fit: the Jacobian there is integrator noise and the covariance meaningless.)

    def jac(q):
        J = []
        for k in range(6):
            d = np.zeros(6); d[k] = H; J.append((res(q + d) - res(q - d)) * W / (2 * H))
        return np.array(J).T
    f = least_squares(lambda q: res(q) * W, np.zeros(6), jac=jac, x_scale=1.0, max_nfev=max_nfev)
    r = res(f.x); chi2 = float(np.sum((r * np.repeat(w, 2)) ** 2)); dof = max(2 * len(points) - 6, 1)
    U_, s_, Vt_ = np.linalg.svd(f.jac, full_matrices=False); covq = (Vt_.T / s_ ** 2) @ Vt_ * max(1.0, chi2 / dof)   # covfix 2026-11-23
    cov = covq * np.outer(sc, sc)  # in state units (au, au/day)
    return dict(state=(s0 + f.x * sc).tolist(), epoch=ep, rms=float(np.sqrt(np.mean(r ** 2))), chi2=chi2,
                resid=r.reshape(-1, 2).tolist(), maxres=float(np.max(np.hypot(*r.reshape(-1, 2).T))), cov=cov.tolist())


def predict_cloud(fr, times_sites, n=60, seed=1):
    """Positions at times_sites for n states drawn from the fit's covariance (linear), plus the best state's.
    Returns best (list of (ra, dec)) and spread (list of 1-sigma major-axis sizes, arcsec)."""
    rng = np.random.default_rng(seed); st = np.array(fr["state"]); cov = np.array(fr["cov"])
    best = predict(st, fr["epoch"], times_sites); cloud = []
    for _ in range(n):
        cloud.append(predict(rng.multivariate_normal(st, cov), fr["epoch"], times_sites))
    cloud = np.array(cloud); spread = []
    for k, (a, d) in enumerate(best):
        x = ((cloud[:, k, 0] - a + 180) % 360 - 180) * math.cos(math.radians(d)) * 3600; y = (cloud[:, k, 1] - d) * 3600
        spread.append(float(np.sqrt(np.max(np.linalg.eigvalsh(np.cov(np.c_[x, y].T))))))
    return best, spread


def rXX():
    d = json.load(open(os.path.join(C, "cand6_F3_rXX_pX.json")))
    pts = [(p[0], p[1], p[2], p[3], "CFHT" if p[4] == "CFHT" else "DECam") for p in d["points"]]
    ep = d["best"]["epoch"]; s = np.array(d["best"]["state"], float)  # heliocentric (cand6 used two-body about the Sun)
    with solar_system_ephemeris.set("builtin"):
        sp, sv = get_body_barycentric_posvel("sun", Time(ep, format="mjd", scale="tdb"))
    return pts, s + np.r_[sp.xyz.to(u.au).value, sv.xyz.to(u.au / u.day).value], ep


PLAN_DATES = []


if __name__ == "__main__":
    pts, s0, ep = rXX(); f = fit(pts, s0, ep)
    print(f"first arc alone, n-body: rms {f['rms']:.3f}\", max {f['maxres']:.3f}\", chi2 {f['chi2']:.1f} for {2 * len(pts)} numbers")
    for dte, mj in PLAN_DATES:  # (date, mjd) pairs of the HSC test; removed from this snapshot (dates tied to an unannounced body)
        (a, dd), = predict(f["state"], ep, [(mj, "Subaru")]); print(f"  {dte}: predicted RA {a:.4f} Dec {dd:.4f} (from the best first-arc fit only)")
    json.dump(f, open(os.path.join(C, "cand9_rXX_arc1fit.json"), "w"))
