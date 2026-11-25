"""Two-body orbits, the observer, frames. Units: au, days, radians; heliocentric J2000 equatorial frame.

The synthetic catalogue and the linker share nothing but the detection file, so the observer model only has to be
self-consistent: Earth-Moon barycentre from Standish's mean elements (~arcmin), geocentric observer.
"""
import numpy as np

GM = 2.959122082855911e-4          # au^3 / day^2 (Sun)
C_AU = 173.1446326846693           # au / day
EPS = np.radians(23.43928)         # obliquity J2000
MJD_J2000 = 51544.5
ARCSEC = np.pi / 180 / 3600


def ecl_to_eq(v):
    c, s = np.cos(EPS), np.sin(EPS)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([x, c * y - s * z, s * y + c * z], -1)


def eq_to_ecl(v):
    c, s = np.cos(EPS), np.sin(EPS)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([x, c * y + s * z, -s * y + c * z], -1)


def kepler_E(M, e):
    E = M + e * np.sin(M)
    for _ in range(30):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.all(np.abs(dE) < 1e-14):
            break
    return E


def elements_to_rv(a, e, inc, Om, om, M):
    """Vectorised Keplerian elements (ecliptic angles) -> heliocentric ecliptic position, velocity."""
    E = kepler_E(np.mod(M, 2 * np.pi), e)
    cE, sE = np.cos(E), np.sin(E)
    b = a * np.sqrt(1 - e * e)
    n = np.sqrt(GM / a ** 3)
    xp, yp = a * (cE - e), b * sE
    edot = n / (1 - e * cE)
    vxp, vyp = -a * sE * edot, b * cE * edot
    cO, sO, co, so, ci, si = np.cos(Om), np.sin(Om), np.cos(om), np.sin(om), np.cos(inc), np.sin(inc)
    P = np.stack([cO * co - sO * so * ci, sO * co + cO * so * ci, so * si], -1)
    Q = np.stack([-cO * so - sO * co * ci, -sO * so + cO * co * ci, co * si], -1)
    r = xp[..., None] * P + yp[..., None] * Q
    v = vxp[..., None] * P + vyp[..., None] * Q
    return r, v


def rv_to_elements(r, v):
    """Heliocentric ecliptic state -> (a, e, inc, Om, om, M) for bound orbits."""
    rn = np.linalg.norm(r, axis=-1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)
    inc = np.arccos(np.clip(h[..., 2] / hn, -1, 1))
    Om = np.arctan2(h[..., 0], -h[..., 1])
    ev = np.cross(v, h) / GM - r / rn[..., None]
    e = np.linalg.norm(ev, axis=-1)
    a = 1 / (2 / rn - np.sum(v * v, -1) / GM)
    nvec = np.stack([np.cos(Om), np.sin(Om), np.zeros_like(Om)], -1)
    hz = h / hn[..., None]
    om = np.arctan2(np.sum(np.cross(nvec, ev) * hz, -1), np.sum(nvec * ev, -1))
    f = np.arctan2(np.sum(np.cross(ev, r) * hz, -1), np.sum(ev * r, -1))
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f / 2))
    M = E - e * np.sin(E)
    return a, e, inc, Om, om, M


def earth_eq(mjd):
    """Heliocentric equatorial position of the Earth-Moon barycentre (Standish 1992 mean elements, 1800-2050)."""
    T = (np.asarray(mjd, float) - MJD_J2000) / 36525.0
    a = 1.00000261 + 0.00000562 * T
    e = 0.01671123 - 0.00004392 * T
    inc = np.radians(-0.00001531 - 0.01294668 * T)
    L = np.radians(100.46457166 + 35999.37244981 * T)
    varpi = np.radians(102.93768193 + 0.32327364 * T)
    Om = np.zeros_like(T)
    r, _ = elements_to_rv(a, e, inc, Om, varpi - Om, L - varpi)
    return ecl_to_eq(r)


def radec_to_unit(ra, dec):
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], -1)


def unit_to_radec(u):
    return np.mod(np.arctan2(u[..., 1], u[..., 0]), 2 * np.pi), np.arcsin(np.clip(u[..., 2], -1, 1))


def tangent_frame(ra0, dec0):
    """Rows x, y, z: z toward (ra0, dec0), x toward increasing RA, y toward north."""
    z = radec_to_unit(ra0, dec0)
    x = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    y = np.cross(z, x)
    return np.stack([x, y, z])
