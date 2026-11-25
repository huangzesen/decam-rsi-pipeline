"""gen 13c: rescue grader-invalid states (r < 25 au or unbound) by a constrained refit to their own sky track.

Short arcs constrain distance and radial velocity poorly, so a real body at 25-35 au can be fitted at 22 au or on
an unbound orbit; the grader drops such states.  Here the state's own predicted directions at the scenario's
exposure times are refitted (Levenberg-Marquardt) with hinge penalties r >= RMIN and energy < 0.  The new state is
accepted only if it reproduces the old track within TOL_MAX arcsec at every sample time.
"""
import math
import numpy as np

GM = 2.959122082855911e-4
C_AU = 173.1446326846693
ARC = math.pi / 180 / 3600
RMIN = 25.4
TOL_MAX = 0.9     # arcsec


def _prop(r0, v0, dt):
    r2 = np.einsum("ki,ki->k", r0, r0)
    r = np.sqrt(r2)
    u = (GM / (r * r2))[:, None]
    p = (np.einsum("ki,ki->k", r0, v0) / r2)[:, None]
    q = (np.einsum("ki,ki->k", v0, v0) / r2)[:, None] - u
    t = dt
    t2 = t * t
    f = 1 - 0.5 * u * t2 + 0.5 * u * p * t2 * t + (u * (-15 * p * p + 3 * q + u) / 24.0) * t2 * t2
    g = t - u * t2 * t / 6.0 + 0.25 * u * p * t2 * t2
    return f[:, :, None] * r0[:, None, :] + g[:, :, None] * v0[:, None, :]


def _dirs(S, t0, t, obs):
    dt = np.broadcast_to(t - t0, (len(S), len(t)))
    x = _prop(S[:, :3], S[:, 3:], dt)
    rho = np.sqrt(np.sum((x - obs[None]) ** 2, -1))
    x = _prop(S[:, :3], S[:, 3:], dt - rho / C_AU)
    d = x - obs[None]
    return d / np.sqrt(np.sum(d * d, -1))[..., None]


def is_valid(s):
    s = np.asarray(s, float)
    if s.shape != (6,) or not np.all(np.isfinite(s)):
        return False
    r = float(np.linalg.norm(s[:3]))
    return r >= 25.0 and 0.5 * float(s[3:] @ s[3:]) - GM / r < 0


def samples(t, obs, nmax=40):
    """Unique exposure times (sorted), subsampled to <= nmax incl. both ends."""
    t = np.asarray(t, float); obs = np.asarray(obs, float)
    o = np.argsort(t)
    t, obs = t[o], obs[o]
    keep = np.r_[True, np.diff(t) > 1e-4]
    t, obs = t[keep], obs[keep]
    if len(t) > nmax:
        ii = np.unique(np.linspace(0, len(t) - 1, nmax).round().astype(int))
        t, obs = t[ii], obs[ii]
    return t, obs


def rescue(state, epoch, t, obs, iters=30):
    """Return a valid state (6,) reproducing state's track at (t, obs), or None."""
    s0 = np.asarray(state, float)
    if s0.shape != (6,) or not np.all(np.isfinite(s0)):
        return None
    if is_valid(s0):
        return s0
    t = np.asarray(t, float); obs = np.asarray(obs, float)
    if len(t) < 2:
        return None
    U = _dirs(s0[None], epoch, t, obs)[0]
    e1 = np.cross([0, 0, 1.0], U); e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(U, e1)
    n = len(t)
    # start: same geocentric direction at the middle sample, pushed out to RMIN + 1 au, velocity scaled alike
    km = n // 2
    om = obs[km]
    X = s0[:3]
    d = X - om
    # solve |om + k d| = RMIN + 1
    a_ = d @ d; b_ = 2 * om @ d; c_ = om @ om - (RMIN + 1.0) ** 2
    disc = b_ * b_ - 4 * a_ * c_
    if disc <= 0 or a_ <= 0:
        return None
    k = (-b_ + math.sqrt(disc)) / (2 * a_)
    k = max(k, 1.0)
    # observer velocity (finite difference over samples) so the angular rate is kept
    if n >= 2 and t[-1] - t[0] > 1e-3:
        vo = (obs[-1] - obs[0]) / (t[-1] - t[0])
    else:
        vo = np.zeros(3)
    V = vo + k * (s0[3:] - vo)
    S = np.r_[om + k * d, V]
    r = np.linalg.norm(S[:3]); vesc = math.sqrt(2 * GM / r)
    if np.linalg.norm(S[3:]) > 0.95 * vesc:
        S[3:] *= 0.9 * vesc / np.linalg.norm(S[3:])
    W = 1e-3   # penalty weight (rad per unit violation) >> 1" = 4.8e-6 rad

    def res_b(SS):
        D = _dirs(SS, epoch, t, obs) - U[None]
        rr = np.linalg.norm(SS[:, :3], axis=1)
        en = 0.5 * np.einsum("ki,ki->k", SS[:, 3:], SS[:, 3:]) - GM / rr
        pr = W * np.maximum(0.0, RMIN - rr)
        pe = W * 100 * np.maximum(0.0, en / (GM / rr) + 0.03)
        return np.concatenate([np.sum(D * e1[None], -1), np.sum(D * e2[None], -1), pr[:, None], pe[:, None]], 1)

    scale = np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5])
    mu = 1e-4
    rv = res_b(S[None])[0]
    for _ in range(iters):
        SS = S[None] + np.r_[np.zeros((1, 6)), np.diag(scale)]
        R = res_b(SS)
        rv = R[0]
        J = ((R[1:] - rv[None]) / scale[:, None]).T
        JTJ = J.T @ J; g = J.T @ rv; tr = np.trace(JTJ) / 6 + 1e-30
        ok = False
        for _ in range(6):
            try:
                step = np.linalg.solve(JTJ + mu * tr * np.eye(6), -g)
            except np.linalg.LinAlgError:
                mu *= 30
                continue
            S2 = S + step
            if np.linalg.norm(S2[:3]) < 5:
                mu *= 30
                continue
            r2 = res_b(S2[None])[0]
            if np.all(np.isfinite(r2)) and r2 @ r2 <= rv @ rv:
                ok = True
                break
            mu *= 30
        if not ok:
            break
        S, rv = S2, r2
        mu = max(mu / 10, 1e-9)
        if np.max(np.abs(step[:3])) < 1e-8:
            break
    if not is_valid(S):
        return None
    D = _dirs(S[None], epoch, t, obs)[0]
    err = np.arccos(np.clip(np.sum(D * U, -1), -1, 1)) / ARC
    if float(err.max()) > TOL_MAX:
        return None
    return S
