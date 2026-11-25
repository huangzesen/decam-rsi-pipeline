"""Shared helpers: timing, frames, two-body propagation with light time, orbit fit."""
import math, time
import numpy as np

T0 = time.time()
GM = 2.959122082855911e-4
C_AU = 173.1446326846693
ARC = math.pi / 180 / 3600


def left(budget):
    return budget - (time.time() - T0)


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def radec_to_unit(ra, dec):
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], -1)


def tangent_frame(ra0, dec0):
    z = radec_to_unit(ra0, dec0)
    x = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    y = np.cross(z, x)
    return np.stack([x, y, z])


# ----------------------------------------------------------------------------------------------- orbit model
def propagate(r0, v0, dt):
    """Two-body f,g series (accurate to ~1e-9 over months beyond 20 au). r0,v0 (3,), dt (N,) -> (N,3)."""
    r = np.linalg.norm(r0)
    mu = GM
    u = mu / r ** 3
    p = np.dot(r0, v0) / r ** 2
    q = np.dot(v0, v0) / r ** 2 - u
    t = dt
    f = 1 - 0.5 * u * t ** 2 + 0.5 * u * p * t ** 3 + (u * (-15 * p * p + 3 * q + u) / 24.0) * t ** 4
    g = t - u * t ** 3 / 6.0 + 0.25 * u * p * t ** 4
    return f[:, None] * r0[None] + g[:, None] * v0[None]


def propagate_b(r0, v0, dt):
    """Batched f,g series: r0, v0 (K,3), dt (K,N) -> (K,N,3)."""
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


def predict_dirs_b(states, t0, t, obs):
    """states (K,6) -> unit vectors (K,N,3), light time by one fixed-point iteration (error << 1 mas)."""
    r0, v0 = states[:, :3], states[:, 3:]
    dt = np.broadcast_to(t - t0, (len(states), len(t)))
    x = propagate_b(r0, v0, dt)
    rho = np.sqrt(np.sum((x - obs[None]) ** 2, -1))
    x = propagate_b(r0, v0, dt - rho / C_AU)
    d = x - obs[None]
    return d / np.sqrt(np.sum(d * d, -1))[..., None]


def predict_dirs(state, t0, t, obs):
    """Unit vectors observer->body at observation times t (light time included)."""
    return predict_dirs_b(np.asarray(state, float)[None], t0, np.asarray(t, float), np.asarray(obs, float))[0]


def fit_orbit(state0, t0, t, obs, uobs, iters=8, vr_sig=0.002):
    """Gauss-Newton on tangent-plane residuals (rad), Jacobian by batched finite differences.
    Returns state, residuals (arcsec)."""
    s = np.array(state0, float)
    t = np.asarray(t, float); obs = np.asarray(obs, float)
    e1 = np.cross([0, 0, 1.0], uobs); e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(uobs, e1)
    scale = np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5]) * max(np.linalg.norm(s[:3]) / 40, 0.3)
    n = len(t)

    def res_b(S):
        d = predict_dirs_b(S, t0, t, obs) - uobs[None]
        vr = np.einsum("ki,ki->k", S[:, :3], S[:, 3:]) / np.linalg.norm(S[:, :3], axis=1)
        # weak prior: radial velocity ~ 0 +- 0.002 au/day (poorly constrained by short arcs)
        return np.concatenate([np.sum(d * e1[None], -1), np.sum(d * e2[None], -1), (vr / vr_sig * 0.3 * ARC)[:, None]], 1)

    r = res_b(s[None])[0]
    mu = 1e-6
    for _ in range(iters):
        S = s[None] + np.r_[np.zeros((1, 6)), np.diag(scale)]
        R = res_b(S)
        r = R[0]
        J = ((R[1:] - r[None]) / scale[:, None]).T
        JTJ = J.T @ J
        g = J.T @ r
        tr = np.trace(JTJ) / 6
        improved = False
        for _ in range(4):                    # Levenberg-Marquardt: raise damping until the step helps
            step = np.linalg.solve(JTJ + mu * tr * np.eye(6), -g)
            s2 = s + step
            r2 = res_b(s2[None])[0]
            if np.all(np.isfinite(r2)) and np.sum(r2 ** 2) <= np.sum(r ** 2):
                improved = True
                break
            if np.max(np.abs(step[:3])) < 1e-6:      # converged: the failed step is negligible anyway
                break
            mu *= 30
        if not improved:
            break
        s, r = s2, r2
        mu = max(mu / 10, 1e-7)
        if np.max(np.abs(step[:3])) < 1e-7:
            break
    n = len(t)
    return s, np.hypot(r[:n], r[n:2 * n]) / ARC


