"""Conditional astrometry of a moving source through three visits and later returns."""
import itertools
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

import quartet_geometry as orbit


RETURNS = 10
FEATURES = 28
MAX_VIEWS = 3 + RETURNS
STEP = np.dtype([('coef', 'f4', (FEATURES, 2)), ('chol', 'f4', 3)])
MODEL = np.dtype([('view', 'i4', MAX_VIEWS), ('count', 'i4'), ('family', 'i4'),
                  ('gamma', 'f4'), ('step', STEP, MAX_VIEWS),
                  ('correlation', 'f4', (RETURNS, RETURNS, 4))])


def schedule(xy, ex, visits, groups, cov, pole, trials, deadline, observed=None):
    """Collect three-visit starts and all additional visits observable on their paths."""
    rng = np.random.default_rng(4581)
    cover = orbit.Coverage(cov, len(visits))
    t, nights = visits['t'], visits['night']
    depth = visits['depth']
    observer = np.column_stack([visits[k] for k in ('ox', 'oy', 'oz')])
    edges = np.searchsorted(ex, np.arange(len(t) + 1))
    weights = np.sqrt(cover.area / np.bincount(nights)[nights]) * (np.diff(edges) > 0)
    if not weights.sum():
        return []
    weights /= weights.sum()
    records = {}
    paths = []
    combos = {i: np.array(list(itertools.combinations(range(i), 3)))
              for i in range(5, 11)}
    for _ in range(0, trials, 256):
        if time.monotonic() >= deadline:
            break
        anchor = rng.choice(len(t), 256, p=weights)
        site = edges[anchor] + (rng.random(len(anchor))*np.diff(edges)[anchor]).astype('i8')
        family = rng.choice(len(orbit.MIX), len(anchor), p=orbit.MIX)
        z = rng.standard_normal((len(anchor), 6)).clip(-2.8, 2.8)
        state = orbit.directions(xy[site], *orbit.variation(z, family), observer[anchor], pole)
        pos = orbit.apparent(*state, t - t[anchor, None], observer)
        mag = depth[anchor] + .30 - rng.exponential(.65, len(anchor))
        prob = cover.lookup(pos) / (1. + np.exp(np.clip((mag[:, None]-depth-.12)/.25, -28., 28.)))
        detected = rng.random(prob.shape) < prob
        for i in range(len(anchor)):
            used = np.flatnonzero(detected[i])
            if len(used) < 5:
                continue
            used = used[np.argsort(-prob[i, used]*rng.uniform(.5, 1.5, len(used)))]
            _, first = np.unique(nights[used], return_index=True)
            if len(first) < 5:
                continue
            used = used[np.sort(first)]
            if observed is not None:
                observed.append((float(mag[i]),int(family[i]),used[:24].copy()))
            obs = used[:10]
            a, b, c = obs[combos[len(obs)]].T
            epochs = np.stack((a, b, c), axis=1)
            epochs.sort(axis=1)  # Exposure rows need not be time-ordered.
            epochs = np.take_along_axis(epochs, np.argsort(t[epochs], axis=1), axis=1)
            dates = t[epochs]
            gap = np.diff(dates, axis=1)
            later = gap[:, 1] < gap[:, 0]
            a = np.where(later, epochs[:, 1], epochs[:, 0])
            b = np.where(later, epochs[:, 2], epochs[:, 1])
            c = np.where(later, epochs[:, 0], epochs[:, 2])
            base = np.minimum(gap[:, 0], gap[:, 1])
            span = dates[:, 2] - dates[:, 0]
            # Very short arcs have few accidental pairs. Long arcs resolve
            # the range and radial velocity of a subsequent return.
            utility = (.44*np.log1p(span/45.) - .85*np.log1p(base/12.)
                       - .28*np.log1p(span/(14.+4.*base))
                       - .6*np.maximum(0., 1.-base))
            order = np.argsort(-(utility + .88*rng.gumbel(size=len(span))))[:5]
            weight = (5./len(used))**.75
            votes = []
            for j in order:
                ab = (int(groups[a[j]]), int(groups[b[j]]))
                key = (*sorted(ab), int(groups[c[j]]), int(family[i]))
                if key not in records:
                    records[key] = [len(records), 0., {}]
                idx, mass, ret = records[key]
                records[key][1] = mass+weight
                votes.append(idx)
                for e in used[:18]:
                    g = int(groups[e])
                    if g not in key[:3]:
                        ret[g] = ret.get(g, 0.) + weight*prob[i, e]
            paths.append((votes, weight))
    if not paths:
        return []
    values = list(records.values())
    order = orbit.cover_order([p for p, _ in paths], np.array([r[1] for r in values]),
                              np.array([w for _, w in paths]), rng)
    keys = list(records)
    return [(keys[j], sorted(values[j][2], key=values[j][2].get, reverse=True)[:RETURNS])
            for j in order if len(values[j][2]) >= 2]


def physical_draws():
    u = qmc.Sobol(6 + 2*MAX_VIEWS, scramble=True, seed=1619).random_base2(8)
    return ndtri(np.clip(u, .00025, .99975))


def features(z):
    n = z.shape[-1]
    i, j = np.triu_indices(n)
    return np.concatenate((np.ones(z.shape[:-1]+(1,)), z, z[..., i]*z[..., j]), axis=-1)


def models(plans, visits, center, spread, pole, z, with_paths=False):
    """Integrate over a broad bound-orbit family, keeping the shared error between returns."""
    batch = len(plans)
    out = np.zeros(batch, MODEL)
    for j, ((a, b, c, family), follow) in enumerate(plans):
        out['view'][j] = np.resize([a, b, c, *follow], MAX_VIEWS)
        out['count'][j], out['family'][j] = len(follow), family
    views = out['view']
    a, family = views[:, 0], out['family']
    ob = np.column_stack([visits[k] for k in ('ox', 'oy', 'oz')])
    xy = center[a, None, :] + np.deg2rad(.66)*z[None, :, :2]
    pos, vel = orbit.directions(xy, *orbit.variation(z[None, :, :6], family[:, None]),
                                ob[a, None, :], pole)
    t = visits['t'][views]
    samples = orbit.apparent(pos, vel, (t-t[:, :1])[:, None, :], ob[views][:, None, :, :])/orbit.AS
    # Several close exposures can measure a night. A grouped coordinate
    # has residual angular-rate uncertainty from its transport to mid-visit.
    rate = np.array([.9, 1.8, 2.5, 3.4, 3.7, 3.4])[family]
    jitter = np.hypot(.23, rate[:, None]*spread[views]*7.7)
    samples += jitter[:, None, :, None]*z[:, 6:].reshape(1, len(z), MAX_VIEWS, 2)
    out['gamma'] = np.sqrt(1.+np.sum(center[a]**2, axis=1))/orbit.FAMILIES[family, 0]
    previous = np.empty((batch, len(z), 0))
    base = samples[:, :, 0].copy()
    future = []
    for k in range(MAX_VIEWS):
        if k <= 3:
            design = features(previous)
            xt = design.transpose(0, 2, 1)
            normal = xt @ design
            # Use a smooth physical map for each visit. The output is
            # relative to the first position to preserve local translations.
            ridge = np.full(normal.shape[1], .055)
            ridge[:1+previous.shape[2]] = .00002
            normal += np.diag(ridge)
            inverse = np.linalg.solve(normal, xt)
        response = samples[:, :, k] if k == 0 else samples[:, :, k] - base
        coef = inverse @ response
        residual = response-design@coef
        cov = residual.transpose(0, 2, 1)@residual / (len(z)-design.shape[2])
        cov += np.eye(2)*.075**2
        L = np.linalg.cholesky(cov)
        white = np.linalg.solve(L, residual.transpose(0, 2, 1)).transpose(0, 2, 1)
        out['step']['coef'][:, k, :coef.shape[1]] = coef
        out['step']['chol'][:, k] = L[:, [0, 1, 1], [0, 0, 1]]
        if k < 3:
            previous = np.concatenate((previous, white), axis=2)
        else:
            future.append(white)
    u = np.stack(future, axis=1)
    # For an uncertain three-night orbit the two later positions need
    # not be individually precise; their *difference* can be very precise.
    corr = np.einsum('brni,bsnj->brsij', u, u, optimize=True)/(len(z)-design.shape[2])
    out['correlation'] = (.997*corr).reshape(batch, RETURNS, RETURNS, 4)
    if with_paths:
        return out, (samples[:, ::19] - base[:, ::19, None, :]).astype('f4')
    return out
