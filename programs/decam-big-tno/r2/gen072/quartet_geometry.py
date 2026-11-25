"""Observable four-visit schedules and two-epoch astrometric coordinates."""
import itertools
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc


AS = np.deg2rad(1. / 3600.)
MU = 2.96309e-4
# Median range, logarithmic range width, prograde speed relative to circular,
# and dispersions in along-track, cross-track and radial speed.
FAMILIES = np.array([
    (44., .085, 1., .105, .15, .15),
    (37., .16, .98, .19, .37, .25),
    (60., .23, .91, .25, .48, .36),
    (160., .43, .82, .38, .58, .46),
    (700., .69, .65, .52, .63, .49),
    (42., .22, .08, .74, .70, .45),
])
MIX = np.array([.46, .28, .13, .060, .035, .035])


def directions(xy, radius, along, across, radial, observer, pole):
    """Heliocentric state conditional on a measured topocentric direction."""
    direction = np.concatenate((xy, np.ones_like(xy[..., :1])), axis=-1)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    dot = np.sum(observer * direction, axis=-1)
    delta = np.sqrt(radius**2 + dot**2 - np.sum(observer**2, axis=-1)) - dot
    pos = observer + delta[..., None] * direction
    unit = pos / radius[..., None]
    east = np.cross(pole, unit)
    east /= np.maximum(1.e-8, np.linalg.norm(east, axis=-1, keepdims=True))
    north = np.cross(unit, east)
    speed = np.stack((along, across, radial), axis=-1)
    # Smooth truncation at escape speed keeps the broad family bound.
    speed *= np.minimum(1., 1.395 / np.maximum(.01, np.linalg.norm(speed, axis=-1)))[..., None]
    vel = (speed[..., :1]*east + speed[..., 1:2]*north + speed[..., 2:3]*unit)
    return pos, vel * np.sqrt(MU / radius)[..., None]


def apparent(pos, vel, dt, observer):
    """Two-body angles across the last (exposure) axis of dt."""
    r = np.linalg.norm(pos, axis=-1)[..., None]
    alpha = 2. / r - np.sum(vel**2, axis=-1)[..., None] / MU
    n = np.sqrt(MU*alpha**3)
    sin0 = np.sum(pos*vel, axis=-1)[..., None] * np.sqrt(alpha/MU)
    cos0 = 1. - r*alpha
    mean = n*dt
    ecc = mean.copy()
    for _ in range(5):
        s, c = np.sin(ecc), np.cos(ecc)
        ecc -= (ecc - cos0*s + sin0*(1.-c) - mean)/(1. - cos0*c + sin0*s)
    f = 1. - 2.*np.sin(.5*ecc)**2/(r*alpha)
    g = dt + (np.sin(ecc) - ecc)/n
    xyz = f[..., None]*pos[..., None, :] + g[..., None]*vel[..., None, :] - observer
    return xyz[..., :2]/xyz[..., 2:]


def variation(samples, family):
    r, dr, speed, du, dv, dh = np.moveaxis(FAMILIES[family], -1, 0)
    return (r*np.exp(dr*samples[..., 2]), speed+du*samples[..., 3],
            dv*samples[..., 4], dh*samples[..., 5])


class Coverage:
    def __init__(self, cov, count):
        origin, keys = cov[:2]
        self.origin = np.asarray(origin)
        ex = keys >> 42
        x = ((keys >> 21) & ((1 << 21) - 1)) // 6
        y = (keys & ((1 << 21) - 1)) // 6
        edges = np.searchsorted(ex, np.arange(count+1))
        self.low = np.zeros((count, 2), 'i8')
        self.size = np.ones((count, 2), 'i8')
        pieces = []
        for e, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            if b == a:
                pieces.append(np.zeros(1, 'u1'))
                continue
            u, v = x[a:b], y[a:b]
            self.low[e] = [u.min(), v.min()]
            w, h = int(np.ptp(u))+1, int(np.ptp(v))+1
            if w*h > 1000000:
                pieces.append(np.zeros(1, 'u1'))
                continue
            self.size[e] = [w, h]
            key = (u-u.min())*h + v-v.min()
            pieces.append(np.bincount(key, minlength=w*h).clip(0, 36).astype('u1'))
        self.area = np.array([a.sum() for a in pieces], float) * (45./3600.)**2
        self.start = np.r_[0, np.cumsum([len(p) for p in pieces])[:-1]]
        self.grid = np.concatenate(pieces)

    def lookup(self, xy):
        cell = np.floor((xy-self.origin)/(270.*AS)).astype('i8')-self.low
        valid = np.all((cell >= 0) & (cell < self.size), axis=-1)
        k = self.start + cell[..., 0]*self.size[:, 1]+cell[..., 1]
        return np.where(valid, self.grid[np.clip(k, 0, len(self.grid)-1)]/36., 0.)


def schedule(xy, ex, visits, cov, pole, trials, deadline, seed=1973, group=None):
    """Importance sample tuples on orbits with at least five accessible nights."""
    rng = np.random.default_rng(seed)
    ne = len(visits)
    area = Coverage(cov, ne)
    et, night = visits['t'], visits['night']
    ob = np.column_stack([visits[k] for k in ('ox', 'oy', 'oz')])
    depth = visits['depth'].copy()
    depth[~np.isfinite(depth)] = 23.
    edges = np.searchsorted(ex, np.arange(ne+1))
    # Integrate over live silicon. Dense observing bursts cannot supply many
    # independent detections merely by observing the same footprint repeatedly.
    counts = np.bincount(night)
    weights = np.sqrt(area.area) / np.sqrt(counts[night]) * (np.diff(edges) > 0)
    if not weights.sum():
        return np.empty((0, 5), 'i4')
    weights /= weights.sum()
    tuples = {};mass=[];draws=[];importance=[]
    # Distinguish a narrow apparent trajectory from a set of unrelated sources
    # by combining two measured displacements in order of observing time.
    combos = {k: np.array(list(itertools.combinations(range(k), 4)))
              for k in range(4, 12)}
    for _ in range(0, trials, 256):
        if time.monotonic() > deadline:
            break
        anchor = rng.choice(ne, 256, p=weights)
        site = edges[anchor] + (rng.random(len(anchor))*np.diff(edges)[anchor]).astype('i8')
        fam = rng.choice(len(MIX), len(anchor), p=MIX)
        z = rng.normal(size=(len(anchor), 6)).clip(-2.7, 2.7)
        pos, vel = directions(xy[site], *variation(z, fam), ob[anchor], pole)
        path = apparent(pos, vel, et-et[anchor, None], ob)
        live = area.lookup(path)
        # The distribution of detectable flux is steep near an image limit.
        mag = depth[anchor] + .29 - rng.exponential(.63, len(anchor))
        prob = live/(1.+np.exp(np.clip((mag[:, None]-depth-.15)/.25, -25., 25.)))
        detection_draws = rng.random(prob.shape)
        for i in range(len(anchor)):
            idx = np.flatnonzero(detection_draws[i] < prob[i])
            if len(idx) < 5:
                continue
            order = idx[np.argsort(-prob[i, idx]*rng.uniform(.55, 1.45, len(idx)))]
            _, first = np.unique(night[order], return_index=True)
            if len(first) < 5:
                continue
            pick = order[np.sort(first)[:11]]
            pick = pick[np.argsort(et[pick])]
            quad = pick[combos[len(pick)]]
            t = et[quad]
            gaps = np.diff(t, axis=1)
            admissible = (t[:, 3]-t[:, 0] > 18.) & (gaps[:, 0] > .40) & (gaps[:, 2] > .40)
            quad, gaps = quad[admissible], gaps[admissible]
            if not len(quad):
                continue
            cost = np.log1p(gaps[:, 0]/9.) + np.log1p(gaps[:, 2]/9.)
            info = .65*np.log1p(np.minimum(gaps[:, 0], gaps[:, 2])/2.)
            # A few draws explore expensive cross-year pairs with no local arc.
            priority = rng.gumbel(size=len(quad)) + info - .58*cost
            take = np.argsort(-priority)[:6]
            weight = (5./max(5, len(first)))**.65
            choices=[]
            for q in quad[take]:
                if group is not None:
                    q = group[q]
                key = (*map(int, q), int(fam[i]))
                slot=tuples.get(key)
                if slot is None:
                    slot=len(tuples);tuples[key]=slot;mass.append(0.)
                mass[slot]+=weight;choices.append(slot)
            draws.append(choices);importance.append(weight)
    if not tuples:
        return np.empty((0, 5), 'i4')
    keys = np.asarray(list(tuples), 'i4')
    quality=np.asarray(mass);weight=np.asarray(importance)
    return keys[cover_order(draws,quality,weight,rng)]


def cover_order(draws,mass,weight,rng):
    """Discount repeats of the same simulated trajectory in the early schedule."""
    sizes=np.fromiter(map(len,draws),'i4')
    path=np.repeat(np.arange(len(draws),dtype='i4'),sizes)
    ids=np.concatenate(draws).astype('i4')
    ordered=np.argsort(ids,kind='stable')
    edge=np.r_[0,np.cumsum(np.bincount(ids,minlength=len(mass)))]
    ends=np.r_[0,np.cumsum(sizes)]
    noise=.47*rng.gumbel(size=len(mass))
    score=np.log(mass+.14)+noise
    parts=[]
    taken=np.zeros(len(mass),bool)
    # A scheduled quartet may still fail detection, association or fitting.
    # Keep some value for a second independent chance on the same path.
    while sum(map(len,parts))<min(20000,len(mass)):
        n=min(96,len(mass)-sum(map(len,parts)))
        indices=np.argpartition(-score,n-1)[:n]
        selected=indices[np.argsort(-score[indices],kind='stable')]
        parts.append(selected);taken[selected]=True
        touched=np.concatenate([path[ordered[edge[j]:edge[j+1]]] for j in selected])
        trials,multiplicity=np.unique(touched,return_counts=True)
        shrink=weight[trials]*(1.-.45**multiplicity);weight[trials]-=shrink
        for j,amount in zip(trials,shrink):
            np.subtract.at(mass,ids[ends[j]:ends[j+1]],amount)
        score=np.log(np.maximum(0.,mass)+.14)+noise
        score[taken]=-np.inf
    rest=np.flatnonzero(~taken)
    parts.append(rest[np.argsort(-score[rest],kind='stable')])
    return np.concatenate(parts)


def quadrature(size):
    # A scrambled low-discrepancy sequence keeps the orbit distribution
    # reproducible for two visits appearing in several different tuples.
    uniform = qmc.Sobol(14, scramble=True, seed=8693).random_base2(int(np.log2(size)))
    return ndtri(np.clip(uniform, .0005, .9995))


def sample(visits, center, pole, plans, z):
    e = plans[:, :4]
    f = plans[:, 4, None]
    ob = np.column_stack([visits[k] for k in ('ox', 'oy', 'oz')])
    xy = center[e[:, 0], None, :] + np.deg2rad(.54)*z[None, :, :2]
    pos, vel = directions(xy, *variation(z[None, :, :6], f), ob[e[:, :1]], pole)
    t = visits['t'][e]
    xy = apparent(pos, vel, (t-t[:, :1])[:, None, :], ob[e][:, None, :, :])/AS
    # Vary centroid errors independently by exposure, including a mild
    # heavy tail. These coordinates serve only to retrieve hypotheses.
    return xy + .22*z[:, 6:].reshape(-1, 4, 2)


def pair_vector(xy):
    return np.concatenate((xy[..., 0, :], xy[..., 1, :] - xy[..., 0, :]), axis=-1)


def covar(x, y):
    return np.swapaxes(x, -1, -2) @ y / x.shape[-2]


def whitening(x):
    # Return features normalized to unit covariance and a matrix that
    # can be applied directly to the four measured angular coordinates.
    center = np.mean(x, axis=1, keepdims=True)
    xc = x-center
    scale = np.maximum(.08, np.std(xc, axis=1))
    v = xc/scale[:, None]
    chol = np.linalg.cholesky(covar(v, v)+1.e-9*np.eye(4))
    trans = np.linalg.inv(chol).transpose(0, 2, 1)/scale[:, :, None]
    return xc@trans, center[:, 0], trans


_I, _J = np.triu_indices(4)


def squared(x):
    return x[..., _I]*x[..., _J]


def relation(samples):
    """Fit common coordinates to two independent pairs from the same orbit."""
    va, ca, ta = whitening(pair_vector(samples[..., :2, :]))
    vb, cb, tb = whitening(pair_vector(samples[..., 2:, :]))
    left, _, rt = np.linalg.svd(covar(va, vb))
    right = rt.transpose(0, 2, 1)
    delta = va@left-vb@right
    qa, qb = squared(va), squared(vb)
    quadratic = np.concatenate((qa, qb), axis=-1)
    mean = quadratic.mean(axis=1, keepdims=True)
    quadratic -= mean
    # Regress only curvature of the two tightest linear invariants.
    # The ridge prevents nearly identical polynomial modes on the two
    # sides from cancelling the useful linear coordinates.
    coef = np.linalg.solve(covar(quadratic, quadratic)+.025*np.eye(20),
                           covar(quadratic, delta))
    coef[..., 2:] = 0.
    residual = delta-quadratic@coef
    sig = np.linalg.cholesky(covar(residual, residual)+1.e-12*np.eye(4))
    rescale = np.linalg.inv(sig).transpose(0, 2, 1)
    pa = np.concatenate((mean@coef, left, -coef[:, :10]), axis=1)@rescale
    pb = np.concatenate((np.zeros_like(pa[:, :1]), right, coef[:, 10:]), axis=1)@rescale
    return ca, ta, pa, cb, tb, pb


def pair_gate(samples):
    """Apparent displacement conditional on the measured first position."""
    a, b = samples[..., 0, :], samples[..., 1, :]
    origin, drift = a.mean(axis=1), (b-a).mean(axis=1)
    p, d = a-origin[:, None], (b-a)-drift[:, None]
    shear = np.linalg.solve(covar(p, p), covar(p, d))
    residual = d-p@shear
    cov = covar(residual, residual)+.12**2*np.eye(2)
    return np.column_stack((origin, drift, shear.reshape(-1, 4), cov[:, 0, 0],
                            cov[:, 0, 1], cov[:, 1, 1]))
