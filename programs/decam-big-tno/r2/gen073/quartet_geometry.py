"""Bound orbital populations, apparent motion and measured survey coverage."""

import numpy as np


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


