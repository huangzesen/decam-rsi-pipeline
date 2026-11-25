"""Leave-one-night-out flux predictions in the thresholded source catalogue."""
import json
from pathlib import Path
import os

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit, logit, ndtr


def measurements(candidates, directory, zero, colour, lever, nights, components, orbital=None):
    path = Path(directory)
    def raw(k):
        return np.load(path / ('det_' + k + '.npy'), mmap_mode='r')
    ex, flux, sn = raw('exp'), raw('flux'), raw('snr')
    groups = [np.asarray(c['det'], 'i8') for c in candidates]
    if not groups:
        return []
    rows = np.unique(np.concatenate(groups))
    e = ex[rows]
    mag = (zero-colour)[e] - 2.5*np.log10(np.maximum(1.e-12, flux[rows]))
    snr = sn[rows]
    out = []
    for d in groups:
        ix = np.searchsorted(rows, d)
        visit, m, s = e[ix], mag[ix], snr[ix].astype(float)
        key = tuple(sorted(d))
        comp = components.get(key) if components else None
        if comp:
            for i, row in enumerate(d):
                if int(row) in comp:
                    m[i], s[i] = comp[int(row)][2:]
        if orbital and key in orbital[1]:
            from visit_likelihood import range_magnitude
            state,ref=orbital[1][key]
            correction=range_magnitude(orbital[0],dict(x=state,ref=ref,e=visit),members=True)
            if np.isfinite(correction).all():
                m=m-correction
        use = np.isfinite(m+s) & (s>=4.48)
        if use.sum() < 5 or use.sum() > 60:
            out.append(None)
            continue
        out.append((nights[visit[use]], m[use], s[use], lever[visit[use]]))
    return out


def forecasts(group):
    night, mag, sn, lever = group
    labels, idx, count = np.unique(night, return_inverse=True, return_counts=True)
    if len(labels) < 5:
        return None
    # Retain all colours of the training nights. A rotational brightness
    # offset is shared between exposures on the same night.
    X = np.column_stack((np.ones(len(sn)), lever))
    base = np.median(mag)
    weights = 1./((1.086/sn)**2 + .055**2 + count[idx]*.105**2)
    # A source near the catalogue threshold has positive flux bias. The
    # prior on colour allows both neutral and very red reflected spectra.
    target = mag - base + .060 * np.exp(-np.maximum(sn-4.5, 0.)/1.3)
    robust = np.minimum(1., .60/np.maximum(.01, abs(target)))
    weights *= robust
    normal = X[:,:,None]*X[:,None,:]*weights[:,None,None]
    rhs = X*(weights*target)[:,None]
    N = normal.sum(0)+np.diag([.015, 1./.28**2])
    B = rhs.sum(0)
    nights_N = np.zeros((len(labels), 2, 2)); nights_B = np.zeros((len(labels), 2))
    np.add.at(nights_N, idx, normal)
    np.add.at(nights_B, idx, rhs)
    cov = np.linalg.inv(N-nights_N)
    fit = (cov@(B-nights_B)[:,:,None])[:,:,0]
    # Choose a representative exposure by predicted precision. The camera
    # noise in magnitude units uses the observed flux *divided by its S/N*,
    # which avoids choosing a high noise fluctuation as confirmation.
    mu = base + np.sum(X*fit[idx], axis=1)
    zp_noise = mag + 2.5*np.log10(sn)
    signal = 10.**np.clip(.4*(zp_noise-mu), -4., 4.)
    order = np.argsort(-signal, kind='stable')
    _, take = np.unique(idx[order], return_index=True)
    take = order[take]
    pred, obs = signal[take], sn[take]
    phot_var = np.einsum('ni,nij,nj->n',X[take],cov[idx[take]],X[take])
    # Use a mixture for nightly rotation and calibration error. Condition
    # the predictive distribution on the actual 4.5 sigma catalogue limit.
    def cumulative(scatter):
        scale = np.sqrt(1. + (pred/1.086)**2 * (phot_var+scatter**2))
        return ndtr((obs-pred)/scale), ndtr((4.5-pred)/scale)
    c1, t1 = cumulative(.115); c2, t2 = cumulative(.335)
    tail = .81*(1.-t1) + .19*(1.-t2)
    u = (.81*(c1-t1) + .19*(c2-t2)) / np.maximum(1.e-8, tail)
    return np.clip(u, 0., 1.), pred


def shifts(candidates, directory, zero, colour, lever, nights, components=None, orbital=None):
    groups = measurements(candidates, directory, zero, colour, lever, nights, components, orbital)
    raw = [forecasts(g) if g is not None else None for g in groups]
    rows = [[(j, float(u), float(s)) for u,s in zip(*record)]
            for j,record in enumerate(raw) if record is not None]
    if not rows:
        return np.zeros(len(candidates))
    rec = np.asarray([r for block in rows for r in block])
    owner = rec[:,0].astype('i4')
    signal = np.digitize(rec[:,2], [4.4,5.4,6.8,8.8,13.,25.])
    edges = np.array([0.,.025,.065,.13,.23,.36,.50,.64,.77,.87,.935,.975,1.])
    cell = np.digitize(rec[:,1], edges[1:-1])
    p = np.array([c['p'] for c in candidates])
    weak = p <= np.clip(np.quantile(p,.62), .04, .69)
    # Fit the null on other tracks. It includes flux cuts imposed by the
    # orbit search and local noise peaks at the edge of the catalogue.
    fold = np.array([min(c['det'])*2654435761%1000003%3 for c in candidates])
    evidence = np.zeros(len(owner))
    area = np.diff(edges)
    for f in range(3):
        train = weak[owner] & (fold[owner] != f)
        if train.sum() < 60:
            continue
        h = np.bincount(signal[train]*len(area)+cell[train], minlength=7*len(area)).reshape(7,-1)
        h = gaussian_filter(h.astype(float), (.55, .40), mode='nearest')
        overall = (h.sum(0)+30.*area)/(h.sum()+30.)
        h += 35.*overall[None,:]+12.*area
        density = h/(h.sum(1)[:,None]*area)
        test = fold[owner]==f
        evidence[test] = np.clip(-np.log(density[signal[test],cell[test]]), -1.85, 1.75)
    total = np.bincount(owner, weights=evidence, minlength=len(candidates))
    return np.clip(total, -7., 5.)


def run(F, outdir):
    G = F.G
    path = Path(outdir)/'candidates.json'
    candidates = json.loads(path.read_text())
    if len(candidates) < 24:
        return
    orbits=dict(G.get('survey_orbits',{}))
    orbits.update(G.get('report_orbits',{}))
    if orbits and 'ob_np_follow' not in G:
        G['ob_np_follow']=G['ob'].cpu().numpy()
    values = shifts(candidates, G['scen'], G['mz_np'], G['cb_np'], G['lev_e'],
                    G['night_e_np'], G.get('report_components'), (F,orbits))
    for c, shift in zip(candidates,values):
        c['flux_log_odds'] = round(float(shift), 3)
        c['p'] = float(expit(logit(np.clip(c['p'], 1.e-14, 1.-1.e-14))+.78*shift))
    candidates.sort(key=lambda c:-c['p'])
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(candidates))
    os.replace(tmp,path)
    F.log('held-out flux evidence:',int(np.count_nonzero(values)),'reports')
