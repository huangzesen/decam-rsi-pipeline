"""Test a reported peak against independently measured foreground light."""
import time

import numpy as np
from scipy.special import logsumexp, ndtr

AS = np.deg2rad(1. / 3600.)
AXIS = np.array([-np.sqrt(3.), 0., np.sqrt(3.)])
NODES = np.array(np.meshgrid(AXIS, AXIS, AXIS, indexing='ij')).reshape(3, -1).T
WEIGHTS = np.array([1./6., 2./3., 1./6.])
LOG_WEIGHT = np.log(np.einsum('i,j,k->ijk', WEIGHTS, WEIGHTS, WEIGHTS).ravel())


def predict(G, v):
    ix = v['take']
    sigma = np.hypot(v['errors'][ix], .065)
    design = v['design'][ix]
    J = design / sigma[:, None, None]
    A = J.transpose(0, 2, 1) @ J
    B = np.einsum('nki,nk->ni', J, v['resxy'][ix] / sigma[:, None])
    prior = np.diag([1.e-12, 1.e-12, 3., 4., 4., 9.])
    V = np.linalg.inv(prior + A.sum(0) - A)
    correction = (V @ (B.sum(0) - B)[:, :, None])[:, :, 0]
    delta = v['resxy'][ix] - (design @ correction[:, :, None])[:, :, 0]
    delta += (v['xy'][ix] - v['actual_xy'][ix]) / AS
    covariance = design @ V @ design.transpose(0, 2, 1)
    levers = G['lev_e'][v['e'][ix]]
    X = np.column_stack((np.ones(len(ix)), levers))
    mag = v['model_mag'][ix]
    valid = np.isfinite(mag)
    if valid.sum() < 3:
        return None
    center = np.median(mag[valid])
    color = float(np.clip(v['color'], -.5, .6))
    # Suppress the influence of a second foreground peak in the training
    # nights, while allowing rotational variability and passband differences.
    weight = valid / (.145**2 + (1.086/np.maximum(v['snr'][ix], 3.))**2
                      + .22*np.nan_to_num(mag-center-color*levers)**2)
    P = X[:, :, None] * X[:, None, :] * weight[:, None, None]
    Y = X * (weight * np.nan_to_num(mag-center))[:, None]
    W = np.linalg.inv(P.sum(0) + np.diag([.0001, 1./.27**2]) - P)
    par = (W @ (Y.sum(0) - Y)[:, :, None])[:, :, 0]
    m = center + (par*X).sum(1) + v['mag'][ix] - mag
    scatter = np.sqrt(.145**2 + np.einsum('ni,nij,nj->n', X, W, X))
    return delta, covariance, m, scatter


def densities(star, delta, orbit_cov, mover_flux, scatter, sn, fwhm, density):
    """Integrate the flux and the gradient of the two-source PSF at its peak."""
    xy = star[:2]
    width = np.clip(fwhm/2.355, .27, 1.1)
    if xy@xy > (2.9*width)**2:
        return 0.
    signal = max(4.5, sn)
    noise = np.hypot(.050, 1./signal)
    centroid = np.hypot(.075, 1.12/signal)
    ps = np.hypot(centroid, star[3])
    fs = np.hypot(noise, star[2]*star[4])
    # A repeated peak at this position and flux increases the conditional
    # local background above the average density of unrelated field peaks.
    star_chi = (xy@xy)/ps**2 + ((1.-star[2])/fs)**2
    norm = 1.5*np.log(2.*np.pi)
    static = np.logaddexp(np.log(.94)-.5*star_chi,
                          np.log(.06/5.**3)-.5*star_chi/25.) - norm - np.log(fs*ps**2)
    average = max(.00006, density)*1.086
    reliability = (star[5]-1.)/(star[5]+2.)
    background = np.logaddexp(0., static+np.log(reliability/average))
    orbit_cov = .5*(orbit_cov+orbit_cov.T)
    eigen, rotation = np.linalg.eigh(orbit_cov)
    if eigen[-1] > 2.8:
        # Cubature is inaccurate for an orbit aperture wider than the PSF.
        return -min(background, 5.) if star_chi<6. else 0.
    rotation = rotation * np.sqrt(np.maximum(eigen, 1.e-8))
    location = delta + NODES[:, :2] @ rotation.T
    body = mover_flux * np.exp(np.clip(.921034*scatter*NODES[:, 2], -1.5, 1.5))
    d2 = np.einsum('ij,ij->i', location, location)
    a = star[2]*np.exp(-(xy@xy)/(4.*width**2))
    b = body*np.exp(-d2/(4.*width**2))
    flux = np.maximum(a+b, .001)
    frac = b/flux
    residual = (1.-frac[:, None])*xy+frac[:, None]*location
    # Near a saddle the inferred centre is not a single catalogue peak.
    separation = np.sum((location-xy)**2, axis=1)
    curvature = 1.-frac*(1.-frac)*separation/(2.*width**2)
    sf2 = noise**2+(a*star[4])**2+.045**2
    sp2 = centroid**2+((1.-frac)*star[3])**2+.04**2
    chi = (1.-flux)**2/sf2+np.sum(residual**2,axis=1)/sp2
    inclusion = np.maximum(1.e-18,ndtr((b*signal-3.15)/1.30)*ndtr((frac-.12)/.055)
                           *ndtr((curvature-.09)/.12))
    blend = logsumexp(LOG_WEIGHT + np.log(inclusion) - .5*chi
                      - .5*np.log(sf2) - np.log(sp2)) - norm
    # Density of the same peak under the orbit alone, in the same units.
    cov = orbit_cov+np.eye(2)*centroid**2
    _, area = np.linalg.slogdet(cov)
    phot2 = noise**2+(.921034*mover_flux*scatter)**2
    chi = delta @ np.linalg.solve(cov, delta)+(1.-mover_flux)**2/phot2
    single = -.5*(chi+area+np.log(phot2))-norm
    overlap = np.clip(np.exp(-(xy@xy)/(4.*width**2))*reliability, .05, .91)
    gain = np.logaddexp(np.log1p(-overlap), np.log(overlap)+np.clip(blend-single,-17.,14.))
    change = gain-background
    if change>0.:
        if star[12]<=0. or star[10]<12. or star[11]>.65 or star[9]<3.:
            return 0.
        change=min(change,max(0.,blend-np.log(average)-2.))
    return float(np.clip(change, -8., 5.))


def nightly(G, v):
    ix = v['take']
    result = np.zeros(len(ix))
    if v['candidate'].get('r', 42.) > 150.:
        return result
    refs = v['foreground_ref'][ix]
    use = np.flatnonzero((refs[:, 5]>=2) & (refs[:, 2]>.07))
    if not len(use):
        return result
    model = predict(G, v)
    if model is None:
        return result
    delta, cov, m, scatter = model
    rho = v.get('null_density', np.full(len(ix), .001))
    for j in use:
        i = ix[j]
        flux = 10.**np.clip(-.4*(m[j]-v['actual_flux_mag'][i]), -2.5, 2.)
        if np.isfinite(flux+delta[j].sum()+cov[j].sum()+scatter[j]):
            result[j] = densities(refs[j],delta[j],cov[j],flux,scatter[j],v['actual_sn'][i],v['fwhm'][i],rho[j])
    return result


def evaluate(G, items, deadline):
    ctrl = G.get('survey_control') or {}
    sky = ctrl.get('original_sky')
    changes = np.zeros(len(items))
    if sky is None or not items:
        return changes
    rows = np.unique(np.concatenate([v['d'][v['take']] for v in items]))
    refs = np.zeros((len(rows),13))
    for start in range(0,len(rows),96):
        if time.monotonic()>=deadline:
            break
        refs[start:start+96] = sky.photocenters(rows[start:start+96])
    for j,v in enumerate(items):
        raw = v['d']
        v['foreground_ref'] = np.zeros((len(raw),13))
        take = v['take']
        v['foreground_ref'][take] = refs[np.searchsorted(rows,raw[take])]
        v['actual_xy'],v['actual_flux_mag'] = sky.pos[raw],sky.mag[raw]
        v['actual_sn'],v['fwhm'] = sky.arrays[0][raw],sky.seeing[v['e']]
        try:
            evidence = nightly(G,v)
        except (np.linalg.LinAlgError,FloatingPointError):
            continue
        changes[j] = association(evidence)
    return changes


def association(evidence):
    # Foreground evidence for an unnecessary sixth or seventh assignment
    # has less bearing on the existence of a five-night orbit.
    ordered=np.sort(evidence)
    weight=np.ones(len(ordered))
    extra=min(2,max(0,len(ordered)-5))
    weight[:extra]=np.where(ordered[:extra]<0.,.28,1.)
    return float(np.clip(.54*(weight@ordered),-6.2,3.0))
