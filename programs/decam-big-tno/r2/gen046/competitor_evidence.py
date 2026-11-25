"""Assess stationary and proper-motion alternatives in the original visits."""
import ctypes
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression


EVIDENCE = np.dtype([(n,'f4') for n in ('fixed','follow','linear','control','stationary','offset')]
                    + [('repeats','i4'),('epochs','i4')])


def measure(index, reports, seconds):
    rows = [np.asarray(c['det'],'i4') for c in reports]
    ends = np.r_[0,np.cumsum([len(r) for r in rows],dtype='i4')].astype('i4')
    joined = np.concatenate(rows)
    ranges = 1./np.clip([c.get('r',43.) for c in reports],12.,8500.)
    result = np.zeros(len(joined), EVIDENCE)
    result['epochs']=-1
    ptr,integer,real = ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    fn = index.lib.competing_tracks
    fn.argtypes = [ptr,ptr,ptr,ptr,integer,integer,real,ptr]
    fn.restype = integer
    completed = fn(index.handle,joined.ctypes.data,ends.ctypes.data,ranges.ctypes.data,
                   len(rows),2,max(.1,seconds),result.ctypes.data)
    if completed < 0:
        raise RuntimeError('comparison with stationary sources failed')
    return joined,ends,result


def empirical_odds(actual,controls,rows,weak):
    # The control positions represent a body superposed on unrelated sky.
    # Use other orbit members to estimate the competing background class.
    edges = [1.5,3.5,6.,9.,13.,17.]
    obs,ctrl = np.digitize(actual,edges),np.digitize(controls,edges)
    bins = np.arange(len(edges)+1)
    folds = (rows.astype('i8')*2246822519%1000003)%2
    out = np.zeros(len(rows))
    for fold in (0,1):
        sel = weak & (folds!=fold)
        if sel.sum()<50:continue
        a = np.bincount(obs[sel],minlength=len(bins)) + 2.5
        b = np.bincount(ctrl[sel],minlength=len(bins)) + 2.5
        a /= a.sum()
        b /= b.sum()
        lr = IsotonicRegression(increasing=False).fit_transform(bins,np.log(b/a),sample_weight=a+b)
        take = folds==fold
        out[take] = lr[obs[take]] - np.log(np.sum(a*np.exp(lr)))
    return np.clip(out,-5.,.55)


def evaluate(reports, rows, ends, result, exposure, nights):
    p = np.asarray([c['p'] for c in reports])
    weak = np.repeat((p<=np.clip(np.quantile(p,.67),.06,.80)),np.diff(ends))
    # A nonzero stellar motion should improve on a fixed star in the same
    # visits, including when randomly displaced apertures contain a star.
    linear = np.minimum(result['linear'],1.4*np.maximum(0.,result['linear']-result['stationary']))
    ctrl = np.minimum(result['control'],1.4*np.maximum(0.,result['control']-result['offset']))
    valid=result['epochs']>=0
    moving_star = empirical_odds(linear,ctrl,rows,weak&valid)*valid
    still = empirical_odds(result['stationary'],result['offset'],rows,weak&valid)*valid
    # When independent visits favour both hypotheses the star may be a
    # blend. Only the difference in their support distinguishes them.
    relative = np.clip(result['follow']-result['fixed'],-8.,8.)
    odds = np.column_stack((moving_star,still,relative))
    offsets = np.zeros((len(reports),3))
    nn = nights[exposure[rows]]
    for i,c in enumerate(reports):
        sl=slice(ends[i],ends[i+1])
        for n in np.unique(nn[sl]):
            v = odds[sl][nn[sl]==n]
            # Multiple image measurements of one sky position are dependent.
            # A separately supported moving source can cross a real star.
            offsets[i,0] += np.mean(v[:,0])
            offsets[i,1] += np.mean(v[:,1])
            neg = np.minimum(v[:,2],0.)
            pos = np.maximum(v[:,2],0.)
            signed = pos.max()+neg.min()
            if signed>0.:
                reported = len(v)>1 or c.get('h3m',0)>0 or c.get('visit_motion',0.)>.5
                offsets[i,2] += (.15 if reported else .63)*max(0.,signed-1.1)
            else:
                offsets[i,2] -= .85*max(0.,-signed-1.)
    return np.clip(offsets,[-7.,-8.,-10.],[3.,3.,6.])


def apply(reports, offsets):
    for c,terms in zip(reports,offsets):
        shift = .80*terms[0] + .53*terms[1] + .8*terms[2]
        if abs(shift)>.002:
            # Long, repeatedly measured orbits are insensitive to the odds
            # of a single night's chance overlap with a foreground star.
            shift *= min(1.,6./max(6,c.get('nights',5)))
            c['p'] = float(expit(logit(np.clip(c['p'],1.e-14,1.-1.e-14))+shift))
            c['alternative_log_odds'] = round(float(shift),3)
    return sorted(reports,key=lambda c:-c['p'])


def run(F, outdir, deadline):
    G=F.G
    index=(G.get('survey_control') or {}).get('original_sky')
    if index is None or time.time()+.6>=deadline:return
    path=Path(outdir)/'candidates.json'
    reports=sorted(json.loads(path.read_text()),key=lambda c:-c['p'])
    if not reports:return
    rows,ends,result=measure(index,reports,deadline-time.time()-.4)
    odds=evaluate(reports,rows,ends,result,index.exposure,G['night_e_np'])
    reports=apply(reports,odds)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(reports))
    os.replace(tmp,path)
    F.log('competing source trajectories:',int((result['repeats']>0).sum()),'rows with independent intra-night astrometry')
