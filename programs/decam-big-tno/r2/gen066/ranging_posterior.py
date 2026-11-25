"""Integrate distance and radial speed for competing nightly assignments."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from scipy.special import expit,logit


def measure(F, items, lib, deadline):
    func = lib.sparse_membership_evidence
    ptr = ctypes.c_void_p
    func.argtypes = [ptr,ctypes.c_int,ptr,ctypes.c_double,ptr,ptr]
    func.restype = ctypes.c_double
    G = F.G
    eps = np.deg2rad(23.43928)
    pole = G['frame_np']@np.array([0.,-np.sin(eps),np.cos(eps)])
    def evaluate(v):
        if time.monotonic()>=deadline:
            return
        ix = v['take'];n = len(ix)
        if n<5 or n>8:
            return
        x = np.ascontiguousarray(v['x'],'f8')
        direction = np.r_[x[:2],1.]
        east = np.cross(pole,direction)
        east /= max(1.e-12,np.linalg.norm(east))
        e = v['e'][ix]
        obs = np.ascontiguousarray(np.column_stack((G['et_np'][e],G['ob_np_follow'][e],v['xy'][ix],
                np.hypot(.07,v['errors'][ix]),v['model_mag'][ix],G['lev_e'][e],
                v['snr'][ix],v['null_density'])),dtype='f8')
        if not np.isfinite(obs).all() or np.any(obs[:,-1]<=0):
            return
        full = np.full(1,-1.e8)
        value = func(obs.ctypes.data,n,x.ctypes.data,v['ref'],east.ctypes.data,full.ctypes.data)
        if full[0]>-1.e6 and np.isfinite(value) and value>-1.e7:
            return float(value),float(full[0])
    # Highest probability sparse reports are evaluated first. Finite sample
    # budgets should not depend on the arbitrary order of catalogue rows.
    order = sorted(items,key=lambda v:-v['candidate']['_report']['p'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,values in enumerate(pool.map(evaluate,order)):
            if values is not None:
                yield order[i],values


def apply(F,items,lib,deadline):
    count=0
    for v,(score,full) in measure(F,items,lib,deadline):
        c=v['candidate']['_report']
        c['range_log_bf']=round(score,3)
        c['range_log_evidence']=round(full,3)
        if 'partial_log_bf' in c:
            # Both use the same measured rows. Only the change in the
            # dynamical evidence updates the already calibrated odds.
            log_odds=logit(np.clip(c['p'],1.e-14,1.-1.e-14))
            delta=np.clip(score-c['partial_log_bf'],-8.,8.)
            c['p']=float(expit(log_odds+.45*delta))
        count+=1
    return count
