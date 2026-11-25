"""Allocate long-baseline continuation work by a marginalized orbital prior.

A pair leaves range unresolved. Integrate a broad cold/hot orbit mixture over
logarithmic distance, rather than treating its best-fitting distance as known.
The original photometric queue retains a separate share of the proposals.
This affects search order only; confirmation and confidence are unchanged.
"""
import math
import os
import time
import numpy as np
import torch


def order(F, pairs, key, rr, deadline):
    G=F.G;dev=G['dev'];started=time.time();n=len(pairs);total=n
    old=torch.argsort(key,stable=True)
    if n<2:return old
    # A cold concentration and a broad hot component allocate finite work.
    # They impose no acceptance cut. The original photometric ordering
    # retains a separate share for unusual orbital populations.
    # Subtract measured positions in double precision below. The broad,
    # approximate population score itself needs only single precision.
    grid=torch.exp(torch.linspace(math.log(1/rr[1]),math.log(1/rr[0]),33,
                                   dtype=torch.float32,device=dev))
    nc=torch.sqrt(F.MU*grid**3)[None]
    axis=G['epro'];cross=torch.stack((-axis[1],axis[0]))
    prior=torch.empty(n,dtype=torch.float32,device=dev)
    batch=32768 if dev!='cpu' else 8192
    processed=0
    for lo in range(0,n,batch):
        if time.time()>deadline:
            break
        # Refine the photometric prefix first, so a short allowance still
        # yields useful ordering without an arbitrary night-pair preference.
        visit=old[lo:lo+batch]
        pp=pairs[visit];ee=G['de'][pp];tt=G['et'][ee]
        th=G['th'][pp];ob=G['ob'][ee]
        dt=(tt[:,1]-tt[:,0]).abs().clamp(min=.3)
        D=ob[...,:2]-th*ob[...,2:3]
        w0=(th[:,1]-th[:,0])/dt[:,None]
        w1=(D[:,1]-D[:,0])/dt[:,None]
        along=(w0@axis).float()[:,None]+grid[None]*(w1@axis).float()[:,None]
        across=(w0@cross).float()[:,None]+grid[None]*(w1@cross).float()[:,None]
        u=along/nc;v=across/nc
        # A broad distribution of eccentricity and inclination induces a
        # concentration of forward transverse rates near the circular rate.
        # The hot component covers slow, fast, and strongly inclined motion.
        cold=-.5*((u-1.)/.25).square()-.5*(v/.22).square()
        hot=-.5*((u-.8)/.7).square()-.5*(v/.65).square()
        mixture=torch.logaddexp(math.log(.7)+cold,math.log(.3)+hot)
        sn=G['snr_t'][pp].float().clamp(min=4.5)
        sigma=(((1.1/sn).square()+.04**2).sum(1)).sqrt()*F.AS/dt.float()
        excess=(torch.sqrt(along.square()+across.square())-math.sqrt(2)*nc).clamp(min=0)
        bound=-.5*(excess/(sigma[:,None]+.10*nc)).square()
        prior[visit]=torch.logsumexp(mixture+bound,dim=1)-math.log(len(grid))
        processed=lo+len(pp)
    if not processed:
        F.log('pair orbit priority: time limit, retained original queue')
        return old
    chosen=remaining=None
    if processed<n:
        chosen=old[:processed];remaining=old[processed:]
        key=key[chosen];prior=prior[chosen];n=processed
        old=torch.arange(n,device=dev)
    prior=torch.nan_to_num(prior,nan=-30.,neginf=-30.,posinf=0.)
    physical=key/2147483648.-2.*prior
    better=torch.argsort(physical,stable=True)
    a=torch.empty(n,dtype=torch.int64,device=dev);b=torch.empty_like(a)
    ranks=torch.arange(n,device=dev);a[better]=ranks;b[old]=ranks
    schedule=torch.minimum(4*(a//3)+a%3,4*b+3)
    result=torch.argsort(schedule,stable=True)
    if chosen is not None:
        result=torch.cat((chosen[result],remaining))
    F.log('pair orbit priority:',processed,'of',total,'pairs;',round(time.time()-started,1),'s')
    if os.environ.get('FINDER_PAIR_DUMP') and processed==total:
        np.savez(os.environ['FINDER_PAIR_DUMP'],p=pairs.cpu().numpy(),key=key.cpu().numpy(),prior=prior.cpu().numpy(),order=result.cpu().numpy())
    return result
