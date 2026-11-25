"""Odds that a burst follows the linked orbit in the original catalogues."""
import ctypes
import time

import numpy as np
from scipy.special import logsumexp
import torch

QUERY=np.dtype([('row','i4'),('pad','i4')]+[(k,'f8') for k in
               ('vx','vy','g','mag','colour','magvar','colourvar')])
VALUE=np.dtype([(k,'f4') for k in ('linked','steady','fast','slow')]+
               [('visits','i4'),('peaks','i4')])


def queries(F, items):
    G=F.G
    count=np.array([len(v['take']) for v in items])
    owner=np.repeat(np.arange(len(items)),count)
    if not len(owner):return np.empty(0,QUERY),owner
    q=np.zeros(len(owner),QUERY)
    exposure=np.concatenate([v['e'][v['take']] for v in items])
    q['row']=np.concatenate([v['d'][v['take']] for v in items])
    q['g']=np.stack([v['x'] for v in items])[owner,2]
    for lo in range(0,len(owner),10000):
        sl=slice(lo,lo+10000)
        dev=G['dev']
        x=torch.as_tensor(np.stack([v['x'] for v in items])[owner[sl]],device=dev)
        e=torch.as_tensor(exposure[sl],device=dev)
        ref=torch.as_tensor(np.asarray([v['ref'] for v in items])[owner[sl]],device=dev)
        dt=(G['et'][e]-ref)[:,None]
        ob=G['ob'][e,None]
        velocity=(F.kmodel(x,dt+.05,ob)-F.kmodel(x,dt-.05,ob))[:,0].cpu().numpy()/(.1*F.AS)
        q['vx'][sl],q['vy'][sl]=velocity.T
        q['g'][sl]/=np.maximum(.7,1.-q['g'][sl]*ob[:,0,2].cpu().numpy())
    # Fit the photometry of other nights to predict the colour locally.
    # The anchor conditions its own brightness term inside the burst fit.
    at=0
    for v,n in zip(items,count):
        j=v['take'];m=v['model_mag'][j];lev=G['lev_e'][v['e'][j]]
        sn=np.maximum(v['snr'][j],4.5)
        prior=np.nanmedian(m)
        valid=np.isfinite(m)
        m=np.where(valid,m,prior)
        s2=.15**2+(1.086/sn)**2
        design=np.column_stack((np.ones(n),lev))
        outer=design[:,:,None]*design[:,None,:]*(valid/s2)[:,None,None]
        rhs=design*((valid/s2)*m)[:,None]
        normal=np.diag([.005,1/.29**2])+outer.sum(0)-outer
        cov=np.linalg.inv(normal)
        theta=(cov@(rhs.sum(0)-rhs+[.005*prior,0.])[:,:,None])[:,:,0]
        slab=slice(at,at+n)
        delta=np.where(valid,v['mag'][j]-m,0.)
        q['mag'][slab]=(design*theta).sum(1)+delta
        q['colour'][slab]=theta[:,1]
        q['magvar'][slab]=np.einsum('ni,nij,nj->n',design,cov,design)
        q['colourvar'][slab]=cov[:,1,1]+.055**2
        at+=n
    valid=np.isfinite(q['vx']+q['vy']+q['mag']+q['colour'])&(q['g']>0)&(q['g']<.15)
    return q[valid],owner[valid]


def raw_measure(sky,q,zp,depth,seconds):
    ptr,integer,real=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    fn=sky.lib.burst_alternatives
    fn.argtypes=[ptr,ptr,integer,ptr,ptr,integer,real,ptr];fn.restype=integer
    zp=np.ascontiguousarray(zp,'f8');depth=np.ascontiguousarray(depth,'f8')
    out=np.zeros(len(q),VALUE)
    count=fn(sky.handle,q.ctypes.data,len(q),zp.ctypes.data,depth.ctypes.data,2,
             max(.01,seconds),out.ctypes.data)
    if count<0:raise RuntimeError('burst alternative likelihood failed')
    return out


def mixture(scores,mask,prior,regularize):
    obs=scores[mask]
    weight=prior
    if len(obs):
        for _ in range(14):
            z=obs+np.log(weight)
            post=np.exp(z-logsumexp(z,axis=1)[:,None])
            weight=(post.sum(0)+regularize*prior)/(len(obs)+regularize)
    return weight


def odds(q,result,owner,items,sky):
    usable=(result['visits']>0)&np.isfinite(result['linked'])
    if not usable.any():return np.zeros(len(items))
    # The single-night alternative can be an uncatalogued counterpart, a
    # star, a faster asteroid, or another slowly moving source. Infer their
    # relative frequency from other low-confidence associations in the field.
    sky_class=getattr(sky,'population',None)
    tier=sky_class[q['row']] if sky_class is not None else np.ones(len(q),'i4')
    sn=sky.arrays[0][q['row']]
    groups=(tier==0).astype('i4')*4+np.digitize(sn,[6.,10.,25.])
    loglikes=np.column_stack((np.zeros(len(q)),result['steady'],result['fast'],result['slow']))
    p=np.array([v['candidate']['_report']['p'] for v in items])
    cal=(p[owner]<=np.clip(np.quantile(p,.63),.07,.72))&usable
    # Duplicates in the assignment pool do not represent new sky sources.
    _,unique=np.unique(q['row'][cal],return_index=True)
    which=np.flatnonzero(cal)[unique];cal[:]=False;cal[which]=True
    folds=(q['row'].astype('i8')*2654435761%1000003)%3
    ratio=np.zeros(len(q))
    for fold in range(3):
        use=cal&(folds!=fold)
        global_prior=mixture(loglikes,use,np.array([.65,.28,.05,.02]),55.)
        for level in range(8):
            w=mixture(loglikes,use&(groups==level),global_prior,70.)
            test=usable&(folds==fold)&(groups==level)
            ratio[test]=result['linked'][test]-logsumexp(loglikes[test]+np.log(w),axis=1)
    ratio=np.clip(ratio,-8.,5.)
    total=np.bincount(owner,weights=ratio,minlength=len(items))
    for i,v in enumerate(items):
        c=v['candidate']['_report'];sel=(owner==i)&usable
        if not sel.any():continue
        # Some repeated measurements contributed to earlier discovery and
        # confirmation. Discount their already counted positive motion odds.
        prior=max(.7*c.get('visit_repeat',0.),.65*min(c.get('h3m',0),3))
        total[i]-=min(max(0.,total[i]),prior)
        c['burst_nights']=int(sel.sum())
        c['burst_log_ratio']=round(float(total[i]),3)
    return np.clip(total,-10.,8.)


def evaluate(F,items,deadline):
    G=F.G
    sky=(G.get('survey_control') or {}).get('original_sky')
    if sky is None or not hasattr(sky.lib,'burst_alternatives') or time.monotonic()+.5>=deadline:
        return np.zeros(len(items))
    q,owner=queries(F,items)
    if not len(q):return np.zeros(len(items))
    response=raw_measure(sky,q,G['mz_np']-G['cb_np'],G['m5c_np'],deadline-time.monotonic())
    return odds(q,response,owner,items,sky)
