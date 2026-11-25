"""Calibrate a marginal extra night by searching displaced copies of its orbit."""
import ctypes
import time

import numpy as np
from scipy.special import expit,logit


QUERY=np.dtype([('rows','i4',18),('n','i4'),('gamma','f4')])
RESULT=np.dtype([(k,'f4') for k in ('score','tail','contrast','available','clutter','encounters')]+
                [('checked','i4'),('omitted','i4')])


def measure(sky,depth,candidates,seconds,threads=2):
    q=np.zeros(len(candidates),QUERY)
    for i,c in enumerate(candidates):
        d=c['det']
        if not 5<=len(d)<=18:continue
        # The native refit profiles the directions and range again after
        # deleting a night. The initial radius only selects its starting basin.
        r=c.get('r',43.)
        if not 14.<r<8500.:continue
        q['rows'][i,:len(d)]=d
        q['n'][i]=len(d);q['gamma'][i]=1./r
    depth=np.ascontiguousarray(depth,'f8')
    out=np.zeros((len(q),12),RESULT)
    p,n,d=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    fn=sky.lib.displaced_confirmations
    fn.argtypes=[p,p,p,n,n,d,p];fn.restype=n
    tested=fn(sky.handle,depth.ctypes.data,q.ctypes.data,len(q),threads,seconds,out.ctypes.data)
    if tested<0:raise RuntimeError('withheld orbit confirmation controls failed')
    return out


def features(out):
    single=out[(out['checked']>0)&(out['omitted']==1)]
    if not len(single):return None
    def evidence(x):
        # The empirical tail includes the look-elsewhere effect over the
        # observed visits. The score margin distinguishes exceptionally
        # precise matches once none of the finite controls is as strong.
        fap=np.clip(x['tail'],.009,.99)
        odds=np.log(.5/fap)+.37*np.clip(x['contrast'],-8.,8.)
        return np.clip(odds,-6.,7.)
    single_score=evidence(single)
    pair=out[(out['checked']>0)&(out['omitted']==2)]
    return np.array([np.min(single_score),np.median(single_score),np.mean(single_score),
                     np.min(evidence(pair)) if len(pair) else np.nan,len(single)],float)


def adjust(candidates,values):
    count=0
    for c,v in zip(candidates,values):
        f=features(v)
        if f is None or f[-1]<2:continue
        n=c.get('nights',5)
        # A weak remote position can invalidate a five-night association.
        # Additional secure nights continue to support a longer one.
        weakest=f[0] if n<=5 else .50*f[0]+.50*f[1]
        pred=.56*weakest+.34*f[1]+.10*f[2]
        if np.isfinite(f[3]) and n<=7:pred=.82*pred+.18*f[3]
        reliability=np.clip(f[-1]/min(5,n),.25,1.)
        adj=np.clip((pred-.85)*reliability,-4.0,3.8)
        p=np.clip(c['p'],1.e-14,1.-1.e-14)
        c['control_visit_odds']=round(float(adj),3)
        c['p']=float(expit(logit(p)+.76*adj))
        count+=1
    return count


def run(F,reports,deadline):
    ctrl=F.G.get('survey_control') or {}
    sky=ctrl.get('original_sky')
    if sky is None or not hasattr(sky.lib,'displaced_confirmations'):return
    eligible=[r for r in sorted(reports,key=lambda c:-c['p']) if
              5<=r.get('nights',5)<=8 and 5<=len(r['det'])<=18 and
              tuple(sorted(r['det'])) not in F.G.get('report_components',{})][:550]
    if not eligible:return
    available=deadline-time.monotonic()
    if available<.35:return
    val=measure(sky,F.G['m5c_np'],eligible,available)
    count=adjust(eligible,val)
    F.log('displaced confirmation search:',count,'sparse trajectories measured')
