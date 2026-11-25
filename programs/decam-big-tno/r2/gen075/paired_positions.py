"""Membership ambiguity from correlated alternative peaks on two omitted nights."""
import ctypes
from itertools import combinations
import time

import numpy as np

QUERY=np.dtype([('rows','i4',2),('offset','f8',4),('cov','f8',(4,4)),
                ('mag','f8',2),('phot','f8',3)])
VALUE=np.dtype([('same','f4'),('one','f4'),('any','f4'),('pairs','i4')])


def queries(G,item):
    ix=item['take'];n=len(ix)
    if not 5<=n<=8:return np.empty(0,QUERY)
    pairs=np.array(list(combinations(range(n),2)))
    j=item['design'][ix];res=item['resxy'][ix]
    sigma=np.hypot(item['errors'][ix],.065)
    mag=item['model_mag'][ix];sn=np.maximum(4.5,item['snr'][ix])
    if not np.isfinite(j).all() or not np.isfinite(mag).all():return np.empty(0,QUERY)
    weighted=j/sigma[:,None,None]
    atoms=weighted.transpose(0,2,1)@weighted
    rhs=np.einsum('nci,nc->ni',weighted,res/sigma[:,None])
    keep=np.ones((len(pairs),n))
    keep[np.arange(len(pairs))[:,None],pairs]=0.
    norm=(keep@atoms.reshape(n,36)).reshape(-1,6,6)+np.diag([1.e-12,1.e-12,1.5,3.,3.,6.])
    cov=np.linalg.inv(norm)
    step=(cov@(keep@rhs)[:,:,None])[:,:,0]
    design=j[pairs].reshape(-1,4,6)
    center=res[pairs].reshape(-1,4)-(design@step[:,:,None])[:,:,0]
    sky=design@cov@design.transpose(0,2,1)
    sky[:,range(4),range(4)]+=np.repeat(sigma[pairs]**2,2,axis=1)
    var=np.diagonal(sky,axis1=1,axis2=2)
    valid=(np.abs(step[:,2])<.4)&np.isfinite(center).all(1)&(var.min(1)>.0001)&(var.max(1)<110.**2)
    take=np.flatnonzero(valid)
    if not len(take):return np.empty(0,QUERY)
    # Use the area jointly searched on the two nights to select geometries,
    # independently of the observed rows at either position.
    area=np.linalg.slogdet(sky[take])[1]
    take=take[np.argsort(-area,kind='stable')[:5]]
    p=pairs[take]
    e=item['e'][ix];lev=G['lev_e'][e]
    d=np.column_stack([np.ones(n),lev]);mu=np.median(mag)
    wt=1./(.15**2+(1.086/sn)**2)
    info=(d[:,:,None]*d[:,None,:])*wt[:,None,None]
    V=np.linalg.inv((keep[take]@info.reshape(n,4)).reshape(-1,2,2)+np.diag([.003,1/.30**2]))
    theta=(V@(keep[take]@(d*((mag-mu)*wt)[:,None]))[:,:,None])[:,:,0]
    pred=d[p];m=(pred@theta[:,:,None])[:,:,0]+mu
    m+=item['mag'][ix][p]-mag[p]
    P=pred@V@pred.transpose(0,2,1)
    P[:,range(2),range(2)]+=.12**2
    out=np.empty(len(take),QUERY)
    out['rows']=item['d'][ix][p];out['cov']=sky[take];out['offset']=center[take]
    out['mag']=m;out['phot']=P[:,[0,0,1],[0,1,1]]
    return out


def evaluate(F,items,deadline,library=None):
    G=F.G;sky=(G.get('survey_control') or {}).get('original_sky')
    lib=library or (sky.lib if sky is not None else None)
    adj=np.zeros(len(items))
    if sky is None or not hasattr(lib,'paired_assignments'):return adj
    fn=lib.paired_assignments
    ptr=ctypes.c_void_p;integer=ctypes.c_int;real=ctypes.c_double
    fn.argtypes=[ptr,ptr,integer,real,ptr];fn.restype=integer
    for lo in range(0,len(items),80):
        if time.monotonic()+.2>=deadline:break
        v=items[lo:lo+80]
        qs=[queries(G,x) for x in v]
        if not any(map(len,qs)):continue
        q=np.concatenate(qs)
        response=np.zeros(len(q),VALUE)
        count=fn(sky.handle,q.ctypes.data,len(q),max(.02,deadline-time.monotonic()),response.ctypes.data)
        if count<0:break
        at=0
        for j,x in enumerate(v):
            out=response[at:at+len(qs[j])];at+=len(out)
            out=out[out['pairs']>0]
            if not len(out):continue
            same=np.clip(out['same']-out['any'],-12.,0.)
            one=np.clip(out['one']-out['any'],-12.,0.)
            if len(x['take'])==5:penalty=same
            else:penalty=.36*same+.64*one
            # These tests share observations. The most ambiguous geometry
            # and the median share a single capped contribution.
            adj[lo+j]=.57*np.min(penalty)+.43*np.median(penalty)
            c=x['candidate']['_report']
            c['alternative_pair_odds']=round(float(adj[lo+j]),3)
    return adj
