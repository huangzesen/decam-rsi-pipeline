"""Estimate the foreground work and loss of true branches for a visit triple."""
import numpy as np
from itertools import permutations
from scipy.special import ndtri
from scipy.stats import qmc

import quartet_geometry as orbit


def areas(position,err):
    # Successive 2-D Schur complements give the area in which an unrelated
    # measurement can outrank the correct one after earlier positions are known.
    n,s,k,_=position.shape
    relative=position-position[:,:,:1]
    z=relative[:,:,1:].reshape(n,s,-1)
    z-=z.mean(1,keepdims=True)
    cov=z.transpose(0,2,1)@z/(s-1)
    out=[]
    for j in range(k-1):
        jj=2*j
        for axis in range(2):
            coord=jj+axis
            cov[:,coord,axis::2]+=err[:,0,None]**2
            cov[:,coord,coord]+=err[:,j+1]**2+.11**2
    L=np.linalg.cholesky(cov)
    for j in range(k-1):
        out.append(L[:,2*j,2*j]*L[:,2*j+1,2*j+1])
    return np.stack(out,axis=1)


def order(cat,plans,pole):
    if len(plans)<60:return plans
    visits=cat.nights.visits
    ob=np.column_stack([visits[k] for k in ('ox','oy','oz')])
    density=cat.fields['density'][:len(ob)]
    nsource=np.array(list(map(len,cat.roots)))
    strength=1.28*visits['depth']-np.log(np.maximum(25,nsource))
    # Use only the simulated head when the field contains very many starts.
    # The rest retain the original survey-coverage ordering.
    size=min(45000,len(plans))
    keys=np.asarray([k for k,r in plans[:size]])
    a,b,c,f=keys.T
    anchor=np.where(strength[a]>=strength[b],a,b)
    second=np.where(anchor==a,b,a)
    last=np.asarray([r[:2] for k,r in plans[:size]])
    views=np.column_stack((anchor,second,c,last))
    z=ndtri(np.clip(qmc.Sobol(6,scramble=True,seed=1427).random_base2(5),.004,.996))
    score=np.zeros(size)
    directed=views[:,:3].copy()
    for lo in range(0,size,512):
        sl=slice(lo,lo+512);e=views[sl];home=e[:,0]
        family=f[sl]
        xy=cat.center[home,None,:]+orbit.AS*np.zeros((1,len(z),2))
        states=orbit.directions(xy,*orbit.variation(z[None,:,:],family[:,None]),
                                ob[home,None],pole)
        t=visits['t'][e]
        pos=orbit.apparent(*states,(t-t[:,:1])[:,None,:],ob[e][:,None])/orbit.AS
        jitter=np.hypot(.24,cat.nights.spread[e]*orbit.TRANSPORT_RATE[family,None]*7.7)
        projected=areas(pos,jitter)
        rho=projected*density[e[:,1:]]
        # The four-dimensional area of two displacement measurements is
        # invariant under interchanging the three nights. Dividing it by
        # the two-dimensional area for a proposed pair gives the conditional
        # area of the third visit. All six orders can therefore be costed
        # using the same trajectories and local source counts.
        volume=projected[:,0]*projected[:,1]
        pair_area={(0,1):projected[:,0]}
        for h,j in ((0,2),(1,2)):
            delta=pos[:,:,j]-pos[:,:,h]
            delta-=delta.mean(1,keepdims=True)
            cov=delta.transpose(0,2,1)@delta/(delta.shape[1]-1)
            variance=jitter[:,h]**2+jitter[:,j]**2+.11**2
            cov[:,0,0]+=variance;cov[:,1,1]+=variance
            pair_area[h,j]=np.sqrt(np.maximum(1.e-12,np.linalg.det(cov)))
        best=np.full(len(e),-np.inf)
        for h,j,k in permutations(range(3)):
            ab=pair_area[min(h,j),max(h,j)]
            competitors=ab*density[e[:,j]]
            next_competitors=volume/ab*density[e[:,k]]
            # A useful displacement must survive the measured-pair list.
            # Bright time-limited roots and an original-catalogue third
            # measurement in a compact ellipse each improve completeness.
            q=-np.log1p(.56*competitors)-.82*np.log1p(.56*next_competitors)
            q+=.30*(visits['depth'][e[:,h]]-visits['depth'][e[:,k]])
            q-=.28*np.log(np.maximum(30.,nsource[e[:,h]])/3000.)
            q+=.22*(1./(1.+next_competitors))
            if h==0 and j==1:q+=.30
            better=q>best
            best[better]=q[better]
            directed[sl][better]=e[better][:,[h,j,k]]
        # The nearest competitors occupy a few positional standard deviations
        # and a small fraction of the image's full magnitude distribution.
        # A query searches at most eight second and five third measurements.
        survival=-np.log1p(.35*rho[:,0]/8.)-np.log1p(.35*rho[:,1]/5.)
        survival-=.46*np.log1p(.32*rho[:,2]/5.)+.55*np.log1p(.32*rho[:,3]/2.)
        foreground=nsource[home]*(.35+np.minimum(7.,1.5*rho[:,0]))*(1.+.30*np.minimum(4.,rho[:,1]))
        score[sl]=np.maximum(-6.5,survival)-.43*np.log1p(foreground/3000.)
    index=np.arange(size)
    priority=.65*score-np.log1p(index/16.)
    ranked=np.argsort(-priority,kind='stable')
    # Revisit the original coverage ordering throughout, including unusually
    # wide or inclined families whose informativeness is less Gaussian.
    extra=np.empty(size+int(np.ceil(size/2)),dtype='i4')
    explore=np.arange(len(extra))%3==0
    extra[explore]=index[:explore.sum()]
    extra[~explore]=ranked[:len(extra)-explore.sum()]
    _,first=np.unique(extra,return_index=True)
    selected=extra[np.sort(first)]
    remaining=np.setdiff1d(index,selected,assume_unique=False)
    indices=np.r_[selected,remaining]
    # Keep the coverage simulation's original ordering of the three visits
    # on part of the schedule, including poor Gaussian approximations on
    # the longest arcs.
    directed[::4]=views[::4,:3]
    cat.directed_starts=True
    ordered=[((*map(int,directed[i]),int(f[i])),plans[i][1]) for i in indices]
    for (a,b,c,family),r in plans[size:]:
        if strength[b]>strength[a]:a,b=b,a
        ordered.append(((a,b,c,family),r))
    return ordered
