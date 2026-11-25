"""Project broad range intervals as several smaller sky rectangles."""
import torch

def queries(F,theta,observer,times,reference,g0,g1,uncertainty,elite,
            epochs,observers,rotation,padnight,potential,low,high,width,box):
    G=F.G;AS=F.AS;dev=G['dev']
    ci,ni=torch.nonzero(potential,as_tuple=True)
    if not len(ci):return ci,ni,low[ci,ni],high[ci,ni],0
    # Only boxes intersecting the observed sky can add a constraint. Bounds
    # are deliberately inclusive, as in the catalogue's ordinary box query.
    bounds=box.bounds[ni.clamp(max=len(box.bounds)-1)]
    c0=torch.floor((low[ci,ni]-box.lo_t)/box.S).long()
    c1=torch.floor((high[ci,ni]-box.lo_t)/box.S).long()
    covered=(ni<len(box.bounds))&((c1>=bounds[:,:2])&(c0<=bounds[:,2:])).all(1)
    ci,ni=ci[covered],ni[covered]
    w=width[ci,ni]
    split=(w>=16000.)&~elite[ci]
    ordinary=~split&(w<16000.)
    owners=[ci[ordinary]];nights=[ni[ordinary]]
    lows=[low[ci[ordinary],ni[ordinary]]];highs=[high[ci[ordinary],ni[ordinary]]]
    ci,ni,w=ci[split],ni[split],w[split]
    if not len(ci):return owners[0],nights[0],lows[0],highs[0],len(owners[0])
    # The original four-degree width cap rejected these queries entirely.
    # Bound additional work to eight or sixteen smaller range intervals.
    needed=torch.where(w<=19200.,8,16)
    for ns in (8,16):
        sel=needed==ns
        a,b=ci[sel],ni[sel]
        if not len(a):continue
        frac=torch.arange(ns+1,dtype=torch.float64,device=dev)/ns
        span=(g1[a]-g0[a])[:,None]
        lower=(g0[a,None]+span*frac[None,:-1]).reshape(-1)
        upper=(g0[a,None]+span*frac[None,1:]).reshape(-1)
        a=a.repeat_interleave(ns);b=b.repeat_interleave(ns)
        gs=torch.stack((lower,lower,upper,upper),1)
        rd=.4*torch.sqrt(F.MU*gs**3)*torch.tensor([-1.,1.,-1.,1.],device=dev)
        tau=times[a]-reference[a,None];mg=F.MU*gs**3
        ca=1-.5*mg[:,:,None]*tau[:,None]**2
        sa=tau[:,None]-mg[:,:,None]*tau[:,None]**3/6
        ph=theta[a,None]*(ca+rd[:,:,None]*tau[:,None]-gs[:,:,None]*observer[a,None,:,2])[...,None]+gs[:,:,None,None]*observer[a,None,:,:2]
        den=ca[:,:,0]*sa[:,:,1]-ca[:,:,1]*sa[:,:,0]
        pos=(ph[:,:,0]*sa[:,:,1,None]-ph[:,:,1]*sa[:,:,0,None])/den[:,:,None]
        rate=(ca[:,:,0,None]*ph[:,:,1]-ca[:,:,1,None]*ph[:,:,0])/den[:,:,None]
        t=epochs[b]-reference[a]
        c=1-.5*mg*t[:,None]**2+.5*mg*rd*t[:,None]**3
        ss=t[:,None]-mg*t[:,None]**3/6
        rf=1-.5*mg*t[:,None]**2+(mg*rd/3)*t[:,None]**3+rd*t[:,None]
        pred=(c[:,:,None]*pos+ss[:,:,None]*rate-gs[:,:,None]*observers[b,None,:2])/(rf-gs*observers[b,None,2])[:,:,None]
        pred=pred@rotation.T
        pad=padnight[b]+t.abs()*uncertainty[a]+.01*torch.sqrt(F.MU*upper**3)**3*t.abs()**3
        lo=pred.min(1).values-pad[:,None];hi=pred.max(1).values+pad[:,None]
        valid=torch.isfinite(lo).all(1)&torch.isfinite(hi).all(1)&((hi-lo).max(1).values<16000*AS)
        owners.append(a[valid]);nights.append(b[valid]);lows.append(lo[valid]);highs.append(hi[valid])
    return tuple(torch.cat(v) for v in (owners,nights,lows,highs))+(len(owners[0]),)
