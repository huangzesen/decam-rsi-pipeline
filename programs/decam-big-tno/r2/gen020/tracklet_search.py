"""Resolved within-night motion followed by sequential orbital confirmation.

The initial pair must move significantly relative to its centroid errors.
Physical rate bounds and per-night quotas control the catalogue size. A later
night supplies an orbital hypothesis; three more independent nights must be
associated with their full background trial probabilities before reporting it.
"""
import time,math,os
import numpy as np
import torch
from tracklet_covariance import extend

def capture(F,deadline):
 G=F.G;dev=G['dev'];AS=F.AS
 ids=np.flatnonzero(G['seed2_np'])
 if not len(ids):return None
 d=G['de_np'][ids];night=G['night_e_np'][d]
 bx=F.BoxIndex(G['th_np'][ids],night,np.eye(2),8*AS,dev)
 allids=torch.as_tensor(ids,device=dev);sn=G['snr_t'];ne=G['night_e'];et=G['et'];rows=[]
 anchors=ids[G['snr_np'][ids]>=4.5]
 for lo in range(0,len(anchors),4096):
  if time.time()>deadline:break
  a=torch.as_tensor(anchors[lo:lo+4096],device=dev);ex=G['de'][a];pt=G['th'][a]
  qi,hi=bx.query(ne[ex],pt-36*AS,pt+36*AS,maxc=12)
  if not len(qi):continue
  aa=a[qi];bb=allids[hi];ea=G['de'][aa];eb=G['de'][bb];dt=(et[eb]-et[ea]).abs()
  sep=(G['th'][bb]-G['th'][aa]).norm(dim=1)/AS
  sig=torch.sqrt((1.1/sn[aa].double())**2+(1.1/sn[bb].double())**2+2*.04**2)
  ok=(dt>=.005)&(dt<=.15)&(sep>3.5*sig)&(sep<250*dt+2*sig)&((G['mag'][aa]-G['mag'][bb]).abs()<.5+.3*(G['lev'][aa]-G['lev'][bb]).abs())
  aa=aa[ok];bb=bb[ok]
  if len(aa):rows.append(torch.stack([aa,bb],1).cpu().numpy())
 if not rows:return None
 p=np.concatenate(rows);p=np.sort(p,axis=1);p=np.unique(p,axis=0)
 ex=G['de_np'][p];tt=G['et_np'][ex];p=np.take_along_axis(p,np.argsort(tt,axis=1),axis=1);tt=np.sort(tt,axis=1)
 # Reject rates that cannot come from a bound outer-system orbit before
 # allocating a tracklet slot. This also removes most fast foreground movers.
 accept=[]
 grid=torch.linspace(.01,.04,49,dtype=torch.float64,device=dev);nc=torch.sqrt(F.MU*grid**3)
 for lo in range(0,len(p),16384):
  pp=torch.as_tensor(p[lo:lo+16384],device=dev);ex=G['de'][pp];tt=et[ex];delta=tt[:,1]-tt[:,0]
  th=G['th'][pp];oo=G['ob'][ex];dd=oo[...,:2]-th*oo[...,2:3]
  w0=(th[:,1]-th[:,0])/delta[:,None];w1=(dd[:,1]-dd[:,0])/delta[:,None]
  vel=w0[:,None]+grid[None,:,None]*w1[:,None];speed=vel.norm(dim=-1)
  noise=2*torch.sqrt(((1.1/sn[pp].double())**2+.04**2).sum(1))*AS/delta
  margin=noise[:,None]+.5*float(grid[1]-grid[0])*w1.norm(dim=1)[:,None]
  good=(speed>=.2*nc[None]-margin)&(speed<=1.5*nc[None]+margin)&((vel@G['epro'])>=.2*speed-margin)
  accept.append(good.any(1).cpu().numpy())
 valid=np.concatenate(accept);p=p[valid];tt=np.sort(G['et_np'][G['de_np'][p]],axis=1)
 snp=G['snr_np'][p];sigma=np.sqrt((1.1/snp)**2+.04**2).sum(1)/(tt[:,1]-tt[:,0])
 used={};keep=[]
 for i in np.argsort(sigma):
  a,b=p[i]
  if max(used.get(int(a),0),used.get(int(b),0))>=2:continue
  used[int(a)]=used.get(int(a),0)+1;used[int(b)]=used.get(int(b),0)+1;keep.append(i)
 p=p[keep];sigma=sigma[keep]
 # The densest repeated night must not consume every slot. Redistribute spare
 # quotas after guaranteeing representation to each observed night.
 nights=G['night_e_np'][G['de_np'][p[:,0]]];un=np.unique(nights)
 groups=[np.flatnonzero(nights==n) for n in un];sizes=np.array([len(v) for v in groups]);quota=np.zeros(len(groups),int);left=min(len(p),20000)
 while left:
  active=np.flatnonzero(sizes>quota)
  if not len(active):break
  step=max(1,left//len(active))
  for j in active:
   add=min(int(sizes[j]-quota[j]),step,left);quota[j]+=add;left-=add
   if not left:break
 selected=np.concatenate([v[np.argsort(sigma[v])[:q]] for v,q in zip(groups,quota) if q]) if len(p) else np.empty(0,int)
 p=p[selected];F.log('resolved tracklets:',len(p),'of',len(sigma),'night pairs')
 return p

def _verify(F,P,gs,qs,deadline,rr,long_mode=True):
 G=F.G;dev=G['dev'];AS=F.AS;nE=len(G['et']);et=G['et'];ob=G['ob'];ne=G['night_e'];out=[]
 order=np.argsort(qs);P=P[order];gs=gs[order]
 for lo in range(0,len(P),256 if dev=='cpu' else 2048):
  if time.time()>deadline:break
  core=torch.as_tensor(P[lo:lo+(256 if dev=='cpu' else 2048)],device=dev)
  gg=torch.linspace(.01,.04,16,dtype=torch.float64,device=dev)
  pp=core.repeat_interleave(len(gg),dim=0);g=gg.repeat(len(core));ee=G['de'][pp];tt=et[ee];ref=tt.mean(1);dt=tt-ref[:,None];th=G['th'][pp];oo=ob[ee];D=oo[...,:2]-th*oo[...,2:3]
  phi=th+g[:,None,None]*D;v=(phi[:,-1]-phi[:,0])/(tt[:,-1]-tt[:,0])[:,None];pos=phi.mean(1)+v/(F.C_AU*g[:,None]);x=torch.cat([pos,g[:,None],v,torch.zeros_like(g[:,None])],1)
  x,res=F.kfit(x,dt,oo,th,torch.ones_like(dt),iters=8,free=(0,1,3,4,5))
  ok=(res.max(1).values<.6*AS)&(x[:,2]>.009)&(x[:,2]<.044)&(x[:,3:6].square().sum(1)<1.99*F.MU*x[:,2]**3)
  quality=torch.where(ok,res.square().sum(1),torch.inf).reshape(len(core),len(gg))
  best=quality.argmin(1);selected=torch.arange(len(core),device=dev)*len(gg)+best;selected=selected[torch.isfinite(quality.min(1).values)]
  x=x[selected];pp=pp[selected];ee=ee[selected];ref=ref[selected]
  if not len(x):continue
  mb=torch.full((len(x),nE),-1,dtype=torch.int64,device=dev);mb.scatter_(1,ee,pp);chance=torch.full((len(x),),float('inf'),dtype=torch.float64,device=dev)
  x,mb,n=extend(F,x,mb,ref,deadline,seed_n=2,chance_out=chance)
  if not n:continue
  x,mb,res=F.cfit(x,mb,ref,6,clip=.7)
  for xx,mm,rs,r,fa in zip(x.cpu().numpy(),mb.cpu().numpy(),res.cpu().numpy()/AS,ref.cpu().numpy(),chance.cpu().numpy()):
   ee=np.flatnonzero(mm>=0);nn=len(np.unique(G['night_e_np'][ee]));rms=np.sqrt(np.mean(rs[ee]**2))
   if nn<5 or rms>=.35 or not np.isfinite(fa):continue
   if (mm[ee]>=G['nk']).any():
    G.setdefault('aux_completed',[]).append(dict(det=mm[ee],exps=ee,res=rs[ee],x=xx,tref=float(r),nn=nn,src=4,sn=3,srms=.1,fa=float(fa),n0=nn,nax=int((mm[ee]>=G['nk']).sum()),nn0=len(np.unique(G['night_e_np'][ee[mm[ee]<G['nk']]]))))
   else:
    out.append(dict(det=mm[ee],nights=nn,rms=float(rms),g=float(xx[2]),tref=float(r),st=xx[[0,1,3,4]].tolist(),gd=float(xx[5]),h=90.,src=4,sn=3,srms=.1,fa=float(fa),n0=nn))
 return out


def verify(F,*args,**kwargs):
 from auxiliary_completion import catalogue
 with catalogue(F):return _verify(F,*args,**kwargs)


def search(F,deadline):
    G=F.G;dev=G['dev'];AS=F.AS;started=time.time()
    units=capture(F,time.time()+.18*max(deadline-time.time(),0))
    if units is None:return []
    p=units
    de=G['de_np'];t=G['et_np'][de];sn=G['snr_np'];mag=G['mag_np'];lev=G['lev_np']
    dt=t[p[:,1]]-t[p[:,0]]
    # Resolved motion within a night constrains the admissible distance range.
    good=(dt>=.005)&(dt<=.15)
    p=p[good];dt=dt[good]
    if not len(p):return []
    pairmag=mag[p];pairlev=lev[p]
    sig2=.2**2+(1.09/np.maximum(sn[p],4.5))**2
    dm=np.maximum(np.abs(pairmag[:,1]-pairmag[:,0])-.3*np.abs(pairlev[:,1]-pairlev[:,0]),0)
    quality=dm**2/sig2.sum(1)-.5*np.log(np.clip(np.min(sn[p],axis=1),4.5,10.)/4.5)
    # A deterministic breadth order inside quality bands avoids spending the
    # deadline on a single observing season or on one exposure's pair list.
    jitter=((p[:,0]*2654435761+p[:,1]*2246822519)&0xffff)/65536
    quality=np.floor(quality*4)+jitter
    order=np.argsort(quality,kind='stable');p=p[order];dt=dt[order]
    F.log('tracklet continuation:',len(p),'two-night arcs')
    # A night catalogue avoids repeatedly querying adjacent images of the same
    # field. Actual exposure times remain attached to every matched row.
    single=np.flatnonzero(G['seed2_np'])
    snight=G['night_e_np'][de[single]]
    nt=len(G['nights_u']);et=G['et'];ob=G['ob'];ne=G['night_e']
    te=np.array([np.mean(G['et_np'][G['night_e_np']==n]) for n in range(nt)])
    oe=np.array([ob[ne==n].mean(0).cpu().numpy() for n in range(nt)])
    spread=np.array([np.ptp(G['et_np'][G['night_e_np']==n]) for n in range(nt)])
    box=F.BoxIndex(G['th_np'][single],snight,np.eye(2),128*AS,dev)
    te=torch.as_tensor(te,device=dev);oe=torch.as_tensor(oe,device=dev)
    padnight=torch.as_tensor(spread,device=dev)*120*AS+4*AS
    sp=torch.as_tensor(single,device=dev)
    grid=torch.linspace(.01,.04,49,dtype=torch.float64,device=dev)
    step=float(grid[1]-grid[0]);ncgrid=torch.sqrt(F.MU*grid**3)
    out=[];seen=set();nqueries=0;ntrip=0;processed=0
    batch=512 if dev!='cpu' else 256
    previous=0.
    for lo in range(0,len(p),batch):
        if time.time()+.8*previous>=deadline:break
        bt=time.time();pp=torch.as_tensor(p[lo:lo+batch],device=dev);ee=G['de'][pp]
        tt=et[ee];ref=tt.mean(1);d=tt[:,1]-tt[:,0]
        theta=G['th'][pp];oo=ob[ee];D=oo[...,:2]-theta*oo[...,2:3]
        w0=(theta[:,1]-theta[:,0])/d[:,None];w1=(D[:,1]-D[:,0])/d[:,None]
        vel=w0[:,None]+grid[None,:,None]*w1[:,None]
        speed=vel.norm(dim=-1);margin=.5*step*w1.norm(dim=1)[:,None]+2*torch.sqrt(((1.1/G['snr_t'][pp].double())**2+.04**2).sum(1))[:,None]*AS/d[:,None]
        # A broad prograde, bound-orbit envelope, expanded by centroid error.
        allowed=(speed>=.20*ncgrid[None]-margin)&(speed<=1.50*ncgrid[None]+margin)
        allowed &= (vel@G['epro'])>=.2*speed-margin
        g0=torch.where(allowed,grid[None],torch.inf).min(1).values-step*.5
        g1=torch.where(allowed,grid[None],-torch.inf).max(1).values+step*.5
        valid=torch.isfinite(g0)&torch.isfinite(g1)
        if not valid.any():processed+=len(pp);continue
        pp,ee,tt,ref,d,theta,oo,D,g0,g1=[v[valid] for v in (pp,ee,tt,ref,d,theta,oo,D,g0,g1)]
        g0=g0.clamp(.01,.04);g1=g1.clamp(.01,.04)
        # Four corners of the admissible range/radial-rate rectangle. Solar
        # acceleration is included in both the two-point state and prediction.
        gs=torch.stack([g0,g0,g1,g1],1)
        rd=.35*torch.sqrt(F.MU*gs**3)*torch.tensor([-1.,1.,-1.,1.],device=dev)
        tau=tt-ref[:,None];mg=F.MU*gs**3
        ca=1-.5*mg[:,:,None]*tau[:,None]**2
        sa=tau[:,None]-mg[:,:,None]*tau[:,None]**3/6
        ph=theta[:,None]*(ca+rd[:,:,None]*tau[:,None]-gs[:,:,None]*oo[:,None,:,2])[...,None]+gs[:,:,None,None]*oo[:,None,:,:2]
        den=ca[:,:,0]*sa[:,:,1]-ca[:,:,1]*sa[:,:,0]
        pos=(ph[:,:,0]*sa[:,:,1,None]-ph[:,:,1]*sa[:,:,0,None])/den[:,:,None]
        rate=(ca[:,:,0,None]*ph[:,:,1]-ca[:,:,1,None]*ph[:,:,0])/den[:,:,None]
        target=te[None]-ref[:,None]
        c=1-.5*mg[:,:,None]*target[:,None]**2+.5*mg[:,:,None]*rd[:,:,None]*target[:,None]**3
        ss=target[:,None]-mg[:,:,None]*target[:,None]**3/6
        rf=1-.5*mg[:,:,None]*target[:,None]**2+(mg*rd/3)[:,:,None]*target[:,None]**3+rd[:,:,None]*target[:,None]
        pred=(c[:,:,:,None]*pos[:,:,None]+ss[:,:,:,None]*rate[:,:,None]-gs[:,:,None,None]*oe[None,None,:,:2])/(rf-gs[:,:,None]*oe[None,None,:,2])[:,:,:,None]
        unc=2*torch.sqrt(((1.1/G['snr_t'][pp].double())**2+.04**2).sum(1))*AS/d
        pad=padnight[None]+target.abs()*unc[:,None]
        low=pred.min(1).values-pad[:,:,None];high=pred.max(1).values+pad[:,:,None]
        width=(high-low).max(-1).values/AS
        eligible=(target.abs()>=20)&(target.abs()<=730)&(width<2400)&torch.isfinite(low).all(-1)
        ci,ni=torch.nonzero(eligible,as_tuple=True)
        if not len(ci):processed+=len(pp);continue
        nqueries+=len(ci)
        qi,hit=box.query(ni,low[ci,ni],high[ci,ni],maxc=32)
        if not len(qi):processed+=len(pp);continue
        owners=ci[qi];hd=sp[hit]
        m=G['mag'];lv=G['lev']
        pm=m[pp].mean(1);pl=lv[pp].mean(1)
        dm=(m[hd]-pm[owners]).abs()
        good=dm<.65+.3*(lv[hd]-pl[owners]).abs()
        owners,hd=owners[good],hd[good]
        if not len(hd):processed+=len(pp);continue
        triples=torch.cat([pp[owners],hd[:,None]],1)
        ordert=torch.argsort(et[G['de'][triples]],dim=1)
        triples=torch.gather(triples,1,ordert)
        # Curvature orders the continuation queue. Very short temporal legs
        # leave distance uncertain; bounded trial distances are fitted below.
        es=G['de'][triples];ts=et[es];ths=G['th'][triples];obs3=ob[es]
        ds=obs3[...,:2]-ths*obs3[...,2:3]
        ta=ts[:,0]-ts[:,1];tc=ts[:,2]-ts[:,1];span=tc-ta
        h=tc/span
        A=ths[:,1]-h[:,None]*ths[:,0]-(1-h[:,None])*ths[:,2]
        B=ds[:,1]-h[:,None]*ds[:,0]-(1-h[:,None])*ds[:,2]
        bb=B.square().sum(1).clamp(min=1e-25);g=-(A*B).sum(1)/bb
        for _ in range(3):
            mg=F.MU*g.clamp(.01,.04)**3;ca=1-.5*mg*ta**2;cc=1-.5*mg*tc**2
            ua=(ta-mg*ta**3/6)/ca;uc=(tc-mg*tc**3/6)/cc;h=uc/(uc-ua)
            A=ths[:,1]-h[:,None]*ths[:,0]-(1-h[:,None])*ths[:,2]
            B=ds[:,1]-h[:,None]*ds[:,0]/ca[:,None]-(1-h[:,None])*ds[:,2]/cc[:,None]
            bb=B.square().sum(1).clamp(min=1e-25);g=-(A*B).sum(1)/bb
        C=h[:,None]*ta[:,None]*(ths[:,0]-ths[:,1])+(1-h[:,None])*tc[:,None]*(ths[:,2]-ths[:,1])
        Cp=C-B*((C*B).sum(1)/bb)[:,None];nc=torch.sqrt(F.MU*g.clamp(.01,.04)**3)
        residual=(A+g[:,None]*B).norm(dim=1)/AS
        tolerance=.65+.4*nc*Cp.norm(dim=1)/AS+.005*nc**3*span**3/AS
        velocity=(ths[:,2]+g[:,None]*ds[:,2]-ths[:,0]-g[:,None]*ds[:,0])/span[:,None]
        speed=velocity.norm(dim=1)
        good=torch.isfinite(g)&torch.isfinite(residual)
        g=g.clamp(.01,.04)
        triples=triples[good];g=g[good]
        if len(triples):
            from chain_search import photometric_chi
            quality=(residual[good]/tolerance[good])**2+.2*photometric_chi(F,triples)
            ppn=triples.cpu().numpy();gn=g.cpu().numpy();qn=quality.cpu().numpy()
            ppn,unique=np.unique(ppn,axis=0,return_index=True);gn=gn[unique];qn=qn[unique]
            ntrip+=len(ppn)
            if len(ppn)>12000:
                keep=np.argpartition(qn,11999)[:12000];ppn=ppn[keep];gn=gn[keep];qn=qn[keep]
            seeds=verify(F,ppn,gn,qn,deadline,(25.,100.),long_mode=True)
            for cseed in seeds:
                key=tuple(sorted(cseed['det'].tolist()))
                if key not in seen:out.append(cseed);seen.add(key)
        processed+=len(pp)
        previous=time.time()-bt
        if processed%max(batch*16,1)==0:
            F.log('tracklet continuation progress:',processed,'arcs,',ntrip,'triples,',len(out),'verified seeds')
    F.log('tracklet continuation finished:',processed,'of',len(p),'arcs,',nqueries,'night queries,',ntrip,'triples,',len(out),'seeds;',round(time.time()-started,1),'s')
    return out
