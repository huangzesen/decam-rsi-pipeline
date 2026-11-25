"""Global continuation of admissible three-night orbital hypotheses."""
import os,time
import numpy as np
import torch
from chain_search import photometric_chi
import orbital_priority
from auxiliary_completion import catalogue

LIMIT=int(os.environ.get('FINDER_LONGCAP','50000'))

def collect(F,p,x,ref,r):
 G=F.G
 if not len(p):return
 t=G['et'][G['de'][p]];span=t.max(1).values-t.min(1).values
 chi=photometric_chi(F,p)
 good=(span>=180)&(r<.5)&(chi<6)
 if not bool(good.any()):return
 p,x,ref,r,chi=[v[good] for v in (p,x,ref,r,chi)]
 sn=G['snr_t'][p].double().clamp(min=4.5)
 rexp=(((1.1/sn)**2+.04**2).mean(1)).sqrt()
 q=-.5*(r/rexp)**2-.5*chi
 physical=q+orbital_priority.log_prior(F,x)
 G.setdefault('long_triangle_parts',[]).append(tuple(v.cpu().numpy() for v in (p,x,ref,r,q,physical)))
 G['long_triangle_count']=G.get('long_triangle_count',0)+len(p)
 if G['long_triangle_count']>100000:compact(F)

def compact(F):
 G=F.G;parts=G.get('long_triangle_parts',[])
 if not parts:return
 a=[np.concatenate([v[k] for v in parts]) for k in range(6)]
 _,ids=np.unique(np.sort(a[0],axis=1),axis=0,return_index=True)
 if len(ids)>LIMIT:ids=ids[orbital_priority.reserve(a[4][ids],a[5][ids],LIMIT)]
 G['long_triangle_parts']=[tuple(v[ids] for v in a)];G['long_triangle_count']=len(ids)

def complete(F,deadline):
 G=F.G;dev=G['dev'];compact(F);parts=G.pop('long_triangle_parts',[]);G.pop('long_triangle_count',None)
 if not parts or time.time()>=deadline:return []
 p,x,ref,r,q,physical=parts[0];order=orbital_priority.order(q,physical,LIMIT)
 p,x,ref,r=[v[order] for v in (p,x,ref,r)]
 start=time.time();result=[];done=0;last=0.;nk=G['nk'];ne=G['night_e_np']
 G['n3']=max(G.get('n3',0),len(order));G.setdefault('bridge_trace',[]).extend(G['keep_np'][p].tolist())
 with catalogue(F):
  batch=128 if dev=='cpu' else 1024
  for lo in range(0,len(p),batch):
   if time.time()+.7*last>=deadline:break
   bt=time.time();pp=torch.as_tensor(p[lo:lo+batch],device=dev);xx=torch.as_tensor(x[lo:lo+batch,[0,1,4,2,3,5]],device=dev).clone()
   refs=torch.as_tensor(ref[lo:lo+batch],device=dev)
   xx[:,:2]+=xx[:,3:5]/(F.C_AU*xx[:,2:3])
   mb=torch.full((len(pp),len(G['et'])),-1,dtype=torch.long,device=dev);mb.scatter_(1,G['de'][pp],pp)
   core=mb.clone();xx,mb,_=F.cfit(xx,mb,refs,10)
   chance=torch.full((len(pp),),float('inf'),dtype=torch.float64,device=dev)
   xx,mb,_=F.covariance_extend(xx,mb,refs,deadline,seed_n=3,chance_out=chance)
   xx,mb,res=F.cfit(xx,mb,refs,5,clip=.7)
   retained=((core<0)|(mb==core)).all(1).cpu().numpy()
   for j,(st,mm,rs,fa) in enumerate(zip(xx.cpu().numpy(),mb.cpu().numpy(),res.cpu().numpy()/F.AS,chance.cpu().numpy())):
    ee=np.flatnonzero(mm>=0);dd=mm[ee];nn=len(np.unique(ne[ee]));err=float(np.sqrt(np.mean(rs[ee]**2)))
    if not(retained[j] and nn>=5 and np.isfinite(fa) and err<.35 and np.isfinite(st).all() and st[2]>0):continue
    if np.sum(st[3:6]**2)*np.sqrt(1+np.sum(st[:2]**2))>=1.99*F.MU*st[2]**3:continue
    result.append(dict(det=dd,exps=ee,res=rs[ee],x=st,tref=float(ref[lo+j]),nn=nn,src=4,sn=3,srms=float(r[lo+j]),fa=max(float(fa),1e-12),n0=nn,nax=int((dd>=nk).sum()),nn0=len(np.unique(ne[ee[dd<nk]]))))
   done+=len(pp);last=time.time()-bt
 G.setdefault('aux_completed',[]).extend(result)
 F.log('global triangle completion:',done,'of',len(p),'hypotheses,',len(result),'verified;',round(time.time()-start,1),'s')
 return result
