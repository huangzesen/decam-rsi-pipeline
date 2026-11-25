"""Find independent nights close together in rotating heliocentric coordinates."""
import ctypes
from pathlib import Path
import time

import numpy as np

from companion_graph import build_library
from return_mesh import distinct, ptr, integer, real, SEED

SOURCE=np.dtype([(k,'f8') for k in ('ux','uy','uz')]+[(k,'f4') for k in ('mag','q')]+
                [('row','i4'),('ex','i4')],align=True)
VIEW=np.dtype([(k,'f8') for k in ('t','ox','oy','oz')]+
              [(k,'f8',3) for k in ('g','c','s')]+[('night','i4'),('pad','i4')],align=True)


def sources(solver,target,limit):
    p=solver.points
    if len(p)>limit:
        p=p[np.argpartition(-p['quality'],limit-1)[:limit]]
    rows=p['row'];t=target[rows];s=np.empty(len(t),SOURCE)
    s['uz']=1./np.sqrt(1.+t['x'].astype(float)**2+t['y'].astype(float)**2)
    s['ux'],s['uy']=t['x']*s['uz'],t['y']*s['uz']
    s['mag'],s['q'],s['row'],s['ex']=t['mag'],p['quality'],rows,t['exposure']
    return s[np.argsort(s['ex'],kind='stable')]


def geometry(image,params):
    radius,angle,speed,radial,epoch,*_=params
    v=np.zeros(len(image),VIEW)
    for k in ('t','ox','oy','oz','night'):v[k]=image[k]
    for j,(r,h) in enumerate(((radius,radial),(radius*np.exp(.001),radial),(radius,radial+.001))):
        t=(image['t']-epoch)*.01720209895/r**1.5
        # First-order radial and along-track response to a perturbation
        # of the velocity on a circular orbit (Hill's equations).
        v['g'][:,j]=1./(r*(1.+h*np.sin(t)+2.*(speed-1.)*(1.-np.cos(t))))
        theta=t+(speed-1.)*(4.*np.sin(t)-3.*t)-2.*h*(1.-np.cos(t))
        v['c'][:,j]=np.cos(theta);v['s'][:,j]=-np.sin(theta)
    return v


class Guides:
    def __init__(self,solver,target,limit,device):
        self.src=sources(solver,target,limit)
        self.solver=solver
        self.lib=build_library('rotating_search')
        self.lib.guide_host.argtypes=[ptr,integer,ptr,ptr,real,real,ptr,ptr]
        self.lib.guide_host.restype=None
        self.lib.guide_groups.argtypes=[ptr,ptr,ptr,ptr,ptr,ptr,ptr,integer,ptr,integer,ptr,real]
        self.lib.guide_groups.restype=integer
        self.lib.drift_groups.argtypes=self.lib.guide_groups.argtypes+[real,real]
        self.lib.drift_groups.restype=integer
        self.kernel=None
        if str(device).startswith('cuda'):
            try:
                import torch
                from apparent_backend import cuda_kernel
                self.kernel=cuda_kernel(Path(__file__).with_suffix('.cpp').read_text(),'guide_cells')
                self.source_device=torch.as_tensor(self.src.view('u1'),device='cuda')
            except Exception as exc:
                self.kernel=None
                print('rotating heliocentric coordinates on host:',str(exc)[:150],flush=True)

    def host(self,src,vis,param,px,py):
        xy=np.empty((len(src),6),'f4');keys=np.empty(len(src),'u8')
        self.lib.guide_host(src.ctypes.data,len(src),vis.ctypes.data,param.ctypes.data,
                            px,py,xy.ctypes.data,keys.ctypes.data)
        return xy,keys

    def groups(self,param,seconds,cap,drift=False):
        n=len(self.src);vis=geometry(self.solver.image,param)
        east=self.solver.east
        az=param[1]+np.arctan2(east[1],east[0])
        px,py=float(-np.sin(az)),float(np.cos(az))
        if self.kernel is not None:
            try:
                import torch
                xy_device=torch.empty((n,6),dtype=torch.float32,device='cuda')
                keys_device=torch.empty(n,dtype=torch.int64,device='cuda')
                self.kernel(grid=((n+127)//128,1,1),block=(128,1,1),
                    args=[self.source_device,n,torch.as_tensor(vis.view('u1'),device='cuda'),
                          torch.as_tensor(param,device='cuda'),px,py,xy_device,keys_device])
                xy=xy_device.cpu().numpy()
                if not getattr(self,'checked',False):
                    count=min(384,n);values,expected=self.host(self.src[:count],vis,param,px,py)
                    if (not np.array_equal(expected,keys_device[:count].cpu().numpy().view('u8'))
                            or not np.allclose(values,xy[:count],atol=.03,rtol=1.e-5)):
                        raise ArithmeticError('guide coordinates differ on device')
                    self.checked=True
                keys_device,order_device=torch.sort(keys_device,stable=True)
                ordered=keys_device.cpu().numpy();order=order_device.int().cpu().numpy()
            except Exception as exc:
                self.kernel=None
                print('using host guide coordinates:',str(exc)[:140],flush=True)
        if self.kernel is None:
            xy,keys=self.host(self.src,vis,param,px,py)
            order=np.argsort(keys,kind='stable').astype('i4')
            ordered=keys[order]
        start=np.flatnonzero(np.r_[True,ordered[1:]!=ordered[:-1]])
        length=np.diff(np.r_[start,n]);ok=length>=3
        start=np.ascontiguousarray(start[ok],'i4');length=np.ascontiguousarray(length[ok],'i4')
        # If the estimate expires, cover different parts of the field on
        # each distance trial rather than the same prefix of the sky.
        rank=np.argsort((start.astype('u8')*2654435761+int(param[0]*121))%1000003,kind='stable')
        start,length=start[rank],length[rank]
        sizes=np.zeros(2,'i4');out=np.empty((2,cap),SEED)
        args=(self.src.ctypes.data,vis.ctypes.data,param.ctypes.data,xy.ctypes.data,order.ctypes.data,
              start.ctypes.data,length.ctypes.data,len(start),out.ctypes.data,cap,sizes.ctypes.data,seconds)
        if drift:
            done=self.lib.drift_groups(*args,*map(float,east))
        else:
            done=self.lib.guide_groups(*args)
        if done<0:raise RuntimeError('rotating coordinate associations failed')
        return out[0,:sizes[0]].copy(),out[1,:sizes[1]].copy(),done


def guide_search(solver,target,seconds,device,event=None,limit=1250000,cap=55000):
    end=time.monotonic()+seconds
    solver=Guides(solver,target,limit,device)
    # Low-discrepancy offsets cover radius, pole, phase and eccentricity.
    # Each cycle also includes slow, distant guides and an inclined family.
    radii=[44.,38.,47.,32.,58.,42.,86.,36.,52.,27.,165.,690.]
    widths=[100.,180.,72.,140.,230.,95.]
    phases=[0.,.16,-.16,.34,-.34,.62,-.62]
    epoch=np.median(solver.solver.image['t'])
    parts=[[np.empty(0,SEED),np.empty(0,SEED)] for _ in range(3)]
    batch=[[[],[]] for _ in range(3)];stats=[0,0]
    def collect(mode):
        for j in range(2):
            if batch[mode][j]:
                parts[mode][j]=distinct(np.concatenate([parts[mode][j],*batch[mode][j]]),cap)
                batch[mode][j].clear()
    for i in range(240):
        remain=end-time.monotonic()
        if remain<.16 or (event is not None and event.is_set()):break
        inclined=i%7==4
        disk_step=i-(i+2)//7
        drift=disk_step%4==2
        step=disk_step//4 if drift else disk_step-(disk_step+1)//4
        k=step%len(radii);cycle=step//len(radii)
        radius=radii[k]*np.exp(.052*np.sin(2.40*cycle+.57*k)*min(1,cycle))
        angle=phases[(cycle+([0,1,2,0,3,4,0,2,5,1,4,6][k]))%len(phases)]
        speed=1.+.090*np.sin(1.62*cycle+1.18*k)*min(1,cycle+(k%4!=0))
        radial=.15*np.sin(1.13*k+2.2*cycle)*min(1,cycle)
        width=widths[(k+cycle)%len(widths)]
        window=(2400.,1650.,3800.,940.)[(cycle+k)%4]
        if drift:
            width=(420.,820.,1450.)[step%3]
            window=(2400.,4100.)[step%2]
        if inclined:
            j=i//7
            radius=(31.,43.,65.,96.,163.,265.)[j%6]*np.exp(.04*np.sin(1.9*j))
            angle=np.pi*np.sin(2.39*j+.7)
            speed=.94+.11*np.sin(j);radial=.17*np.sin(2.*j)
            width=(360.,760.,1450.)[j%3];window=4500.
            drift=bool(j%2)
        param=np.array([radius,angle,speed,radial,epoch,width,(i*.618)%1.,
                        (i*.414)%1.,(i*.732)%1.,window])
        tri,quad,done=solver.groups(param,min(remain,1.65),max(1000,cap//4),drift=drift)
        stats[0]+=1;stats[1]+=done
        mode=2 if inclined else int(drift)
        batch[mode][0].append(tri);batch[mode][1].append(quad)
        if len(batch[mode][0])>=4:collect(mode)
    for mode in range(3):collect(mode)
    finished=[]
    for j in range(2):
        narrow,wide,tilted=parts[0][j],parts[1][j],parts[2][j]
        data=np.concatenate((narrow,wide,tilted))
        # The proposal likelihoods integrate different sky volumes.
        # Retain all three orderings when verification ends early.
        rank=np.r_[np.arange(len(narrow)),1.+2.*np.arange(len(wide)),3.+5.*np.arange(len(tilted))]
        chosen=data[np.argsort(rank,kind='stable')]
        _,ix=np.unique(np.sort(chosen['rows'],axis=1),axis=0,return_index=True)
        finished.append(chosen[np.sort(ix)[:cap]])
    return finished,stats
