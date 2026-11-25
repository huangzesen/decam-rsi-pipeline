"""Find three or four measured nights in co-moving charts and certify their orbits."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

import moving_votes
from survey_arcs import CERTIFIED,certify


POINT=np.dtype([(s,'f4') for s in ('x','y','mag','quality')]+
               [(s,'i4') for s in ('exposure','row')])
MAP=np.dtype([(s,'f8') for s in ('ox','oy','xx','xy','yx','yy','fg','inverse_f')]+
             [(s,'i4') for s in ('epoch','night')])
QUAD=np.dtype([('rows','i4',4),('g','f4'),('merit','f4')])


def catalogue(index,ids,quality,repeat,secondary):
    score=np.nan_to_num(quality,nan=-12.,neginf=-12.,posinf=8.).astype('f4')
    # Calibrate the rank of a repeat relative to independent foreground
    # detections on this night, preserving sources seen only once.
    score+=np.clip(.55*repeat[:,0],0.,4.)
    if secondary is not None:
        ids=np.concatenate((ids,secondary[1]))
        score=np.concatenate((score,secondary[2].astype('f4')))
    t=index.target[ids]
    valid=np.isfinite(t['x']+t['y']+t['mag']+score)&(t['mag']>0.)
    ids,t,score=ids[valid],t[valid],score[valid]
    if not len(ids):
        return np.empty(0,POINT)
    # Balance visits with very different background densities. Interleave
    # local and global ranks so sparsely exposed sky can initiate a linkage.
    order=np.lexsort((-score,t['exposure']))
    e=t['exposure'][order]
    local=np.empty(len(e),'i4')
    local[order]=np.arange(len(e))-np.maximum.accumulate(np.where(
        np.r_[True,e[1:]!=e[:-1]],np.arange(len(e)),0))
    rank=score-.60*np.log1p(local/100.)
    order=np.argsort(-rank,kind='stable')[:1550000]
    points=np.empty(len(order),POINT)
    for key in ('x','y','exposure','mag'):points[key]=t[key][order]
    points['row']=ids[order]
    points['quality']=np.clip(rank[order]-np.median(rank[order]),-7.,7.)
    return points


def chart(ex,centers,pole,u,j):
    z=ndtri(np.clip(u,.008,.992))
    # Short arcs constrain angular velocity through parallax; at long spans
    # the templated longitude and inclination also separate the measurements.
    family=j%10
    if family in (3,7):
        radius=40.*np.exp(.245*z[0])
        speed=np.clip(.96+.16*z[1],.52,1.35)
        heading=.28*z[2];radial=.18*z[3]
    elif family==9:
        radius=80.*55.**u[0]
        speed=1.24*np.sqrt(u[1]);heading=2.*np.pi*u[2];radial=.30*z[3]
    else:
        radius=44.*np.exp(.094*z[0])
        speed=1.+.062*z[1];heading=.10*z[2];radial=.090*z[3]
    span=(95.+125.*u[4] if family in (1,3,6) else
          410.+570.*u[4] if family in (0,2,5) else 1250.+3000.*u[4])
    if family==9:span=2800.+2200.*u[4]
    w=np.sqrt(moving_votes.MU/radius**3)
    # Shrink the cells for distant orbits and for templates covering a single
    # apparition. Changes of inverse distance also multiply the Earth baseline.
    width=np.clip((.038/radius+.080*w*span)*(.50+.62*u[5])/moving_votes.AS,12.,360.)
    start=ex['t'].min()
    epoch=np.floor((ex['t']-start)/span+u[6]).astype('i4')
    dt=ex['t']-start-(epoch+.5-u[6])*span
    ia=max(.035,2.-speed**2-radial**2)
    mean=w*ia**1.5*dt
    ecc=1.-ia
    vs=radial*np.sqrt(ia)
    anomaly=mean.copy()
    for _ in range(6):
        s,c=np.sin(anomaly),np.cos(anomaly)
        anomaly-=(anomaly-ecc*s+vs*(1.-c)-mean)/(1.-ecc*c+vs*s)
    f=1.-(1.-np.cos(anomaly))/ia
    v=w*dt+(np.sin(anomaly)-anomaly)/ia**1.5
    cs=f+v*radial;ss=v*speed
    dist=np.hypot(cs,ss)
    cs,ss=cs/dist,ss/dist
    r=radius*dist
    ob=np.column_stack([ex[k] for k in ('ox','oy','oz')])
    def warp(pos):
        ray=np.column_stack((pos,np.ones(len(ex))))
        ray/=np.linalg.norm(ray,axis=1)[:,None]
        proj=(ray*ob).sum(1)
        rho=np.sqrt(r*r-(ob*ob).sum(1)+proj*proj)-proj
        direction=(ob+rho[:,None]*ray)/r[:,None]
        east=np.cross(pole,direction)
        east/=np.maximum(1.e-8,np.linalg.norm(east,axis=1))[:,None]
        tangent=np.cos(heading)*east+np.sin(heading)*np.cross(direction,east)
        origin=direction*cs[:,None]-tangent*ss[:,None]
        return origin[:,:2]/origin[:,2,None]/(width*moving_votes.AS)
    base=warp(centers)
    step=.003
    dx=(warp(centers+[step,0])-warp(centers-[step,0]))/(2.*step)
    dy=(warp(centers+[0,step])-warp(centers-[0,step]))/(2.*step)
    zero=base-dx*centers[:,0,None]-dy*centers[:,1,None]+u[7:9]
    maps=np.empty(len(ex),MAP)
    maps['ox'],maps['oy']=zero.T
    maps['xx'],maps['xy']=dx.T
    maps['yx'],maps['yy']=dy.T
    maps['fg'],maps['inverse_f']=v/(w*f),1./f
    maps['epoch'],maps['night']=epoch,ex['night']
    return maps,radius


class DeviceGroups:
    def __init__(self,points):
        import torch
        self.t=torch
        self.xy=torch.as_tensor(np.stack((points['x'],points['y'])),device='cuda')
        self.mag=torch.as_tensor(points['mag'].copy(),device='cuda')
        self.e=torch.as_tensor(points['exposure'].astype('i8'),device='cuda')

    def partition(self,maps,scale,phase,n):
        t=self.t
        a=t.as_tensor(np.column_stack([maps[k] for k in MAP.names[:6]]),device='cuda')
        e=self.e[:n];x,y=self.xy[:,:n]
        bx=t.floor(a[e,0]+a[e,2]*x+a[e,4]*y).long()
        by=t.floor(a[e,1]+a[e,3]*x+a[e,5]*y).long()
        b=t.floor(self.mag[:n]*scale+phase).long()
        epoch=t.as_tensor(maps['epoch'].astype('i8'),device='cuda')[e]
        key=((epoch&255)<<48)|((b&255)<<40)|((bx&1048575)<<20)|(by&1048575)
        key,order=t.sort(key,stable=True)
        _,count=t.unique_consecutive(key,return_counts=True)
        end=count.cumsum(0);use=count>=3
        # Only copy groups that can supply a triangle. Stable order keeps
        # the most informative points first in saturated cells.
        lengths=count[use].clamp_max(24)
        dest=lengths.cumsum(0)
        start=end[use]-count[use]
        ix=t.repeat_interleave(start-(dest-lengths),lengths)
        ix+=t.arange(len(ix),device='cuda')
        return tuple(z.to(t.int32).cpu().numpy() for z in (order[ix],dest-lengths,dest))


def original_follow(control,chosen,threads,seconds):
    from original_recovery import RECOVERY,ptr,integer,real
    sky=control.get('original_sky')
    if sky is None or 'cell_primary_rows' not in control or not chosen:return
    primary=control['cell_primary_rows'];extra=control['additional']['rows']
    cores=np.zeros(len(chosen),CERTIFIED)
    ids=np.unique(np.concatenate([np.asarray(k,'i4') for k,v in chosen]))
    own=np.empty(len(ids),'i4')
    main=ids<len(primary)
    own[main]=primary[ids[main]];own[~main]=extra[ids[~main]-len(primary)]
    for c,(rows,(g,merit)) in zip(cores,chosen):
        n=len(rows);c['row'][:n]=np.searchsorted(ids,rows)
        c['count']=c['nights']=n;c['x'][2]=g;c['merit']=merit
    fn=sky.lib.burst_raw_follow
    fn.argtypes=[ptr,ptr,ptr,integer,ptr,integer,integer,real,integer,ptr,ptr]
    fn.restype=integer
    result=np.empty(min(1500,2*len(cores)),RECOVERY);done=ctypes.c_int()
    n=fn(sky.handle,own.ctypes.data,None,len(own),cores.ctypes.data,len(cores),
         threads,seconds,len(result),result.ctypes.data,ctypes.byref(done))
    if n<0:raise RuntimeError('original exposure cell continuation failed')
    if n:control['raw_results'].put((result[:n].copy(),[]))


def work(index,ids,quality,repeat,secondary,ex,pole,device,control,seconds,threads):
    stop=control['event']
    deadline=time.monotonic()+seconds
    points=catalogue(index,ids,quality,repeat,secondary)
    if len(points)<4 or stop.is_set():return
    count=np.maximum(1,np.bincount(points['exposure'],minlength=len(ex)))
    centers=np.column_stack([np.bincount(points['exposure'],weights=points[k],minlength=len(ex))/count
                             for k in ('x','y')])
    ptr,integer,real=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    run=index.lib.comoving_quads
    run.argtypes=[ptr,ptr,integer,ptr,real,real,real,ptr,ptr,ptr,integer,integer,integer,real,ptr,ptr]
    run.restype=integer
    cuda=None
    if device=='cuda':
        import torch
        try:
            if torch.cuda.mem_get_info()[0]>1.6e9:cuda=DeviceGroups(points)
        except (RuntimeError,MemoryError) as err:
            print('co-moving partition on host:',str(err)[:120],flush=True)
    rng=qmc.Sobol(10,scramble=True,seed=671).random_base2(10)
    total=np.zeros(6,'i8')
    quads={};triangles={};seen=set();j=0;verified=0
    def verify(room):
        nonlocal verified
        if stop.is_set():return
        start=time.monotonic()
        raw=[]
        for store,portion,rate in ((quads,.43,440.),(triangles,.76,220.)):
            allowance=start+room*portion-time.monotonic()
            if not store or allowance<.15:continue
            items=sorted(store.items(),key=lambda item:-item[1][1])
            chosen=items[:max(8,min(5500,int(allowance*rate)))]
            raw.append(chosen[:500])
            rows=np.array([k for k,v in chosen],'i4')
            gamma=np.array([v[0] for k,v in chosen],'f8')
            results=certify(index,rows,gamma,np.full(len(gamma),-1),threads,allowance,3500)
            for k,v in chosen:
                seen.add(k);store.pop(k,None)
            if len(results):control['certified'].put(results)
            verified+=len(results)
        left=start+room-time.monotonic()
        if raw and left>.15:
            # Interleave seed sizes so numerous cheap quartets cannot use
            # the entire original-exposure follow-up allowance.
            chosen=[items[i] for i in range(max(map(len,raw))) for items in raw if i<len(items)]
            original_follow(control,chosen,threads,left)
    for j,u in enumerate(rng,1):
        room=deadline-time.monotonic()
        if room<3. or stop.is_set():break
        maps,radius=chart(ex,centers,pole,u,j-1)
        frac=(.32,.64,1.,.48,1.)[(j-1)%5]
        n=min(len(points),max(65000,int(len(points)*frac)))
        scale=1./(.50+.46*u[9]);phase=float(u[8]);pieces=(None,None,None);ng=0
        if cuda is not None:
            try:
                arrays=cuda.partition(maps,scale,phase,n)
                ng=len(arrays[1]);pieces=tuple(x.ctypes.data for x in arrays)
            except (RuntimeError,MemoryError) as err:
                print('co-moving partition on host:',str(err)[:120],flush=True)
                cuda=None
        out=np.empty(16000,QUAD);stats=np.zeros(6,'i8')
        slot=max(.18,min(1.7,room*.18))
        k=run(index.handle,points.ctypes.data,n,maps.ctypes.data,scale,phase,1./radius,
              *pieces,ng,len(out),threads,slot,out.ctypes.data,stats.ctypes.data)
        if k<0:raise RuntimeError('co-moving cell enumeration failed')
        total+=stats
        for v in out[:k]:
            rr=v['rows'];triple=rr[3]<0
            key=tuple(sorted(map(int,rr[:3] if triple else rr)))
            score=float(v['merit'])
            proposals=triangles if triple else quads
            if key not in seen and (key not in proposals or proposals[key][1]<score):
                proposals[key]=(float(v['g']),score)
        if len(quads)>36000:
            quads=dict(sorted(quads.items(),key=lambda item:-item[1][1])[:24000])
        if len(triangles)>45000:
            triangles=dict(sorted(triangles.items(),key=lambda item:-item[1][1])[:26000])
        if j in (3,9) or j%12==0:
            verify(max(.5,min(8.,.18*(deadline-time.monotonic()))))
    if deadline-time.monotonic()>.45:
        verify(max(.4,deadline-time.monotonic()))
    print('co-moving orbit cells:',j,'charts;',total.tolist(),';',verified,'confirmed fits',flush=True)


def begin(*args):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='co-moving-orbits')
    job=pool.submit(work,*args)
    def report(f):
        if f.exception() is not None:print('co-moving orbit search:',repr(f.exception())[:250],flush=True)
    job.add_done_callback(report)
    pool.shutdown(wait=False)
    return job
