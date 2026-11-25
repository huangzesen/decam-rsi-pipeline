"""Compare measured arcs on the device in bounded spatial and temporal groups."""
import ctypes
from pathlib import Path
import time

import numpy as np
import torch

from attributables import SEED
from companion_graph import build_library

BLOCK=128
ARC=206264.80624709636


def packed(target,visits,edges,east):
    data=np.empty((len(edges),24),'f4');rows=np.empty((len(edges),4),'i4')
    lib=build_library('chord_join')
    ptr=ctypes.c_void_p;integer=ctypes.c_int;real=ctypes.c_double
    fn=lib.pack_motion_chords
    fn.argtypes=[ptr,ptr,ptr,integer,real,real,ptr,ptr];fn.restype=None
    fn(target.ctypes.data,visits.ctypes.data,edges.ctypes.data,len(edges),*map(float,east),
       data.ctypes.data,rows.ctypes.data)
    use=(rows[:,2]!=rows[:,3])&np.isfinite(data).all(1)
    return data[use],rows[use]


def groups(data,east):
    season=np.floor(data[:,0]/8.).astype('i4')
    drift=np.asarray(east)*(.01720209895/44.**1.5*ARC)
    xy=data[:,2:4]+data[:,6:8]/44.-(data[:,0]-8.*season)[:,None]*drift
    # Nearby sky positions on a Morton ordering share a
    # compact rectangle; each block lies wholly in one eight-day interval.
    cells=np.floor((xy-xy.min(0))/310.).astype('u8')
    bits=np.zeros(len(cells),'u8')
    for k in range(20):
        bits|=((cells[:,0]>>k)&1)<<(2*k)
        bits|=((cells[:,1]>>k)&1)<<(2*k+1)
    order=np.lexsort((bits,season)).astype('i4')
    tile=season[order];starts=np.flatnonzero(np.r_[True,tile[1:]!=tile[:-1]])
    ends=np.r_[starts[1:],len(order)]
    first=np.concatenate([np.arange(a,b,BLOCK,dtype='i4') for a,b in zip(starts,ends)])
    count=np.diff(np.r_[first,len(order)]).astype('i4')
    d=data[order]
    lo=np.minimum.reduceat(d[:,0],first);hi=np.maximum.reduceat(d[:,0],first)
    mid=.5*(lo+hi)
    obslo=np.minimum.reduceat(d[:,6:8],first,axis=0);obshi=np.maximum.reduceat(d[:,6:8],first,axis=0)
    coords=d[:,2:4]+d[:,6:8]/44.-(d[:,0]-np.repeat(mid,count))[:,None]*drift
    lower=np.minimum.reduceat(coords,first,axis=0);upper=np.maximum.reduceat(coords,first,axis=0)
    meta=np.column_stack((mid,.5*(hi-lo),.5*(obslo+obshi),.5*(obshi-obslo),
                          lower[:,0],upper[:,0],lower[:,1],upper[:,1])).astype('f4')
    return meta,order,first,count


def host(data,rows,pairs,east):
    lib=build_library('chord_device')
    fn=lib.chord_scores;ptr=ctypes.c_void_p
    fn.argtypes=[ptr,ptr,ptr,ctypes.c_int,ctypes.c_double,ctypes.c_double,ptr]
    fn.restype=None
    out=np.empty((len(pairs),2),'f8')
    fn(data.ctypes.data,rows.ctypes.data,pairs.ctypes.data,len(pairs),*map(float,east),out.ctypes.data)
    return out


def fixture():
    from companion_graph import VISIT
    from range_search import TARGET
    # Observe a synthetic circular orbit at several annual phases. The
    # false cases below have different sky positions and transverse speeds.
    e=np.zeros(8,VISIT)
    dates=365.25*np.array([0.,.43,1.06,1.94])
    e['t']=np.column_stack((dates,dates+np.arange(3.,7.))).ravel()
    t=e['t'];u=t*(2.*np.pi/365.25)
    e['ox'],e['oy'],e['oz']=np.sin(u),.18*np.cos(u),.98367*np.cos(u)
    e['night']=np.arange(len(e));radius=44.
    theta=t*.01720209895/radius**1.5
    points=np.zeros(8*64,TARGET);edges=np.zeros(4*64,[('a','i4'),('b','i4'),('score','f4'),('gamma','f4')])
    for k in range(64):
        ids=np.arange(8*k,8*k+8)
        off=.0015*(k%8)-.00525;lat=.0013*(k//8)-.00455
        points['x'][ids]=(np.sin(theta+off)-e['ox']/radius)/(np.cos(theta+off)-e['oz']/radius)
        points['y'][ids]=(lat+.06*np.sin(theta)-e['oy']/radius)/(np.cos(theta+off)-e['oz']/radius)
        points['mag'][ids]=22.+.04*(k%13);points['exposure'][ids]=np.arange(8)
        j=np.arange(4*k,4*k+4);edges['a'][j]=ids[::2];edges['b'][j]=ids[1::2]
        edges['score'][j]=4.;edges['gamma'][j]=1./radius
    return packed(points,e,edges,[1.,0.])


def verify(schedule,winners,device):
    d,r=fixture()
    # Verify the winning association and inverse distance in each half
    # warp-group, including groups with only incompatible correspondences.
    trial=np.arange(0,len(d),4,dtype='i8')
    order=np.r_[np.arange(1,128,2),np.arange(129,256,2)].astype('i4')
    pairs=np.column_stack((np.repeat(trial,len(order)),np.tile(order,len(trial)))).astype('i4')
    reference=host(d,r,pairs,[1.,0.]).reshape(len(trial),2,BLOCK//2,2)
    invalid=(pairs[:,1]<=pairs[:,0]).reshape(len(trial),2,BLOCK//2)
    reference[:,:,:,0][invalid]=-1.e19
    expected=reference[np.arange(len(trial))[:,None],np.arange(2)[None,:],np.argmax(reference[...,0],axis=2)]
    def tensor(x):return torch.as_tensor(x,device=device)
    out=torch.empty((len(trial),2,2),device=device,dtype=torch.float32);idx=torch.empty_like(out,dtype=torch.int32)
    winners(grid=(len(trial),1,1),block=(BLOCK,1,1),args=[
        tensor(d),tensor(r),tensor(order),tensor(trial),tensor(np.array([0],'i4')),
        tensor(np.array([128],'i4')),1,0,len(trial),1.,0.,idx,out])
    actual=out.cpu().numpy();valid=expected[:,:,0]>-1.e9
    if not valid.any() or not np.array_equal(valid,actual[:,:,0]>-1.e9):
        raise ArithmeticError('device chord checker missed a measured association')
    if not np.allclose(actual[valid],expected[valid],atol=3.e-4,rtol=2.e-5):
        raise ArithmeticError('host and device chord fits disagree')
    meta,order,base,size=groups(d,[1.,0.])
    membership=np.empty(len(order),'i4')
    membership[order]=np.repeat(np.arange(len(base)),size)
    total=len(d)*len(meta)
    mask=torch.empty(total,dtype=torch.int32,device=device)
    schedule(grid=((total+127)//128,1,1),block=(128,1,1),args=[
        tensor(d),tensor(meta),mask,len(meta),0,total,1.,0.])
    if not mask.cpu().numpy().reshape(len(d),len(meta))[trial,membership[trial+1]].all():
        raise ArithmeticError('device spatial index excluded bound test orbits')


def search(target,visits,bank,east,seconds,cap,event):
    from apparent_backend import cuda_kernel
    start=time.monotonic();dev=torch.device('cuda')
    if torch.cuda.mem_get_info(dev)[0]<650_000_000:raise MemoryError('device workspace in use')
    text=Path(__file__).with_suffix('.cpp').read_text()
    schedule=cuda_kernel(text,'chord_groups')
    winners=cuda_kernel(text,'chord_winners')
    verify(schedule,winners,dev)
    data,rows=packed(target,visits,bank,east)
    if len(data)<2:return np.empty(0,SEED),np.zeros(3,'i8')
    meta,order,base,size=groups(data,east)
    to=lambda v:torch.as_tensor(v,device=dev)
    d,r,m,o,b,sz=map(to,(data,rows,meta,order,base,size))
    ng=len(meta);direction=list(map(float,east))
    batch=max(8,min(120,500000//ng));pieces=[];stats=np.zeros(3,'i8');last=0
    # A separate stream lets the image and orbit fits continue submitting
    # work during this bounded scan of the phase-space index.
    stream=torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        for first in range(0,len(data),batch):
            if event.is_set() or time.monotonic()-start>=seconds:break
            end=min(len(data),first+batch);total=(end-first)*ng
            mask=torch.empty(total,dtype=torch.int32,device=dev)
            schedule(grid=((total+127)//128,1,1),block=(128,1,1),
                     args=[d,m,mask,ng,first,total,*direction])
            tasks=mask.nonzero().flatten().contiguous()
            stats[0]+=end-first;stats[1]+=len(tasks)
            for lo in range(0,len(tasks),48000):
                work=tasks[lo:lo+48000];n=len(work)
                ids=torch.empty((n,2,2),device=dev,dtype=torch.int32)
                val=torch.empty_like(ids,dtype=torch.float32)
                winners(grid=(n,1,1),block=(128,1,1),
                        args=[d,r,o,work,b,sz,ng,first,n,*direction,ids,val])
                good=val[:,:,0]>-1.e9
                pairs=ids[good].cpu().numpy();v=val[good].cpu().numpy()
                if not len(v):continue
                stats[2]+=len(v)
                # Keep the best comparisons for each independent anchor.
                # A single fortunate two-night arc can have thousands of
                # compatible neighbours before the independent fifth test.
                order2=np.lexsort((-v[:,0],pairs[:,0]))
                counts=np.unique(pairs[order2,0],return_counts=True)[1]
                select=np.arange(len(v))-np.repeat(np.r_[0,counts.cumsum()[:-1]],counts)<16
                sel=order2[select]
                result=np.zeros(len(sel),SEED)
                result['rows']=rows[pairs[sel],:2].reshape(-1,4)
                result['gamma']=v[sel,1];result['score']=v[sel,0];pieces.append(result)
                last+=len(result)
            if last>3*cap:
                z=np.concatenate(pieces);top=np.argpartition(-z['score'],cap-1)[:cap]
                pieces=[z[top].copy()];last=cap
    if not pieces:return np.empty(0,SEED),stats
    z=np.concatenate(pieces);z=z[np.argsort(-z['score'],kind='stable')[:cap]]
    return z,stats
