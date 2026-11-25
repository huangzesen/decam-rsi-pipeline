"""Batch the two-parameter orbital search over independent image catalogues."""
import ctypes
from pathlib import Path
import time

import numpy as np
import torch

from original_recovery import integer, ptr
from transient_arcs import PROPOSAL, diversity


AS = np.deg2rad(1./3600.)
STEP = 96.*AS


def pack(sky, full, small, depth):
    ne = len(sky.geometry)
    grid = np.zeros((2*ne,5),'i4')
    geom = np.zeros((ne,12),'f8')
    for j,k in enumerate(('t','ox','oy','oz')):
        geom[:,j]=sky.geometry[k]
    geom[:,4:7]=np.column_stack((depth,sky.lever,sky.geometry['night']))
    geom[:,7]=geom[:,9]=1.e5
    geom[:,8]=geom[:,10]=-1.e5
    area=getattr(sky,'visit_area',None)
    if area is None:
        # Occupied area at camera scale. A rectangular bounding box would
        # count gaps between distant pieces of an exposure in a sky cut.
        raw=sky.exposure
        edges=np.searchsorted(raw,np.arange(ne+1))
        area=np.zeros(ne)
        for e in range(ne):
            xy=sky.pos[edges[e]:edges[e+1]:4]
            if len(xy):
                cells=np.floor(xy/(85.*AS)).astype('i4')
                key=(cells[:,0].astype('i8')<<32)|(cells[:,1].astype('i8')&0xffffffff)
                area[e]=max(900.,len(np.unique(key))*85.**2)
            else:area[e]=900.
        sky.visit_area=area
    points=[];rows=[];starts=[];used=0;cells=0
    density=np.zeros((2*ne,48),'f4')
    for lane,table in enumerate((full,small)):
        ids=table['row']
        ee=sky.exposure[ids]
        order=np.argsort(ee,kind='stable')
        cuts=np.searchsorted(ee[order],np.arange(ne+1))
        for e in range(ne):
            q=order[cuts[e]:cuts[e+1]]
            if not len(q):continue
            raw=ids[q]
            pos=sky.pos[raw]
            bins=np.floor(pos/STEP).astype('i4')
            minimum=bins.min(0)
            bins-=minimum
            nx,ny=bins.max(0)+1
            if int(nx)*int(ny)>1700000:continue
            key=bins[:,0]+nx*bins[:,1];sort=np.argsort(key,kind='stable')
            layer=2*e+lane
            grid[layer]=(*minimum,nx,ny,cells)
            offsets=np.zeros(nx*ny+1,'i4')
            np.cumsum(np.bincount(key,minlength=nx*ny),out=offsets[1:])
            starts.append(offsets+used);cells+=len(offsets)
            if cells>=2**31-1:raise MemoryError('visit grid exceeds signed index range')
            point=np.column_stack((pos,sky.mag[raw],table['merit'][q])).astype('f4')
            points.append(point[sort]);rows.append(raw[sort]);used+=len(raw)
            mb=np.clip(((point[:,2]-11.)*2.).astype('i4'),0,47)
            h=np.bincount(mb,minlength=48)
            density[layer]=(np.convolve(np.pad(h,1,mode='edge'),[.5,1.,.5],mode='valid')+.65)/area[e]
            geom[e,7]=min(geom[e,7],pos[:,0].min());geom[e,8]=max(geom[e,8],pos[:,0].max())
            geom[e,9]=min(geom[e,9],pos[:,1].min());geom[e,10]=max(geom[e,10],pos[:,1].max())
    if not used:return None
    return (geom,grid,np.concatenate(starts),np.concatenate(points),density,np.concatenate(rows).astype('i4'))


def models(sky,depth,east,pairs):
    fn=sky.lib.visit_transfer_table
    fn.argtypes=[ptr,ptr,ptr,ptr,integer,ptr]
    fn.restype=integer
    out=np.empty((len(pairs),68),'f8')
    n=fn(sky.handle,depth.ctypes.data,east.ctypes.data,pairs.ctypes.data,len(pairs),out.ctypes.data)
    if n!=len(out):raise RuntimeError('conditional transfer construction failed')
    return out


class Ranging:
    def __init__(self, packed, device, threads):
        self.ne=len(packed[0]);self.device=device;self.threads=threads
        if device=='cuda':
            from apparent_backend import cuda_kernel
            self.kernel=cuda_kernel(Path(__file__).with_suffix('.cpp').read_text(),'visit_ranging')
            self.arrays=[torch.as_tensor(a,device=device) for a in packed]
        else:
            from companion_graph import build_library
            self.lib=build_library('visit_device')
            self.kernel=self.lib.visit_ranging_host
            self.kernel.argtypes=[ptr]*7+[integer,integer,ptr,ptr,integer]
            self.kernel.restype=None
            self.arrays=packed

    def apply(self,state):
        n=len(state)
        if self.device=='cuda':
            x=torch.as_tensor(state,device=self.device)
            ids=torch.empty((n,4,3),device=self.device,dtype=torch.int32)
            val=torch.empty((n,4,3),device=self.device,dtype=torch.float32)
            self.kernel(grid=(n,1,1),block=(64,1,1),args=[x,*self.arrays,self.ne,n,ids,val])
            return ids.cpu().numpy(),val.cpu().numpy()
        ids=np.empty((n,4,3),'i4');val=np.empty((n,4,3),'f4')
        self.kernel(state.ctypes.data,*(a.ctypes.data for a in self.arrays),self.ne,n,
                    ids.ctypes.data,val.ctypes.data,self.threads)
        return ids,val


def ranked(sky,out,limit):
    if not len(out):return out
    rows=out['row'];e=sky.exposure[rows]
    t=sky.geometry['t'][e];gap=np.ptp(t,axis=1)
    mb=np.clip((np.median(sky.mag[rows],axis=1)-12.)*1.6,0,28).astype('i4')
    group=(e[:,0]*6+np.digitize(gap,[40.,160.,520.,1200.,2400.]))*29+mb
    merit=diversity(out['merit'],group,.64)
    order=np.argsort(-merit,kind='stable')
    out=out[order]
    keys=np.column_stack((np.sort(out['row'],axis=1),out['witness']))
    _,first=np.unique(keys,axis=0,return_index=True)
    return np.ascontiguousarray(out[np.sort(first)[:limit]])


def search(sky,depth,east,full,small,bank,device,threads,deadline,control,limit=160000,emit=None):
    pack0=pack(sky,full,small,depth)
    if pack0 is None:return np.empty(0,PROPOSAL),0
    backend=Ranging(pack0,device,threads)
    if device=='cuda' and len(bank):
        audit=bank[np.linspace(0,len(bank)-1,min(96,len(bank)),dtype='i8')]
        st=models(sky,depth,east,np.ascontiguousarray(audit))
        ref_id,ref_value=Ranging(pack0,'cpu',1).apply(st)
        gpu_id,gpu_value=backend.apply(st)
        agree=np.all(ref_id==gpu_id,axis=2)
        present=(ref_id[:,:,0]>=0)|(gpu_id[:,:,0]>=0)
        if (present&~agree).sum()>max(1,.04*present.sum()):
            raise ArithmeticError('conditional orbit assignments differ on device')
        ok=present&agree
        if ok.any() and not np.allclose(ref_value[ok],gpu_value[ok],rtol=.001,atol=.008):
            raise ArithmeticError('conditional orbit scores differ on device')
    step=1536 if device=='cuda' else 144
    entries=[];pos=0;size=0;fresh=[];last=time.monotonic()
    while pos<len(bank) and time.monotonic()<deadline and not control['event'].is_set():
        q=bank[pos:pos+step]
        state=models(sky,depth,east,q)
        ids,val=backend.apply(state)
        ok=(ids[:,:,0]>=0)&(ids[:,:,1]>=0)&np.isfinite(val).all(2)&(val[:,:,1]>1./8000.)&(val[:,:,1]<1./15.)
        i,j=np.where(ok)
        if len(i):
            out=np.zeros(len(i),PROPOSAL)
            out['row']=np.column_stack([q['a'][i],q['b'][i],ids[i,j,:2]])
            out['witness']=ids[i,j,2]
            out['merit'],out['g'],out['h']=val[i,j].T
            if pos==0:
                if np.max(out['row'])>=len(sky.mag):raise ArithmeticError('invalid device row')
                nights=sky.geometry['night'][sky.exposure[out['row']]]
                if np.any(np.diff(np.sort(nights,axis=1),axis=1)==0):raise ArithmeticError('repeated night in range consensus')
            entries.append(out);size+=len(out)
            if emit is not None:fresh.append(out)
        pos+=len(q)
        if emit is not None and fresh and time.monotonic()-last>13.:
            emit(ranked(sky,np.concatenate(fresh),12000))
            fresh=[];last=time.monotonic()
        if size>2*limit:
            entries=[ranked(sky,np.concatenate(entries),limit)];size=len(entries[0])
    if not entries:return np.empty(0,PROPOSAL),pos
    return ranked(sky,np.concatenate(entries),limit),pos
