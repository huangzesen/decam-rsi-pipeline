"""Device lookup of two raw returns, caching only the shutters in use."""
import ctypes
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from apparent_backend import cuda_kernel
from attributables import SEED
from quartet_device import raw

PIXEL=np.dtype([(k,'f4') for k in ('x','y','mag','sigma','shape')]+[('id','u4')])
ANCHOR=np.dtype([('pos','f4',4),('mag','f4'),('quality','f4'),('a','i4'),('b','i4'),
                 ('raw','f4',4),('ea','i4'),('eb','i4')])


def frame_arrays(sky,e):
    ptr=ctypes.c_void_p
    fn=sky.lib.conditional_frame
    fn.argtypes=[ptr,ctypes.c_int,ptr,ptr,ptr,ptr]
    fn.restype=ctypes.c_int
    shape=np.empty(6,'i4');rho=np.empty(240,'f4')
    if fn(sky.handle,e,shape.ctypes.data,None,None,None)<0:
        raise ValueError('raw exposure outside the image index')
    grid=np.empty(max(1,shape[4]),'i4');points=np.empty(max(1,shape[5]),PIXEL)
    if fn(sky.handle,e,shape.ctypes.data,points.ctypes.data,grid.ctypes.data,rho.ctypes.data)<0:
        raise ValueError('could not pack raw exposure')
    return (points,grid,shape,rho)


def anchors(pairs,sky):
    a=np.empty(len(pairs),ANCHOR)
    for k in ('pos','mag','quality','a','b'):a[k]=pairs[k]
    a['ea'],a['eb']=sky.exposure[pairs['a']],sky.exposure[pairs['b']]
    a['raw'][:,:2]=sky.pos[pairs['a']]*206264.80624709636
    a['raw'][:,2:]=sky.pos[pairs['b']]*206264.80624709636
    return a


class Device:
    def __init__(self,sky,visits):
        if len(sky.mag)>=1<<29 or torch.cuda.mem_get_info()[0]<750000000:
            raise MemoryError('raw image buffers exceed the available capacity')
        path=Path(__file__).with_suffix('.cpp')
        code=path.read_text().replace('#include "quartet_fit.hpp"',path.with_name('quartet_fit.hpp').read_text())
        self.kernel=cuda_kernel(code,'raw_chord_query')
        self.sky=sky;self.visits=visits
        self.shutters=raw(np.column_stack([visits[k] for k in ('t','ox','oy','oz')]))
        self.cache=OrderedDict();self.bytes=0
        self.limit=int(min(850000000,.12*torch.cuda.mem_get_info()[0]))
        self.validate()

    def frame(self,e):
        e=int(e)
        if e in self.cache:
            self.cache.move_to_end(e)
        else:
            arr=frame_arrays(self.sky,e)
            pack=[raw(v) for v in arr]
            size=sum(v.nbytes for v in arr);self.bytes+=size
            self.cache[e]=(pack,size)
            while len(self.cache)>1 and self.bytes>self.limit:
                self.bytes-=self.cache.popitem(last=False)[1][1]
        return self.cache[e][0]

    def launch(self,anchor,n,model,probes,third,fourth,shutters,gamma,color3,color4):
        out=torch.empty((3*n,SEED.itemsize),dtype=torch.uint8,device='cuda')
        scores=torch.empty(3*n,dtype=torch.float32,device='cuda')
        self.kernel(grid=((n+63)//64,1,1),block=(64,1,1),
                    args=[anchor,n,model,probes,*third,*fourth,shutters,float(gamma),float(color3),
                          float(color4),out,scores])
        return out,scores

    def measure(self,pairs,model,third,fourth,lever,gamma,cap):
        n=len(pairs)
        anchor=raw(anchors(pairs,self.sky));m=raw(model);rows=[];merit=[]
        for v in third:
            first=self.frame(v['e']);c3=.24*abs(lever-self.visits['lever'][v['e']])
            for w in fourth:
                second=self.frame(w['e']);c4=.24*abs(lever-self.visits['lever'][w['e']])
                probes=raw(np.array([v,w],dtype=third.dtype))
                o,s=self.launch(anchor,n,m,probes,first,second,self.shutters,gamma,c3,c4)
                rows.append(o);merit.append(s)
        if not rows:
            return np.empty(0,SEED),np.zeros(4,'i8')
        scores=torch.cat(merit);records=torch.cat(rows)
        values,ix=torch.topk(scores,min(cap,len(scores)),sorted=False)
        output=records[ix].cpu().numpy().view(SEED).ravel()
        good=values.cpu().numpy()>-1.e19
        return output[good],np.array([n,0,0,good.sum()],'i8')

    def validate(self):
        a,model,probes,frames,shutters,gamma=fixture()
        o,s=self.launch(raw(a),len(a),raw(model),raw(probes),[raw(x) for x in frames[0]],
                        [raw(x) for x in frames[1]],raw(shutters),gamma,0.,0.)
        sel=s.cpu().numpy()>-1.e19
        seeds=o.cpu().numpy().view(SEED).ravel()[sel]
        if np.sum(np.all(seeds['rows']%len(a)==seeds['rows'][:,:1]%len(a),axis=1))<len(a):
            raise ArithmeticError('raw catalogue orbit fixture failed')


def fixture():
    from chord_returns import PROBE,condition
    from companion_graph import VISIT
    import quartet_geometry as geo
    t=np.array([0.,9.,338.,371.]);vis=np.zeros(4,VISIT);vis['t']=t
    vis['ox']=np.cos(t*.0172);vis['oy']=np.sin(t*.0172);vis['oz']=.12*np.cos(t*.0172)
    ob=np.column_stack([vis[k] for k in ('ox','oy','oz')]);pole=np.array([0.,-.65,.76])
    n=16;j=np.arange(n)
    pos,vel=geo.directions(np.column_stack((j*.00017,-j*.00013)),np.full(n,44.),np.ones(n),
                            np.zeros(n),np.zeros(n),ob[0],pole)
    xy=geo.apparent(pos,vel,t[None,:],ob)/geo.AS
    sim=geo.sample(vis,np.zeros((4,2)),pole,np.array([[0,1,2,3,0]]),geo.quadrature(128))
    m=condition(sim)
    a=np.zeros(n,ANCHOR);a['pos']=geo.pair_vector(xy[:,:2])
    a['mag']=21.4+.14*j;a['a']=j;a['b']=n+j;a['eb']=1;a['raw']=xy[:,:2].reshape(n,4)
    probes=np.zeros(2,PROBE);probes['e']=[2,3];probes['scale']=1.
    frames=[]
    for k in (2,3):
        p=np.zeros(n,PIXEL);p['x'],p['y']=xy[:,k,:].T;p['mag']=a['mag'];p['sigma']=.095
        p['id']=(k*n+j)|(2<<29);bucket=np.floor(xy[:,k,:]/24.).astype('i4')
        x0,y0=bucket.min(0);nx,ny=bucket.max(0)-bucket.min(0)+1
        key=(bucket[:,1]-y0)*nx+bucket[:,0]-x0
        start=np.r_[0,np.cumsum(np.bincount(key,minlength=nx*ny))].astype('i4')
        counts=np.r_[np.diff(start),0].astype('i4')
        grid=np.r_[start,counts,counts]
        shape=np.array([x0,y0,nx,ny,len(grid),n],'i4')
        frames.append((p[np.argsort(key)],grid,shape,np.full(240,2.e-7,'f4')))
    return a,m,probes,frames,np.column_stack((t,ob)),1./44.
