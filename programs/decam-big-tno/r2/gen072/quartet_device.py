"""Batched CUDA joins of the two measured angular displacements."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from apparent_backend import cuda_kernel
from attributables import SEED


FIELD = np.dtype([('x0','f4'),('width','f4')]+[(k,'i4') for k in ('nx','first','start','count')])
GPU_PAIR = np.dtype([('a','i4'),('b','i4'),('quality','f4')])
PLAN = np.dtype([('models','f8',(2,80)),('ex','f8',(4,4)),('gamma','f8'),('color','f8')])


def columns(points, edges):
    images = np.zeros(len(edges)-1, FIELD)
    sorted_points = np.empty_like(points)
    pieces = []
    offset = 0
    for e, (a,b) in enumerate(zip(edges[:-1], edges[1:])):
        v = points[a:b]
        im = images[e]
        im['first'], im['start'], im['count'] = a, offset, len(v)
        if len(v):
            im['x0'] = float(v['x'].min())-.1
            im['width'] = max(18., (v['x'].max()-im['x0'])/max(12.,1.8*np.sqrt(len(v))))
            k = np.floor((v['x'].astype(float)-im['x0'])/im['width']).astype('i4')
            im['nx'] = int(k.max())+1
            order = np.lexsort((v['y'], k))
            sorted_points[a:b] = v[order]
            starts = a+np.r_[0, np.cumsum(np.bincount(k, minlength=im['nx']))]
        else:
            im['nx'], im['width'] = 1, 50.
            starts = np.array([a,a])
        pieces.append(starts)
        offset += len(starts)
    return sorted_points, images, np.concatenate(pieces).astype('i4')


def source():
    cpp = Path(__file__).with_suffix('.cpp')
    return cpp.read_text().replace('#include "quartet_fit.hpp"',
                                  cpp.with_name('quartet_fit.hpp').read_text())


def raw(x):
    return torch.as_tensor(np.ascontiguousarray(x).view('u1'), device='cuda')


class Device:
    def __init__(self, points, edges, visits, kernels=None, real_visits=None, footprints=None):
        if torch.cuda.mem_get_info()[0]<900000000:
            raise MemoryError('device memory is occupied by the active orbit search')
        names = ('exposure_pairs','chord_cells','chord_query','companion_footprints')
        self.kernels = kernels or [cuda_kernel(source(), name) for name in names]
        pts, self.images, starts = columns(points, edges)
        self.data = [raw(v) for v in (pts,self.images,starts)]
        if footprints is None:
            from quartet_footprints import Footprints
            footprints=Footprints(points,edges)
        self.footprints=[raw(v) for v in (footprints.tiles,footprints.grid)]
        self.visits = visits
        actual = visits if real_visits is None else real_visits
        self.shutter = raw(np.column_stack([actual[k] for k in ('t','ox','oy','oz')]))
        self.cache = OrderedDict()
        self.bytes = 0
        self.max_bytes = int(min(720000000,.10*torch.cuda.mem_get_info()[0]))

    def pairs(self, a, b, family, gate, cap):
        key = (int(a),int(b),int(family),cap)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        n = int(self.images['count'][a])
        per = int(np.clip(5*cap//max(1,n), 5, 24))
        if not n or not self.images['count'][b]:
            packed = torch.zeros((0,3),dtype=torch.int32,device='cuda')
        else:
            ab = torch.empty((n*per,2),dtype=torch.int32,device='cuda')
            s = torch.empty(n*per,dtype=torch.float32,device='cuda')
            color = float(self.visits['lever'][a]-self.visits['lever'][b])
            self.kernels[0](grid=((n+63)//64,1,1),block=(64,1,1),
                            args=[*self.data,a,b,gate,color,per,ab,s])
            if len(s)>cap:
                s,ix = torch.topk(s,cap,sorted=False)
                ab = ab[ix]
            packed = torch.cat((ab,s.view(torch.int32).reshape(-1,1)),1)
        self.cache[key] = packed
        self.bytes += packed.numel()*4
        while self.bytes>self.max_bytes or len(self.cache)>12000:
            self.bytes -= self.cache.popitem(last=False)[1].numel()*4
        return packed

    def batch(self, batch, maps, ga, gb, centers, shutters, cap, keep=768,routes=None):
        visits = self.visits
        ga,gb = raw(ga).reshape(len(batch),-1),raw(gb).reshape(len(batch),-1)
        left,right = [],[]
        for i,(a,b,c,d,fam) in enumerate(batch.tolist()):
            reservoir=cap*(2 if routes is not None else 1)
            left.append(self.pairs(a,b,fam,ga[i],reservoir))
            right.append(self.pairs(c,d,fam,gb[i],reservoir))
        count = max([len(p) for p in left+right])
        if not count:
            return np.empty(0,SEED),0
        def padded(pairs):
            table = torch.full((len(pairs),count,3),-1,dtype=torch.int32,device='cuda')
            for i,v in enumerate(pairs):
                table[i,:len(v)] = v
            return table
        a,b = padded(left),padded(right)
        if routes is not None:
            scores=torch.empty((len(batch),2,count),device='cuda',dtype=torch.float32)
            self.kernels[3](grid=((count+63)//64,len(batch),2),block=(64,1,1),
                            args=[self.data[0],a,b,count,raw(routes),raw(batch[:,:4]),*self.footprints,scores])
            width=min(cap,count)
            score,ix=torch.topk(scores,width,dim=2,sorted=False)
            def supported(table,role):
                chosen=table.gather(1,ix[:,role,:,None].expand(-1,-1,3))
                chosen[:,:,1].masked_fill_(score[:,role]<-1.e19,-1)
                return chosen
            a,b=supported(a,0),supported(b,1)
            count=width
        plans = np.empty(len(batch), PLAN)
        plans['models'] = maps
        plans['ex'] = shutters[batch[:,:4]]
        from quartet_geometry import FAMILIES
        plans['gamma'] = np.sqrt(1.+np.sum(centers[batch[:,0]]**2,axis=1))/FAMILIES[batch[:,4],0]
        lever = visits['lever'][batch[:,:4]]
        plans['color'] = .5*(lever[:,0]+lever[:,1]-lever[:,2]-lever[:,3])
        plan_dev = raw(plans)
        cells = 1<<(2*count-1).bit_length()
        head = torch.full((len(batch),cells),-1,dtype=torch.int32,device='cuda')
        node = torch.empty((len(batch),count,20),dtype=torch.uint8,device='cuda')
        used = torch.zeros(len(batch),dtype=torch.int32,device='cuda')
        output = torch.empty((len(batch),keep*SEED.itemsize),dtype=torch.uint8,device='cuda')
        best_rows=torch.empty((len(batch),count,SEED.itemsize),dtype=torch.uint8,device='cuda')
        best_scores=torch.empty((len(batch),count),dtype=torch.float32,device='cuda')
        grid=((count+63)//64,len(batch),1)
        self.kernels[1](grid=grid,block=(64,1,1),
                        args=[self.data[0],b,count,plan_dev,head,cells,node])
        self.kernels[2](grid=grid,block=(64,1,1),
                        args=[self.data[0],a,b,count,plan_dev,head,cells,node,output,used,keep,self.shutter,
                              best_rows,best_scores])
        sizes = used.cpu().numpy()
        if (sizes<0).any():
            raise ArithmeticError('invalid device quartet count')
        result = output.cpu().numpy().view(SEED)
        selected = np.arange(keep)[None,:]<sizes[:,None]
        ordinary=result[selected]
        overflow=sizes>keep
        if overflow.any():
            score,ix=torch.topk(best_scores,min(keep,count),dim=1,sorted=False)
            records=best_rows.gather(1,ix[:,:,None].expand(-1,-1,SEED.itemsize))
            present=overflow[:,None]&(score.cpu().numpy()>-1.e19)
            extra=records.cpu().numpy().view(SEED).reshape(present.shape)[present]
            ordinary=np.concatenate((ordinary,extra))
        return ordinary,int(sizes.sum())


def validate(kernels):
    import quartet_geometry as geo
    from quartet_footprints import routes
    from companion_graph import VISIT
    from orbit_quartets import POINT, signature_models
    dates=np.array([0.,7.,300.,323.])
    v=np.zeros(4,VISIT);v['t']=dates
    v['ox']=np.cos(dates*.0172);v['oy']=np.sin(dates*.0172);v['oz']=.1*np.cos(dates*.0172)
    pole=np.array([0.,-.65,.76])
    center=np.zeros((4,2))
    plans=np.array([[0,1,2,3,0]],'i4')
    z=geo.quadrature(128)
    sample=geo.sample(v,center,pole,plans,z)
    # Use noiseless, independent positions for the physical consistency check.
    ob=np.column_stack([v[k] for k in ('ox','oy','oz')])
    m=np.arange(18);pos,vel=geo.directions(np.column_stack((m*1.e-4,-m*1.3e-4)),
                          np.full(18,44.),np.ones(18),np.zeros(18),np.zeros(18),ob[0],pole)
    xy=geo.apparent(pos,vel,dates[None,:],ob)/geo.AS
    p=np.zeros((4,len(m)),POINT)
    p['x'],p['y']=xy.transpose(2,1,0)
    p['rawx'],p['rawy']=p['x'],p['y']
    p['exposure']=np.arange(4)[:,None]
    p['row']=np.arange(p.size).reshape(p.shape)
    p['mag']=21.+.19*m;p['sigma']=.12
    device=Device(p.ravel(),len(m)*np.arange(5),v,kernels=kernels)
    gate=geo.pair_gate(sample[:,:,:2])
    gate2=geo.pair_gate(geo.sample(v,center,pole,plans[:,[2,3,2,3,4]],z)[:,:,:2])
    sh=np.column_stack([v[k] for k in ('t','ox','oy','oz')])
    got,_=device.batch(plans,signature_models(sample),gate,gate2,center,sh,1000,routes=routes(sample))
    true=(got['rows']%len(m)==got['rows'][:,:1]%len(m)).all(1)
    if len(np.unique(got['rows'][true,0]))<len(m)-1:
        raise ArithmeticError('four-position device orbit fixture did not recover the input tracks')
