"""Reuse observed two-night displacements across conditional orbit forecasts."""
from collections import OrderedDict
import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np

PAIR = np.dtype([('a','i4'),('b','i4'),('u','f4'),('v','f4'),('merit','f4')])
TRIPLE = np.dtype([('a','i4'),('b','i4'),('c','i4'),('merit','f4')])
GRID = np.dtype([('x','f4'),('y','f4'),('width','i4'),('height','i4'),('offset','i4')])
PARTNERS = 8
ALTERNATIVES = 2


@lru_cache(maxsize=1)
def device_kernels():
    from apparent_backend import cuda_kernel
    source = Path(__file__).with_name('night_query.cpp').read_text()
    source += '\n' + Path(__file__).with_suffix('.cpp').read_text()
    return {name:cuda_kernel(source,name) for name in
            ('observed_pairs','observed_triples','confirm_triples','search_returns')}


class Observations:
    def __init__(self, query, cache=None):
        from night_search import TRACK
        self.track_type = TRACK
        self.query = query
        self.cat = query.cat
        self.cache = cache if cache is not None else OrderedDict()
        self.tested = set()
        origin, maps = self.cat.visibility
        grid = np.zeros(len(maps), GRID)
        arrays = []
        offset = 0
        for i,(ox,oy,depth) in enumerate(maps):
            grid[i] = (origin[0]+270.*ox,origin[1]+270.*oy,
                       depth.shape[0],depth.shape[1],offset)
            arrays.append(depth.ravel())
            offset += depth.size
        self.grid, self.depth = grid, np.concatenate(arrays).astype('f4',copy=False)
        self.native=query.module
        if not hasattr(self.native,'measured_layout'):
            raise RuntimeError('native measured orbit lookup unavailable')
        from return_ephemeris import RETURN
        sizes=np.empty(4,'i4')
        self.native.measured_layout.argtypes=[ctypes.c_void_p]
        self.native.measured_layout.restype=None
        self.native.measured_layout(sizes.ctypes.data)
        if sizes.tolist()!=[dtype.itemsize for dtype in (PAIR,TRIPLE,GRID,RETURN)]:
            raise RuntimeError('incompatible measured-return catalogue layout')
        self.gpu = None
        for name,n in [('observed_pairs',5),('observed_triples',7),('confirm_triples',6),
                       ('search_returns',9)]:
            fn = getattr(self.native,name)
            fn.argtypes = [ctypes.c_void_p]*n+[ctypes.c_int,ctypes.c_void_p]
            fn.restype = None
        if query.kernel is not None:
            import torch
            try:
                self.gpu = device_kernels()
                with torch.cuda.stream(query.stream):
                    self.device_grid = [torch.as_tensor(a.view('u1'),device='cuda') for a in
                                        (self.grid,self.depth)]
            except (AttributeError,OSError,RuntimeError,MemoryError) as exc:
                print('measured pairs using host:',str(exc)[:160],flush=True)
                self.gpu = None
        self.stat = np.zeros(5, 'i8')

    def call(self,name,model,args,count,repeat,typ,use_grid=False):
        args = [np.ascontiguousarray(a) for a in args]
        result = np.empty((count,repeat),typ)
        native = [self.cat.points,self.cat.edges,self.cat.fields,model,*args]
        if use_grid:
            native.extend((self.grid,self.depth))
        def host(n,out):
            getattr(self.native,name)(*(a.ctypes.data for a in native),n,out.ctypes.data)
        if not count:
            return result
        if self.gpu is not None:
            import torch
            try:
                with torch.cuda.stream(self.query.stream):
                    dev = [torch.as_tensor(a.view('u1'),device='cuda') for a in [model,*args]]
                    out = torch.empty((count,repeat,typ.itemsize),device='cuda',dtype=torch.uint8)
                    extra = self.device_grid if use_grid else []
                    self.gpu[name](grid=((count+63)//64,1,1),block=(64,1,1),
                                   args=self.query.arrays+dev+extra+[count,out])
                    result = out.cpu().numpy().view(typ).reshape(count,repeat)
                if name not in self.tested:
                    n = min(count,128)
                    check = np.empty((n,repeat),typ)
                    host(n,check)
                    fields=('row',) if name in ('confirm_triples','search_returns') else (
                            ('a','b','c') if name=='observed_triples' else ('a','b'))
                    if not all(np.array_equal(check[k],result[k][:n]) for k in fields):
                        raise ArithmeticError('host/device measured associations disagree')
                    score = 'score' if name in ('confirm_triples','search_returns') else 'merit'
                    if not np.allclose(check[score],result[score][:n],rtol=.002,atol=.01):
                        raise ArithmeticError('host/device measured likelihoods disagree')
                    self.tested.add(name)
                return result
            except (ArithmeticError,RuntimeError,MemoryError,OSError) as exc:
                print('measured orbital returns continuing on host:',str(exc)[:160],flush=True)
                self.gpu = None
                result = np.empty((count,repeat),typ)
        host(count,result)
        return result

    def pairs(self, model, family):
        a,b = map(int,model['view'][0,:2])
        original = int(self.cat.fields['pad'][b]) if model['family'][0]>=0 else 0
        key = (a,b,int(family),original)
        if key in self.cache:
            self.stat[0]+=1
            self.cache.move_to_end(key)
            return self.cache[key]
        roots = self.cat.roots[a]
        pairs = self.call('observed_pairs',model,[roots],len(roots),PARTNERS,PAIR)
        pairs = pairs[pairs['a']>=0]
        self.stat[1:3] += [len(roots),len(pairs)]
        self.cache[key] = pairs
        while len(self.cache)>1 and (len(self.cache)>640 or
                                     sum(v.nbytes for v in self.cache.values())>120000000):
            self.cache.popitem(last=False)
        return pairs

    def triples(self,model,family,limit,pairs=None):
        if pairs is None:pairs=self.pairs(model,family)
        # Very broad two-point regions can contain many false associations.
        # Apportion an expensive third-image scan across sky and flux, keeping
        # other families and pointings reachable during a bounded search.
        cap = max(6000,limit*12)
        if len(pairs)>cap:
            p=self.cat.points[pairs['a']]
            score=.50*p['quality']+pairs['merit']
            pairs=pairs[balanced(score,p,cap)]
        live=self.call('observed_triples',model,[pairs],len(pairs),ALTERNATIVES,TRIPLE,True)
        live=live[live['a']>=0]
        self.stat[3]+=len(live)
        if len(live)>limit:
            live=live[balanced(live['merit'],self.cat.points[live['a']],limit)]
        self.stat[4]+=len(live)
        return live

    def confirm(self,models,jobs,seeds,ephemeris=None):
        if ephemeris is None:
            result=self.call('confirm_triples',models,[jobs,seeds],len(seeds),4,self.track_type)
        else:
            result=self.call('search_returns',models,[jobs,seeds,ephemeris],len(seeds),4,self.track_type,True)
        return result[result['row'][:,:,0]>=0]


def balanced(score,points,count):
    """Prefer good associations within each local brightness and sky population."""
    if len(points)<=count:
        return np.arange(len(points))
    size=1100.
    u=np.floor(points['x']/size).astype('i8')
    v=np.floor(points['y']/size).astype('i8')
    m=np.clip(np.floor((points['mag']-12.)/.7).astype('i8'),0,40)
    keys=((u-u.min())*(1+int(np.ptp(v)))+v-v.min())*41+m
    # A rank within the occupied cell accounts for local excess candidate
    # density without assigning an equal quota to an almost empty CCD.
    idx=np.lexsort((-score,keys))
    k=keys[idx]
    starts=np.flatnonzero(np.r_[True,k[1:]!=k[:-1]])
    rank=np.empty(len(idx),'i4')
    rank[idx]=np.arange(len(idx))-np.repeat(starts,np.diff(np.r_[starts,len(idx)]))
    adjusted=score-.62*np.log1p(rank)
    return np.argpartition(-adjusted,count-1)[:count]
