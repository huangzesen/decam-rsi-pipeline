"""Find compatible transverse velocities from separate single-exposure visits."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Full
import time

import numpy as np

from companion_graph import build_library
from original_recovery import integer, ptr, real
import return_confirmation
import transient_arcs as visits


POINT=np.dtype([(k,'f4') for k in ('x','y','mag','var','q')]+[('row','i4'),('ex','i4')])
VISIT=np.dtype([(k,'f8') for k in ('t','ox','oy','oz','lever','depth')]+[('night','i4'),('pad','i4')])
EDGE=np.dtype([(k,'f4') for k in ('x','y','gx','gy','hx','hy','xx','yy','ghx','ghy',
                                   'dt','noise','logrho','mark','mag','lever','variance')]+
              [('row','i4'),('night','i4')])
ORIGIN=np.dtype([(k,'f4') for k in ('g','vc','h','noise','mark','inorm')]+[('row','i4')])


def device_source():
    source=Path(__file__).with_suffix('.cpp').read_text()
    forecast=Path(__file__).with_name('pencil_forecast.hpp').read_text()
    return source.replace('#include "pencil_forecast.hpp"',forecast)


def foreground(sky, table, depth, east, limit=450000):
    rows=table['row']
    ex=sky.exposure[rows]
    m=sky.mag[rows]
    sn=np.maximum(4.5,sky.arrays[0][rows])
    # Estimate the opportunity for a source of this magnitude to be imaged
    # independently. Preserve separate places for single-shutter dropouts.
    magbin=np.clip(np.floor((m-12.)*1.5),0,29).astype('i4')
    rare=visits.normalize(ex,m,table['single'],sn)
    contrast=.78*table['single']+.44*np.minimum(rare,5.)+.48*np.clip(table['merit']-table['single'],0.,9.)
    q=np.clip(contrast-np.nanmedian(contrast),-5.,8.).astype('f4')
    group=30*ex+magbin
    rank=visits.diversity(q,group,.63)
    limit=min(limit,len(rows))
    if limit<len(rows):
        ix=np.argpartition(-rank,limit-1)[:limit]
    else:ix=np.arange(len(rows))
    ix.sort()
    p=np.zeros(len(ix),POINT);p['row']=rows[ix];p['ex']=ex[ix]
    rotation=np.array([east,[-east[1],east[0]]])
    p['x'],p['y']=(sky.pos[p['row']]@rotation.T).T
    p['q']=np.clip(.56*q[ix],-3.2,3.8)
    p['mag']=m[ix];p['var']=.14**2+(1.086/sn[ix])**2
    geometry=sky.geometry
    e=np.zeros(len(depth),VISIT)
    e['t'],e['night'],e['oz']=geometry['t'],geometry['night'],geometry['oz']
    e['ox'],e['oy']=(np.column_stack((geometry['ox'],geometry['oy']))@rotation.T).T
    e['depth'],e['lever']=depth,sky.lever
    # Alternate within-field rarity and per-visit source odds. This avoids
    # exhausting all seeds from the same unusually deep night early on.
    order=np.argsort(-rank[ix],kind='stable').astype('i4')
    return p,e,order


class Catalog:
    def __init__(self,p,e,pole,device,threads):
        self.p=p;self.e=e;self.threads=threads
        lib=build_library('velocity_pencil')
        init=lib.pencil_index;init.argtypes=[ptr,integer,ptr,integer,real,real];init.restype=ptr
        self.handle=init(p.ctypes.data,len(p),e.ctypes.data,len(e),*map(float,pole))
        if not self.handle:raise MemoryError('could not build velocity search index')
        self.free=lib.pencil_free;self.free.argtypes=[ptr];self.free.restype=None
        self.collect=lib.pencil_edges
        self.collect.argtypes=[ptr,ptr,integer,real,integer,integer,integer,ptr,ptr,ptr]
        self.collect.restype=integer
        self.host=lib.pencil_host
        self.host.argtypes=[ptr,ptr,ptr,integer,integer,integer,integer,integer,ptr]
        self.host.restype=None
        self.kernel=None;self.checked=False
        if device=='cuda':
            try:
                from apparent_backend import cuda_kernel
                self.kernel=cuda_kernel(device_source(),'velocity_pencils')
            except (OSError,RuntimeError,ImportError,ValueError) as exc:
                print('velocity comparison using CPU:',str(exc)[:150],flush=True)

    def edges(self,anchors,radius,family,cap):
        a=np.ascontiguousarray(anchors,'i4');n=len(a)
        orig=np.empty(n,ORIGIN);data=np.empty((n,cap),EDGE);counts=np.empty(n,'i4')
        status=self.collect(self.handle,a.ctypes.data,n,radius,family,cap,self.threads,
                            orig.ctypes.data,data.ctypes.data,counts.ctypes.data)
        if status!=n:raise RuntimeError('incomplete velocity neighbourhoods')
        return orig,data,counts

    def compare_cpu(self,orig,data,counts,nb,family):
        n=len(orig);out=np.empty((n,nb,3),visits.PROPOSAL)
        self.host(orig.ctypes.data,data.ctypes.data,counts.ctypes.data,data.shape[1],nb,n,
                  family,self.threads,out.ctypes.data)
        return out

    def compare(self,orig,data,counts,nb,family):
        if self.kernel is not None:
            try:
                import torch
                n,cap=data.shape
                def upload(a):
                    return torch.as_tensor(a.view('u1'),device='cuda')
                out=torch.empty(n*nb*3*visits.PROPOSAL.itemsize,dtype=torch.uint8,device='cuda')
                o,b,c=upload(orig),upload(data),torch.as_tensor(counts,device='cuda')
                self.kernel(grid=(n,(nb+31)//32,1),block=(32,1,1),
                            args=[o,b,c,cap,nb,n,family,out])
                answer=out.cpu().numpy().view(visits.PROPOSAL).reshape(n,nb,3)
                if not self.checked and np.any(answer['merit']>-1.e10):
                    use=np.flatnonzero(np.any(answer['merit']>-1.e10,axis=(1,2)))[:8]
                    ref=self.compare_cpu(orig[use],data[use],counts[use],nb,family)
                    if (not np.array_equal(ref['row'],answer['row'][use]) or
                        not np.allclose(ref['merit'],answer['merit'][use],rtol=.00006,atol=.001)):
                        raise ArithmeticError('host and device velocity fits disagree')
                    self.checked=True
                return answer
            except (RuntimeError,ValueError,ArithmeticError,OSError,MemoryError) as exc:
                print('velocity comparisons continuing on CPU:',str(exc)[:160],flush=True)
                self.kernel=None
        return self.compare_cpu(orig,data,counts,nb,family)

    def close(self):
        if self.handle:
            self.free(self.handle);self.handle=None


def distinct(groups,limit):
    if not groups:return np.empty(0,visits.PROPOSAL)
    fits=np.concatenate(groups)
    if not len(fits):return fits
    order=np.argsort(-fits['merit'],kind='stable')
    _,first=np.unique(np.sort(fits['row'][order],axis=1),axis=0,return_index=True)
    return fits[order[np.sort(first)[:limit]]]


def deeper_pairs(queue,a,data,counts,family):
    # Preserve independently measured pairs even when their next detectable
    # night lies below the bright foreground cut. A pair fixes part of the
    # distance through the transverse-velocity prior.
    if queue.full():return
    valid=np.arange(data.shape[1])<counts[:,None]
    gap=np.abs(data['dt']);bank=[]
    for lo,hi in ((.75,24.),(24.,170.),(170.,6500.)):
        current=valid&(gap>=lo)&(gap<hi)
        for choice in range(2):
            rows=np.flatnonzero(current.any(axis=1))
            if not len(rows):break
            ix=np.argmax(current[rows],axis=1)
            b=data[rows,ix]
            sx,sy=(.73,.73) if family==2 else (.24,.33)
            gx=b['gx']/sx;gy=b['gy']/sy;hx=b['hx']/sx;hy=b['hy']/sy
            x=b['x']/sx;y=b['y']/sy
            sg,sh=(.19,.20) if family==0 else (.29,.36)
            aa=1./sg**2+gx*gx+gy*gy;bb=gx*hx+gy*hy;cc=1./sh**2+hx*hx+hy*hy
            dg=(bb*(hx*x+hy*y)-cc*(gx*x+gy*y))/np.maximum(1.,aa*cc-bb*bb)
            q=np.empty(len(rows),visits.PAIR);q['a']=a['row'][rows];q['b']=b['row']
            q['gamma']=a['g'][rows]*(1.+np.clip(dg,-.50,.50))
            q['merit']=a['mark'][rows]+b['mark']-.25*np.log1p(ix)
            bank.append(q);current[rows]&=data['night'][rows]!=b['night'][:,None]
    if bank:
        pairs=np.concatenate(bank)
        try:queue.put_nowait(pairs[np.argsort(-pairs['merit'],kind='stable')])
        except Full:pass


def scan(index,anchors,radius,family,seconds,control,limit=36000):
    end=time.monotonic()+seconds
    two=[];three=[];queued=0;processed=0;compared=0
    cap=200 if index.kernel is not None else 112
    nb=48 if index.kernel is not None else 20
    batch=192 if index.kernel is not None else 64
    for lo in range(0,len(anchors),batch):
        if time.monotonic()>=end or control['event'].is_set():break
        fold=anchors[lo:lo+batch]
        a,data,n=index.edges(fold,radius,family,cap)
        if 'pencil_pairs' in control:deeper_pairs(control['pencil_pairs'],a,data,n,family)
        v=index.compare(a,data,n,nb,family)
        processed+=len(a);compared+=int((n*np.minimum(nb,n)).sum())
        for i,groups in enumerate((two,three,two)):
            subset=v[:,:,i]
            keep=(subset['merit']>-70.) & np.isfinite(subset['merit'])
            groups.append(subset[keep])
        queued+=v.size//2
        if queued>limit*4:
            two=[distinct(two,limit)];three=[distinct(three,limit)];queued=0
    return distinct(two,limit),distinct(three,limit),processed,compared


def search(sky,table,depth,east,pole,control,seconds,threads,device):
    start=time.monotonic();end=start+seconds
    p,e,anchors=foreground(sky,table,depth,east)
    if len(p)<4 or control['event'].is_set():return
    index=Catalog(p,e,pole,device,threads)
    from conditioned_pencil import accompanied, RowIndex
    source_index=RowIndex(table['row'])
    try:
        # Each round covers a different region in physical distance or a
        # different set of anchors. Publish between rounds, so a short run
        # retains its completed orbit tests.
        at=0
        intervals=((43.,0,.40),(43.,0,.38),(34.,1,.45),(80.,1,.46),(46.,2,.64),(140.,2,1.))
        for radius,family,share in intervals:
            left=end-time.monotonic()
            if left<3. or control['event'].is_set():break
            span=left*share
            order=anchors[at:] if family==0 else anchors
            tri,quad,n,compared=scan(index,order,radius,family,.70*span,control)
            if family==0:at+=n
            if not (len(tri) or len(quad)):continue
            proposals=return_confirmation.interleave((quad,tri))
            if control['event'].is_set():break
            cores=accompanied(sky,table,source_index,proposals)
            result,processed=return_confirmation.joint(sky,cores,depth,
                                  min(end-time.monotonic(),max(.3,.30*span)),threads,cap=3600,stratified=True)
            control['raw_results'].put((result,[]))
            trace=control.setdefault('archival_trace',[])
            if len(trace)<10:
                trace.append(proposals['row'][np.linspace(0,len(proposals)-1,min(3000,len(proposals)),dtype=int)].copy())
            print('independent velocity fits:',n,'anchors,',compared,'triangles inspected,',
                  len(quad),'four-night and',len(tri),'three-night proposals,',
                  processed,'tested,',len(result),'five-night returns',flush=True)
    finally:index.close()


def begin(sky,table,depth,east,pole,control,seconds,threads,device):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='velocity-intersections')
    job=pool.submit(search,sky,table,depth,east,pole,control,seconds,threads,device)
    def finish(done):
        if done.exception() is not None:print('velocity intersection search:',repr(done.exception())[:230],flush=True)
    job.add_done_callback(finish)
    pool.shutdown(wait=False)
    return job
