"""Search full-depth visits conditional on two measured angular positions."""
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
import time

import numpy as np

from companion_graph import build_library
from original_recovery import integer,ptr,real
import return_confirmation
import transient_arcs as visits
import velocity_pencil as pencil


GRID=np.dtype([(k,'f4') for k in ('xmin','xmax','ymin','ymax')]+[('density','f4',60)]+
              [(k,'i4') for k in ('bx','by','nx','ny','offset')])
PAIR=np.dtype([(k,pencil.POINT) for k in ('a','b','peer')]+[('g','f4')])
HIT=np.dtype([('id','i4'),('merit','f4')])
PITCH=108./206264.80624709636


def pack(sky,rows,rotation):
    rows=np.asarray(rows,'i4')
    p=np.empty(len(rows),pencil.POINT)
    xy=sky.pos[rows]
    p['x']=xy[:,0]*rotation[0]+xy[:,1]*rotation[1]
    p['y']=xy[:,1]*rotation[0]-xy[:,0]*rotation[1]
    p['row']=rows;p['ex']=sky.exposure[rows]
    p['mag']=sky.mag[rows];sn=sky.arrays[0][rows]
    p['var']=.14**2+(1.086/np.maximum(4.5,sn))**2
    shape=(sky.arrays[1][rows]-1.05)/np.hypot(.21,3.6/np.maximum(4.5,sn))
    p['q']=(.53*np.log(np.clip(sn,4.5,30.)/6.)-.29*np.minimum(12.,shape*shape)-
            .55*((sky.arrays[5][rows]&2)!=0)-.29*((sky.arrays[5][rows]&4)!=0))
    return p


def organize(p,e):
    grid=np.zeros(len(e),GRID)
    order=np.argsort(p['ex'],kind='stable')
    if len(order) and np.any(order!=np.arange(len(order))):p=p[order]
    edges=np.searchsorted(p['ex'],np.arange(len(e)+1))
    ordered=np.empty_like(p)
    # Build a column index per exposure. Each local grid stores offsets
    # into one contiguous array of sources shared by all pair hypotheses.
    parts=[];total=0
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        if hi<=lo:continue
        points=p[lo:hi];g=grid[j]
        x=points['x'];y=points['y']
        g['xmin'],g['xmax'],g['ymin'],g['ymax']=x.min(),x.max(),y.min(),y.max()
        bx=np.floor(x.astype('f8')/PITCH).astype('i4');by=np.floor(y.astype('f8')/PITCH).astype('i4')
        g['bx'],g['by']=bx.min(),by.min();bx-=g['bx'];by-=g['by']
        nx=int(bx.max())+1;ny=int(by.max())+1
        if nx*ny>2400000:continue
        g['nx'],g['ny'],g['offset']=nx,ny,total
        key=bx*ny+by
        counts=np.bincount(key,minlength=nx*ny).astype('i4')
        lookup=np.empty(nx*ny+1,'i4');lookup[0]=lo
        np.cumsum(counts,out=lookup[1:]);lookup[1:]+=lo
        parts.append(lookup);total+=len(lookup)
        ordered[lo:hi]=points[np.argsort(key,kind='stable')]
        mb=np.clip(((points['mag']-10.)*3.).astype('i4'),0,59)
        h=np.convolve(np.bincount(mb,minlength=60),[.05,.21,.48,.21,.05],'same')
        area=max(3600.,min((x.max()-x.min())*(y.max()-y.min())*206264.806247**2,
                 (counts>0).sum()*108.**2/max(.16,1.-np.exp(-(hi-lo)/(nx*ny)))))
        g['density']=np.log(np.maximum(2.e-9,(3.*h+1.)/area))
    return ordered,grid,np.concatenate(parts) if parts else np.empty(0,'i4')


class Forecasts:
    def __init__(self,p,e,pole,device,threads):
        self.points,self.grid,self.offset=organize(p,e)
        self.e=e;self.pole=pole;self.threads=threads
        lib=build_library('velocity_pencil')
        self.host=lib.conditioned_host
        self.host.argtypes=[ptr,ptr,ptr,ptr,integer,ptr,integer,real,real,integer,ptr]
        self.host.restype=None
        self.fitted=lib.conditioned_fits
        self.fitted.argtypes=[ptr,ptr,ptr,integer,ptr,integer,real,real,integer,ptr,ptr]
        self.fitted.restype=None
        self.kernel=None;self.checked=False
        if device=='cuda':
            try:
                import torch
                from apparent_backend import cuda_kernel
                if torch.cuda.mem_get_info()[0]<self.points.nbytes+300000000:raise MemoryError('device index space in use')
                self.kernel=cuda_kernel(pencil.device_source(),'conditioned_visits')
                self.gpu=[torch.as_tensor(a.view('u1'),device='cuda') for a in
                          (self.points,self.grid,self.offset,e)]
            except (OSError,RuntimeError,ValueError,ImportError,MemoryError) as exc:
                self.kernel=None
                print('conditional forecasts using CPU:',str(exc)[:150],flush=True)

    def host_hits(self,pairs):
        n=len(pairs);ne=len(self.e);hits=np.empty((n,ne,2),HIT)
        self.host(self.points.ctypes.data,self.grid.ctypes.data,self.offset.ctypes.data,
                  self.e.ctypes.data,ne,pairs.ctypes.data,n,*self.pole,self.threads,hits.ctypes.data)
        return hits

    def lookup(self,pairs):
        n=len(pairs);ne=len(self.e)
        if self.kernel is not None:
            try:
                import torch
                p=torch.as_tensor(pairs.view('u1'),device='cuda')
                out=torch.empty(n*ne*2*HIT.itemsize,dtype=torch.uint8,device='cuda')
                self.kernel(grid=(n,(ne+63)//64,1),block=(64,1,1),
                            args=[*self.gpu,ne,p,n,*self.pole,out])
                hits=out.cpu().numpy().view(HIT).reshape(n,ne,2)
                if not self.checked and (hits['id']>=0).any():
                    use=np.flatnonzero((hits['id']>=0).any(axis=(1,2)))[:12]
                    check=self.host_hits(pairs[use])
                    if not np.array_equal(hits['id'][use],check['id']):
                        raise ArithmeticError('device sky index disagrees with host index')
                    self.checked=True
                return hits
            except (RuntimeError,ArithmeticError,ValueError,MemoryError) as exc:
                print('conditional sky lookup continuing on CPU:',str(exc)[:150],flush=True)
                self.kernel=None
        return self.host_hits(pairs)

    def search(self,pairs):
        hits=self.lookup(pairs)
        out=np.empty((len(pairs),6),visits.PROPOSAL)
        self.fitted(self.points.ctypes.data,self.grid.ctypes.data,self.e.ctypes.data,len(self.e),
                    pairs.ctypes.data,len(pairs),*self.pole,self.threads,hits.ctypes.data,out.ctypes.data)
        return out


def build(sky,table,extra,depth,east):
    # Every measured foreground endpoint can supply the third or fourth
    # night. Additional low-S/N or partially masked rows receive a lower
    # prior, while arc confirmation can use the entire original catalogue.
    rows=table['row']
    source=pack(sky,rows,east)
    quality=table['single']+.50*np.clip(table['merit']-table['single'],0.,8.)
    source['q']+=np.clip(.45*(quality-np.median(quality)),-3.,3.5)
    observed=np.zeros(len(sky.mag),bool);observed[rows]=True
    other=np.asarray(extra,dtype='i4')
    other=other[~observed[other]]
    if len(other):
        flag=sky.arrays[5][other];sn=sky.arrays[0][other];sharp=sky.arrays[1][other]
        use=((flag&1)==0)&np.isfinite(sky.mag[other])&np.isfinite(sharp)&(sn>=4.49)
        use&=np.abs(sharp-1.05)<3.2*np.hypot(.22,3.7/np.maximum(4.5,sn))
        other=other[use]
        if len(other)>2300000:
            # Take a uniform part of the faint tail from each exposure.
            other=other[np.linspace(0,len(other)-1,2300000,dtype=int)]
        aux=pack(sky,other,east);aux['q']-=1.5
        source=np.concatenate((source,aux))
    e=np.zeros(len(depth),pencil.VISIT);geom=sky.geometry
    e['t'],e['night'],e['oz']=geom['t'],geom['night'],geom['oz']
    e['ox']=east[0]*geom['ox']+east[1]*geom['oy']
    e['oy']=east[0]*geom['oy']-east[1]*geom['ox']
    e['lever'],e['depth']=sky.lever,depth
    return source,e


def pair_batch(sky,table,lookup,edges,east):
    n=len(edges);result=np.empty(n,PAIR)
    a,b=edges['a'].copy(),edges['b'].copy()
    # Choose the better observed end as the anchor. A second shutter
    # there supplies a third angle with its own observer position.
    k=lookup[a];j=lookup[b]
    swap=(table['peer'][k]<0)&(table['peer'][j]>=0)
    a[swap],b[swap]=b[swap],a[swap]
    k=lookup[a];j=lookup[b]
    result['a'],result['b']=pack(sky,a,east),pack(sky,b,east)
    result['a']['q']+=.38*np.clip(table['merit'][k],-6.,8.)
    result['b']['q']+=.38*np.clip(table['merit'][j],-6.,8.)
    peers=table['peer'][k]
    result['peer']=pack(sky,np.maximum(peers,0),east)
    result['peer']['row']=peers
    result['g']=np.clip(edges['gamma'],1./9000.,1./17.)
    return result


class RowIndex:
    def __init__(self,rows):
        self.order=np.argsort(rows,kind='stable').astype('i4') if np.any(rows[1:]<rows[:-1]) else None
        self.keys=rows if self.order is None else rows[self.order]

    def __getitem__(self,rows):
        loc=np.minimum(np.searchsorted(self.keys,rows),len(self.keys)-1)
        return np.where(self.keys[loc]==rows,loc if self.order is None else self.order[loc],-1)


def accompanied(sky,table,lookup,proposals):
    """Refit also with available burst centroids, up to one per seed night."""
    cores=return_confirmation.prepare(sky,proposals)
    n=len(cores)
    if not n:return cores
    augmented=cores.copy();changed=np.zeros(n,bool);order=np.arange(n)
    for j in range(4):
        r=proposals['row'][:,j];k=lookup[np.maximum(r,0)]
        mate=table['peer'][np.maximum(k,0)]
        eligible=(r>=0)&(k>=0)&(mate>=0)
        used=np.arange(8)<augmented['count'][:,None]
        eligible&=~((augmented['rows'][:,:8]==mate[:,None])&used).any(1)
        use=order[eligible]
        augmented['rows'][use,augmented['count'][use]]=mate[use]
        augmented['count'][use]+=1;changed[use]=True
    # An unrelated same-night measurement need not invalidate the clean
    # seed. The two alternatives compete in raw orbital confirmation.
    return return_confirmation.interleave((augmented[changed],cores))


def work(sky,table,edges,extra,depth,east,pole,control,seconds,threads,device):
    end=time.monotonic()+seconds
    p,e=build(sky,table,extra,depth,east)
    index=Forecasts(p,e,pole,device,threads)
    rows=len(p);del p
    lookup=np.full(len(sky.mag),-1,'i4');lookup[table['row']]=np.arange(len(table),dtype='i4')
    # Some local pairs come from nearby raw rows below the primary cut.
    edges=edges[(lookup[edges['a']]>=0)&(lookup[edges['b']]>=0)]
    count=0;tri=[];quad=[];ready=0;completed=0;chords=0
    def verify(allowance):
        nonlocal tri,quad,ready,completed
        proposals=return_confirmation.interleave((pencil.distinct(quad,24000),pencil.distinct(tri,30000)))
        tri=[];quad=[];ready=0
        if not len(proposals) or control['event'].is_set():return
        cores=accompanied(sky,table,lookup,proposals)
        result,done=return_confirmation.joint(sky,cores,depth,allowance,threads,cap=3400,stratified=True)
        control['raw_results'].put((result,[]));completed+=len(result)
        print('pair-conditioned orbital forecasts:',rows,'catalogue points,',count,'pairs including',chords,'long-arc seeds,',
              len(proposals),'proposals,',done,'raw fits,',len(result),'confirmed',flush=True)
    batch=256 if index.kernel is not None else 112
    start=time.monotonic();last=start
    extra_pairs=np.empty(0,visits.PAIR);queued=control.get('pencil_pairs');seen=set()
    for lo in range(0,len(edges),batch):
        now=time.monotonic()
        if now+max(1.,.12*seconds)>=end or control['event'].is_set():break
        if queued is not None and len(extra_pairs)<batch//2:
            try:extra_pairs=np.r_[extra_pairs,queued.get_nowait()]
            except Empty:pass
        part=edges[lo:lo+batch]
        if len(extra_pairs):
            use=extra_pairs[:batch//2];extra_pairs=extra_pairs[len(use):]
            accept=[]
            for j,c in enumerate(use):
                aa,bb=int(c['a']),int(c['b'])
                key=(min(aa,bb),max(aa,bb),round(7.*np.log(float(c['gamma']))))
                if key not in seen:accept.append(j);seen.add(key)
            use=use[accept];chords+=len(use)
            part=np.r_[part,use]
        pair=pair_batch(sky,table,lookup,part,east)
        found=index.search(pair)
        for group,values in ((tri,found[:,:2]),(quad,found[:,2:])):
            keep=(values['merit']>-70.)&np.isfinite(values['merit'])
            group.append(values[keep])
        ready+=int((found['merit']>-70.).sum());count+=len(pair)
        if ready>65000 or (now-last>min(44.,.22*seconds) and ready>700):
            verify(min(18.,max(2.,.08*seconds)));last=time.monotonic()
    if ready and not control['event'].is_set():verify(max(.2,end-time.monotonic()))


def begin(*args):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='conditioned-forecasts')
    job=pool.submit(work,*args)
    def finish(f):
        if f.exception() is not None:print('conditioned forecast search:',repr(f.exception())[:200],flush=True)
    job.add_done_callback(finish);pool.shutdown(wait=False)
    return job
