"""Prioritize measurements supported by orbital motion on independent visits."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np


VISIT = np.dtype([(k, 'f8') for k in ('t', 'ox', 'oy', 'oz', 'depth', 'lever')] +
                 [('night', 'i4'), ('pad', 'i4')])
LINK = np.dtype([('row', 'i4'), ('score', 'f4'), ('gamma', 'f4')])
RESULT = np.dtype([(k, 'f4') for k in ('evidence', 'burst', 'three', 'control')] +
                  [('rows', 'i4', 3), ('gamma', 'f4'), ('pair', LINK, 2)])


@lru_cache(maxsize=None)
def build_library(name='companion_graph'):
    cc = next((p for c in ('c++', 'g++', 'clang++') if (p := shutil.which(c))), None)
    if not cc:
        raise RuntimeError('no host compiler for motion graph')
    with tempfile.TemporaryDirectory(prefix='companions_') as work:
        path = Path(work) / 'graph.so'
        subprocess.run([cc, '-O3', '-std=c++17', '-shared', '-fPIC', '-pthread',
                        str(Path(__file__).with_name(name+'.cpp')), '-o', str(path)],
                       check=True, capture_output=True, timeout=45.)
        return ctypes.CDLL(str(path))


def measure(xy, mag, sn, mark, ex, ids, visits, east, threads, library=None):
    ids = np.ascontiguousarray(ids, 'i4')
    edges = np.ascontiguousarray(np.searchsorted(ex[ids], np.arange(len(visits)+1)), 'i4')
    out = np.empty(len(ids), RESULT)
    lib = library or build_library()
    f = lib.companion_graph
    ptr, integer, real = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    f.argtypes = [ptr]*8+[integer, real, real, integer, ptr]
    f.restype = integer
    arrays = [np.ascontiguousarray(a, 'f4') for a in (xy, mag, sn, mark)]
    arrays.extend([np.ascontiguousarray(ex, 'i4'), ids, edges, visits])
    done = f(*(a.ctypes.data for a in arrays), len(visits), *map(float, east),
             threads, out.ctypes.data)
    if done != len(ids):
        raise RuntimeError('incomplete cross-visit graph')
    return out


def build(scen, xy, mag, sn, rows, wide, ex, nights, times, ob, depths, lever, east,
          threads=2, library=None):
    t = time.monotonic()
    raw = Path(scen)
    sharp = np.load(raw/'det_sharp.npy', mmap_mode='r')[rows]
    flag = np.load(raw/'det_flag.npy', mmap_mode='r')[rows]
    sigma = np.hypot(.19, 3.6/np.maximum(sn, 4.5))
    shape = np.nan_to_num((sharp-1.04)/sigma, nan=6.)
    valid = ((flag&1)==0) & np.isfinite(mag) & ((wide | (sn>5.4)) & (abs(shape)<3.3))
    ids = np.flatnonzero(valid)
    if not len(ids):
        return None
    exposure_order=ex[ids]
    if np.any(exposure_order[1:]<exposure_order[:-1]):
        ids=ids[np.argsort(exposure_order,kind='stable')]
    # A source mark multiplies the positional intensity. Estimate its mean
    # within each comparison population so the unrelated sky has mean one.
    feature = np.clip(.47*np.log(np.maximum(4.5,sn)/6.)-.31*shape**2
                      -.65*((flag&2)!=0)-.34*((flag&4)!=0), -3., 1.7)
    level = np.clip(np.nan_to_num(2.*(mag[ids]-12.),nan=0.).astype('i4'),0,35)
    key = 36*ex[ids]+level
    count = np.bincount(key, minlength=36*len(times))
    mass = np.bincount(key, weights=np.exp(feature[ids]), minlength=len(count))
    feature[ids] -= np.log((mass[key]+3.)/(count[key]+3.))
    visits = np.zeros(len(times), VISIT)
    visits['t'], visits['night'], visits['lever'] = times, nights, lever
    visits['ox'], visits['oy'], visits['oz'] = ob.T
    visits['depth'] = np.where(np.isfinite(depths), depths, np.nanmedian(mag[ids]))
    data = measure(np.ascontiguousarray(xy*206264.806247, 'f4'), mag, sn, feature, ex,
                   ids, visits, east, threads, library)
    # The reversed displacement measures the chance evidence from dense
    # fields at the same cadence. Use its distribution, not an assumed null
    # independent of coverage, to scale the actual cross-night evidence.
    controls = np.clip(data['control'], -8., 12.)
    nbin = 72
    k = np.clip(((controls+8.)*3.).astype('i4'), 0, nbin-1)
    h = np.bincount(key*nbin+k, minlength=36*len(times)*nbin).reshape(-1,nbin)
    tail = h[:,::-1].cumsum(1)[:,::-1]
    k = np.clip(((data['evidence']+8.)*3.).astype('i4'), 0, nbin-1)
    rarity = -np.log((tail[key,k]-.45*h[key,k]+3.)/(tail[key,0]+5.))
    value = .49*np.clip(data['evidence'], -1.8, 8.) + .43*np.clip(rarity-.7, -.7, 4.7)
    value += .46*np.clip(data['burst'], -2.8, 5.5) + .48*np.clip(data['three']-4., 0., 6.)
    priority = np.zeros(len(rows),'f4')
    priority[ids] = np.clip(value, -2.6, 8.8)
    print('motion companions:', len(ids), 'measurements,', int((data['three']>4.).sum()),
          'local arcs in', round(time.monotonic()-t,1), 's', flush=True)
    return dict(priority=priority, ids=np.asarray(ids,'i4'), data=data)


def start(*args):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cross-visit-motion')
    job=pool.submit(build,*args)
    def finished(done):
        if done.exception() is not None:
            print('motion companions:',repr(done.exception())[:180],flush=True)
    job.add_done_callback(finished)
    pool.shutdown(wait=False)
    return job


def ready(G):
    job=G.get('companion_graph')
    return job.result() if job is not None and job.done() and job.exception() is None else None


def proposals(graph, index):
    import attributables
    d=graph['data']
    take=np.flatnonzero((d['three']>0.)&(d['gamma']>0.))
    rows=np.sort(d['rows'][take],axis=1)
    order=np.argsort(-d['three'][take],kind='stable')
    _,first=np.unique(rows[order],axis=0,return_index=True)
    use=take[order[np.sort(first)]]
    seed=np.zeros(len(use),attributables.SEED)
    seed['rows'][:,:3]=d['rows'][use]
    seed['rows'][:,3]=-1
    seed['score'],seed['gamma']=d['three'][use],d['gamma'][use]
    return seed


def edge_bank(graph, target, visits, limit=680000):
    data=graph['data']
    rows=graph['ids']
    left=np.repeat(rows,2)
    pairs=data['pair'].ravel()
    take=np.flatnonzero((pairs['score']>-4.2)&(pairs['row']>=0))
    a,b=left[take],pairs['row'][take]
    ea,eb=target['exposure'][a],target['exposure'][b]
    dt=np.abs(visits['t'][ea]-visits['t'][eb])
    local=visits['night'][ea]==visits['night'][eb]
    # A pair only minutes long rarely measures an angular velocity.
    # Retain it when the motion relative to a stationary hypothesis is strong.
    sep=206264.806247*np.hypot(target['x'][a]-target['x'][b],target['y'][a]-target['y'][b])
    ok=(dt>.0004)&(~local|((sep>.53)&(pairs['score'][take]>1.2)&(data['burst'][take//2]>.30)))
    a,b,take,dt,local=[x[ok] for x in (a,b,take,dt,local)]
    pri=graph['priority']
    merit=pairs['score'][take]+.30*(pri[a]+pri[b])+np.where(local,-1.6,0.)
    keys=(np.minimum(a,b).astype('i8')<<32)|np.maximum(a,b)
    order=np.lexsort((-merit,keys))
    use=order[np.r_[True,keys[order[1:]]!=keys[order[:-1]]]] if len(order) else order
    # Vary the on-sky speed, baseline and brightness of anchors admitted
    # during any prefix of a time-limited scan.
    brightness=np.clip(np.nan_to_num((target['mag'][a]+target['mag'][b])-26.,nan=0.).astype('i4'),0,35)
    group=(np.digitize(dt,[.45,1.6,4.5,10.])*36+brightness)*2+(pairs['gamma'][take]<1./90.)
    k=np.clip(((merit+8.)*3.).astype('i4'),0,100)
    hist=np.bincount(group[use]*101+k[use],minlength=360*101).reshape(-1,101)
    tail=hist[:,::-1].cumsum(1)[:,::-1]
    rarity=-np.log((tail[group,k]-.48*hist[group,k]+1.)/(tail[group,0]+3.))
    select=merit[use]+.52*rarity[use]+np.where(local[use],-.8,.35*np.log1p(dt[use]))
    order=use[np.argsort(-select,kind='stable')[:limit]]
    edge=np.zeros(len(order),[('a','i4'),('b','i4'),('score','f4'),('gamma','f4')])
    edge['a'],edge['b']=a[order],b[order]
    edge['score'],edge['gamma']=merit[order],pairs['gamma'][take[order]]
    return edge


def join_chords(target, visits, bank, east, threads, seconds, cap=85000):
    import attributables
    if not len(bank):return np.empty(0,attributables.SEED),np.zeros(3,'i8')
    lib=build_library('chord_join')
    fn=lib.join_motion_chords
    ptr,integer,real=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    fn.argtypes=[ptr,ptr,ptr,integer,real,real,integer,real,integer,ptr,ptr]
    fn.restype=integer
    seeds=np.empty(cap,attributables.SEED)
    count=np.zeros(3,'i8')
    n=fn(target.ctypes.data,visits.ctypes.data,bank.ctypes.data,len(bank),*map(float,east),
         threads,seconds,cap,seeds.ctypes.data,count.ctypes.data)
    if n<0:raise RuntimeError('motion chord matching failed')
    return seeds[:n].copy(),count


def follow(index, job, east, control, seconds, threads, exposure=None, lever=None, device='cpu'):
    import attributables
    def work():
        graph=job.result()
        if not graph or control['event'].is_set():return
        seeds=proposals(graph,index)
        if len(seeds):
            complete=attributables.follow(index,seeds,threads,.10*seconds,cap=3500,epro=east)
            control['certified'].put(complete[complete['nights']>=5])
            if control['event'].is_set():return
            more,stats=attributables.surface(index,seeds,east,threads,.20*seconds,refine=True,limit=6500)
            control['certified'].put(more)
            print('cross-visit arc ranging:',len(seeds),'seeds,',len(complete),
                  'fits and',int((more['nights']>=5).sum()),'continued orbits',flush=True)
        if exposure is None or control['event'].is_set():return
        visits=np.zeros(len(exposure),VISIT)
        for k in ('t','ox','oy','oz','night'):visits[k]=exposure[k]
        visits['lever']=lever
        bank=edge_bank(graph,index.target,visits)
        if control['event'].is_set():return
        quads=None
        if str(device).startswith('cuda'):
            try:
                import chord_device
                quads,stats=chord_device.search(index.target,visits,bank,east,.35*seconds,112000,control['event'])
                print('device motion chords:',stats.tolist(),flush=True)
            except (RuntimeError,OSError,ValueError,MemoryError,ArithmeticError,AttributeError,
                    TypeError,subprocess.SubprocessError) as exc:
                print('host motion chord search:',str(exc)[:180],flush=True)
        if quads is None and not control['event'].is_set():
            quads,stats=join_chords(index.target,visits,bank,east,threads,.31*seconds)
        if quads is None:return
        if len(quads) and not control['event'].is_set():
            complete=attributables.follow(index,quads,threads,.37*seconds,cap=11000,epro=east)
            control['certified'].put(complete)
            print('cross-season motion chords:',stats.tolist(),'proposals',len(quads),
                  'five-night orbits',int((complete['nights']>=5).sum()),flush=True)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='companion-continuation')
    future=pool.submit(work)
    def finished(done):
        if done.exception() is not None:
            print('cross-visit arc ranging:',repr(done.exception())[:160],flush=True)
    future.add_done_callback(finished)
    pool.shutdown(wait=False)
    return future
