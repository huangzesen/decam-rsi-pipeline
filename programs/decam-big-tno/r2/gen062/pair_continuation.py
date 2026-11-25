"""Search the independent-night pair catalogue for compatible remote visits."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time

import numpy as np
from element_search import UNIT
from survey_arcs import CERTIFIED

CHAIN=np.dtype([('row','i4',4),('g','f4'),('merit','f4')])

def pair_units(pairs):
    units=np.empty(len(pairs),UNIT)
    units['a'],units['b']=pairs['row'][:,:2].T
    units['score'],units['gamma']=pairs['score'],pairs['g']
    return units


def shared_paths(index,signal,pairs,rarity,threads,seconds):
    units=pair_units(pairs)
    units['score']+=.40*np.minimum(rarity[units['a']],rarity[units['b']])
    ptr,integer=ctypes.c_void_p,ctypes.c_int
    fn=index.lib.shared_night_chains
    fn.argtypes=[ptr,integer,ptr,ptr,ptr,integer,integer,ctypes.c_double,integer,ptr,ptr]
    fn.restype=integer
    signal=np.ascontiguousarray(signal,'f4')
    out=np.empty(min(32000,max(400,len(units)//6)),CHAIN)
    stats=np.zeros(4,'i8')
    n=fn(index.target.ctypes.data,len(index.target),index.geometry.ctypes.data,signal.ctypes.data,
         units.ctypes.data,len(units),threads,seconds,len(out),out.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('shared nightly paths failed')
    return out[:n]


def scan(index, pairs, marks, depth, lever, east, seconds, threads):
    ptr, integer, real=ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    fn=index.lib.range_band_search
    fn.argtypes=[ptr,ptr,integer,ptr,ptr,ptr,ptr,integer,integer,real,integer,ptr,ptr]
    fn.restype=integer
    units=pair_units(pairs)
    out=np.empty(10500,CERTIFIED)
    stats=np.zeros(8,'i8')
    n=fn(index.handle, units.ctypes.data,len(units),marks.ctypes.data,depth.ctypes.data,
         lever.ctypes.data,east.ctypes.data,3200,threads,max(.1,seconds),len(out),
         out.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('pair continuation failed')
    return out[:n].copy(),stats


def run(index, job, signal, depth, lever, east, control, rows, seconds, threads, survey_end=None):
    if hasattr(os,'setpriority'):
        try:os.setpriority(os.PRIO_PROCESS,threading.get_native_id(),8)
        except OSError:pass
    # On dense fields the companion search can outlast a continuation
    # allowance. Waiting for input consumes no search time or native worker.
    if survey_end is None:survey_end=control.get('survey_end',float('inf'))
    while not job.done() and time.monotonic()+7.<survey_end:
        if control['event'].wait(.5):return
    if not job.done() or control['event'].is_set():return
    context=job.result()
    seconds=min(seconds,survey_end-time.monotonic())
    if context is None or seconds<7.:return
    end=time.monotonic()+seconds
    rarity,pairs,triangles=context
    sky=control.get('original_sky')
    if sky is not None:
        import three_recovery
        seeds=triangles['row'];gamma=triangles['g'];merit=triangles['score']
        if len(pairs) and end-time.monotonic()>20. and hasattr(index.lib,'shared_night_chains'):
            chains=shared_paths(index,signal,pairs,rarity,threads,min(9.,.08*seconds))
            four=chains[chains['row'][:,3]>=0]
            three=chains[chains['row'][:,3]<0]
            seeds=np.vstack((seeds,three['row'][:,:3]))
            gamma=np.r_[gamma,three['g']];merit=np.r_[merit,three['merit']]
            if len(four) and not control['event'].is_set():
                cores=three_recovery.cores(sky,rows[four['row']],four['g'],four['merit'])
                recovered,_=sky.recover(cores,min(9.,.08*seconds),cap=1300)
                control['raw_results'].put((recovered,[]))
            print('shared-pair paths:',len(three),'triangles,',len(four),'quartets',flush=True)
        three_recovery.submit(control,rows[seeds],gamma,min(34.,.18*seconds),merit)
    if not len(pairs):return
    mag=index.target['mag']
    depth=np.asarray(depth)
    valid=depth[np.isfinite(depth)]
    pivot=np.median(valid) if len(valid) else 23.
    depth=np.ascontiguousarray(np.nan_to_num(depth,nan=pivot),'f8')
    lever,east=(np.ascontiguousarray(v,'f8') for v in (lever,east))
    marks=(.80*np.log(np.clip(signal,4.5,35.)/6.)-.60*np.log1p(((mag-pivot)/.74)**2)).astype('f4')
    marks[:len(rarity)]+=.57*np.maximum(0.,rarity-1.1)
    marks[~np.isfinite(mag)]=-np.inf
    start=0
    for size,fraction in ((22000,.45),(85000,1.)):
        remaining=end-time.monotonic()
        if start>=len(pairs) or remaining<6. or control['event'].is_set():break
        out,stats=scan(index,pairs[start:start+size],marks,depth,lever,east,remaining*fraction,threads)
        start+=max(1,int(stats[0]))
        full=out['nights']>=5
        four=out['nights']==4
        control['certified'].put(out[full|four])
        tri=out[out['nights']==3]
        if len(tri) and control.get('original_sky') is not None and not control['event'].is_set():
            import three_recovery
            three_recovery.submit(control,rows[tri['row'][:,:3]],tri['x'][:,2],
                                  min(24.,.13*seconds),tri['merit'])
        print('pair continuation:',stats.tolist(),'five-night orbits',int(full.sum()),
              'four-night cores',int(four.sum()),flush=True)


def begin(*args):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pair-continuation')
    future=pool.submit(run,*args)
    def done(f):
        if not f.cancelled() and f.exception() is not None:
            print('pair continuation:',type(f.exception()).__name__,str(f.exception())[:160],flush=True)
    future.add_done_callback(done)
    pool.shutdown(wait=False)
    return future
