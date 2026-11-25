"""Search the overlap of visits along moving, heliocentric trajectories."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np

from survey_arcs import CERTIFIED


def remember(control, orbits, family='visits'):
    if 'primary_rows' not in control:return
    primary=control['primary_rows'];extra=control['additional']['rows']
    # Preserve both independent-night associations and promising incomplete
    # fits. The published groups refer to original catalogue row numbers.
    groups=control.setdefault('cadence_trace',{}).setdefault(family,{})
    if len(groups)>=16000:return
    for u in orbits:
        d=u['row'][:u['count']]
        key=np.sort(d).tobytes()
        if key in groups:continue
        first=d<len(primary);rows=np.empty(len(d),dtype='i4')
        rows[first]=primary[d[first]];rows[~first]=extra[d[~first]-len(primary)]
        groups[key]=rows
        if len(groups)>=16000:break


def write_trace(control, directory):
    collections=list(control.get('cadence_trace',{}).values())
    groups={key:rows for batch in collections for key,rows in batch.copy().items()}
    if not groups:return
    path=Path(directory)/'trace.json'
    with path.open() as fp:stages=json.load(fp)['stages']
    added=dict(name='Visit pairs and moving cells, Kepler fits',groups=[rows.tolist() for rows in groups.values()])
    if len(stages)>=6:
        k=next((i for i,s in enumerate(stages) if 'kept for the carry' in s['name']),len(stages)-1)
        stages[k]=added
    else:stages.append(added)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as fp:json.dump(dict(stages=stages[:6]),fp,separators=(',',':'))
    if tmp.stat().st_size<=100000000:os.replace(tmp,path)
    else:tmp.unlink()


def search(index, ids, quality, exposure, depth, lever, east, seconds, threads, control,
           schedule=((0,.47),(3,.23),(1,.64),(2,1.))):
    # Include an independent proposal tier from the verification catalogue.
    # Source quality and sampling are conditional on the local magnitude bin.
    end = time.monotonic() + seconds
    target = index.target
    ns = len(target)
    if ns < 5 or len(ids) == 0:
        return
    q = np.full(ns, -5., 'f4')
    q[ids] = quality
    good = np.isfinite(target['mag']) & (target['mag'] > 0.)
    mark = np.clip(np.floor((np.nan_to_num(target['mag'], nan=30.)-12.)*2), 0, 39).astype('i4')
    bins = target['exposure']*40 + mark
    # A genuine marker in a sparsely populated magnitude interval should
    # still contribute when the exposure is a very deep one.
    density = np.bincount(bins[good], minlength=len(exposure)*40).reshape(-1,40)
    smooth = .5*density + .25*(np.roll(density,1,axis=1) + np.roll(density,-1,axis=1))
    bright = np.asarray(depth, 'f8')
    finite = bright[np.isfinite(bright)]
    pivot = np.median(finite) if len(finite) else np.nanmedian(target['mag'][ids])
    bright = np.where(np.isfinite(bright),bright,pivot)
    quality = np.asarray(quality, 'f4')
    qloc = np.clip(np.floor((quality + 20.)*4.),0,119).astype('i4')
    grouped = bins[ids]
    table = np.bincount(grouped*120 + qloc, minlength=len(density)*40*120).reshape(-1,120)
    cdf = np.cumsum(table, axis=1)
    percentile = (cdf[grouped,qloc]-.5*table[grouped,qloc]+3.)/(cdf[grouped,-1]+6.)
    q[ids] = .62*(quality - np.median(quality)) - .65*np.log(np.maximum(.025,1.-percentile))
    # Empirical abundance of background peaks per magnitude. The smooth
    # source-count law and visibility both vary continuously across visits.
    z = target['mag'] - pivot
    q += (.34*np.clip(z,-5.,2.) - .50*np.maximum(z-.30,0.)**2
          - .30*np.log1p(smooth.ravel()[bins]/25.)).astype('f4')
    q[~good] = -100.
    depths = np.ascontiguousarray(bright, 'f8')
    levers = np.ascontiguousarray(lever, 'f8')
    east = np.ascontiguousarray(east, 'f8')
    func = index.lib.cadence_orbits
    p, i, d = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    func.argtypes = [p,p,p,p,p,i,i,d,i,i,p,p]
    func.restype = i
    # Each pass builds a separate visit schedule and publishes its result.
    # A partial run therefore still searches several fields and cadences.
    for mode, share in schedule:
        room = end-time.monotonic()
        if room < 3. or control['event'].is_set():
            break
        result = np.empty(12000, CERTIFIED)
        stats = np.zeros(8,'i8')
        n = func(index.handle,q.ctypes.data,depths.ctypes.data,levers.ctypes.data,east.ctypes.data,
                 len(exposure),threads,room*share,mode,len(result),result.ctypes.data,stats.ctypes.data)
        if n < 0:
            raise RuntimeError('visit triangle search failed')
        confirmed=result[:n].copy()
        control['certified'].put(confirmed)
        remember(control,confirmed)
        print('visit triangles',mode,stats.tolist(),'orbits',n,flush=True)
        if control.get('original_sky') is not None and not control['event'].is_set():
            import original_recovery
            original_recovery.verify_cores(control,confirmed,min(12.,max(1.,end-time.monotonic())))


def begin(index, ids, quality, ex, depths, lever, east, seconds, threads, control):
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='visit-triangles')
    job = pool.submit(search,index,ids,quality,ex,depths,lever,east,seconds,threads,control)
    def finished(future):
        if future.exception() is not None:
            print('visit triangle search:',str(future.exception())[:220],flush=True)
    job.add_done_callback(finished)
    pool.shutdown(wait=False)
    return job
