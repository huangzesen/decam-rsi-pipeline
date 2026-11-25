"""Follow a measured source through the independent visits covering its allowed orbits."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from survey_arcs import CERTIFIED


def priorities(target, ids, base, repeat, depths):
    q = np.full(len(target), -6., 'f4')
    ex = target['exposure']
    mag = target['mag']
    good = np.isfinite(mag) & (mag > 0.)
    mb = np.clip(np.nan_to_num((mag-14.)*3., nan=0.), 0, 47).astype('i4')
    key = ex*48+mb
    counts = np.bincount(key[good], minlength=len(depths)*48).reshape(-1,48)
    density = .5*counts + .25*(np.roll(counts,1,axis=1)+np.roll(counts,-1,axis=1))
    finite = depths[np.isfinite(depths)]
    center = np.median(finite) if len(finite) else np.nanmedian(mag)
    marks = base.copy()
    if repeat is not None:
        marks += .62*np.minimum(6.,repeat[:,0])
    # Compare source quality at the same apparent magnitude and in the same
    # exposure. Otherwise nearly every endpoint comes from a bright-only tail.
    level = np.clip((3.*(marks+12.)).astype('i4'), 0, 95)
    cell = key[ids]
    hist = np.bincount(cell*96+level, minlength=len(depths)*48*96).reshape(-1,96)
    cum = np.cumsum(hist,axis=1)
    rank = (cum[cell,level]-.5*hist[cell,level]+5.)/(cum[cell,-1]+10.)
    q[ids] = (.72*marks-.8*np.log(np.maximum(.025,1.-rank))).astype('f4')
    q += (-.52*np.log1p(((mag-center)/.75)**2)
          -.46*np.log1p(density.ravel()[key]/30.)).astype('f4')
    q[~good] = -100.
    return q


def proposals(index, q, depth, lever, east, family, limit, threads, seconds, cap=10000):
    fn = index.lib.encounter_orbits
    p,i,d = ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    fn.argtypes = [p,p,p,p,p,p,i,i,d,i,p,p]
    fn.restype = i
    params = np.ascontiguousarray(family,'f8')
    depth = np.ascontiguousarray(depth,'f8')
    if not np.isfinite(depth).all():
        valid=depth[np.isfinite(depth)]
        fallback=np.median(valid) if len(valid) else np.nanmedian(index.target['mag'])
        depth=np.where(np.isfinite(depth),depth,fallback)
    lever = np.ascontiguousarray(lever,'f8')
    east = np.ascontiguousarray(east,'f8')
    q = np.ascontiguousarray(q,'f4')
    results = np.empty(cap,CERTIFIED)
    stats = np.zeros(8,'i8')
    n = fn(index.handle,q.ctypes.data,depth.ctypes.data,lever.ctypes.data,east.ctypes.data,
           params.ctypes.data,limit,threads,seconds,cap,results.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('source-conditioned orbit search failed')
    return results[:n].copy(),stats


def work(index, ids, base, repeat, depths, lever, east, seconds, threads, control,ex,pole,device):
    end = time.monotonic()+seconds
    if device!='cpu' and hasattr(os,'setpriority'):
        # The main CUDA search needs its host threads to keep feeding the GPU.
        # On Linux the native threads created below inherit this priority.
        try:
            tid=threading.get_native_id()
            os.setpriority(os.PRIO_PROCESS,tid,min(19,os.getpriority(os.PRIO_PROCESS,tid)+3))
        except (OSError,AttributeError):pass
    q = priorities(index.target,ids,base,repeat,depths)
    if seconds>=95. and not control['event'].is_set():
        import moving_votes
        votes,number=moving_votes.measure(index,ids,q[ids],ex,pole,min(50.,.12*seconds),
                                          device,control['event'])
        q[ids]+=.65*votes
        print('moving sky cells',number,'orbital trials',flush=True)
    families = [
        ((43.,.14,1.,.09,.16,.20,4800.),6400,.59),
        ((115.,.40,.65,.54,.64,.52,4800.),6400,.36),
        ((340.,.53,.42,.70,.74,.55,4800.),5200,.56),
        ((1700.,.62,.28,.79,.79,.63,4800.),5200,1.),
    ]
    for family,limit,share in families:
        left = end-time.monotonic()
        if left<6. or control['event'].is_set():break
        result,stats = proposals(index,q,depths,lever,east,family,limit,threads,left*share)
        control['certified'].put(result)
        control.setdefault('encounter_groups',[]).append(result)
        print('source-conditioned visits',family[0],stats.tolist(),'reports',len(result),flush=True)


def begin(index, ids, base, repeat, depths, lever, east, seconds, threads, control,ex,pole,device):
    pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='source-visits')
    job = pool.submit(work,index,ids,base,repeat,depths,lever,east,seconds,threads,control,ex,pole,device)
    def report(done):
        if done.exception() is not None:
            print('source-conditioned visits:',str(done.exception())[:250],flush=True)
    job.add_done_callback(report)
    pool.shutdown(wait=False)
    return job


def write_trace(G,directory):
    control=G.get('survey_control') or {}
    arrays=list(control.get('encounter_groups',()))
    if not arrays:return
    primary=G['keep_np'];additional=control['additional']['rows']
    path=Path(directory)/'trace.json'
    if path.exists() and path.stat().st_size>48_000_000:return
    stages=json.loads(path.read_text())['stages'] if path.exists() else []
    groups=[]
    for a in arrays:
        # Vectorize the index conversion before constructing the JSON lists.
        n=a['count'];valid=np.arange(a['row'].shape[1])[None,:]<n[:,None]
        rows=a['row'][valid];core=rows<len(primary)
        raw=np.empty(len(rows),'i8');raw[core]=primary[rows[core]];raw[~core]=additional[rows[~core]-len(primary)]
        cuts=np.r_[0,np.cumsum(n)]
        groups.extend(raw[lo:hi].tolist() for lo,hi in zip(cuts[:-1],cuts[1:]))
    stage=dict(name='Source-conditioned visits with Kepler verification',groups=groups)
    if len(stages)>=6:
        replace=next((i for i,s in enumerate(stages) if 'kept for the carry' in s['name']),len(stages)-1)
        stages[replace]=stage
    else:stages.append(stage)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as fp:json.dump(dict(stages=stages[:6]),fp,separators=(',',':'))
    os.replace(tmp,path)
