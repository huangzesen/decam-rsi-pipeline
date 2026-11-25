"""Combine two independent three-night measurements at a common detection."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import threading
import time

import numpy as np
from companion_graph import build_library

SEED=np.dtype([('row','i4',3),('gamma','f4'),('score','f4'),('serial','i4')])
POINT=np.dtype([('x','f4'),('y','f4'),('exposure','i4'),('night','i4')])
TRIAL=np.dtype([('row','i4',5),('gamma','f4'),('score','f4')])
LOCK=threading.Lock()

def intersections(parts,sky,previous,seconds,cap=5500):
    seeds=np.concatenate(parts)
    good=(seeds['gamma']>.000008)&(seeds['gamma']<.087)&np.isfinite(seeds['score'])
    seeds=seeds[good]
    if len(seeds)<2:return np.empty(0,TRIAL),[0,0,0]
    # Deduplicate alternative fits of the same three measured directions.
    order=np.argsort(-seeds['score'],kind='stable')
    keys=np.sort(seeds['row'][order],axis=1)
    _,ix=np.unique(keys,axis=0,return_index=True)
    seeds=seeds[order[ix]]
    actual,local=np.unique(seeds['row'].ravel(),return_inverse=True)
    seeds['row']=local.reshape(-1,3)
    p=np.empty(len(actual),POINT)
    p['x'],p['y']=sky.pos[actual].T;p['exposure']=sky.exposure[actual]
    p['night']=sky.geometry['night'][p['exposure']]
    vis=np.ascontiguousarray(np.column_stack([sky.geometry[k] for k in ('t','ox','oy','oz')]),'f8')
    lib=build_library('intersect_arcs')
    fn=lib.intersect_triples
    ptr=ctypes.c_void_p;i=ctypes.c_int
    fn.argtypes=[ptr,i,ptr,ptr,i,ctypes.c_double,i,ptr,ptr];fn.restype=i
    out=np.empty(cap,TRIAL);stats=np.zeros(3,'i8')
    n=fn(seeds.ctypes.data,len(seeds),p.ctypes.data,vis.ctypes.data,previous,seconds,cap,out.ctypes.data,stats.ctypes.data)
    out=out[:n];out['row']=actual[out['row']]
    if n:
        _,ix=np.unique(np.sort(out['row'],axis=1),axis=0,return_index=True)
        out=out[np.sort(ix)]
    return out,stats.tolist()

def offer(G,control,rows,gamma,score,window=False):
    sky=control.get('original_sky')
    if sky is None or not len(rows) or control['event'].is_set():return
    left=G['hard_end']-time.time()
    if left<28. or not hasattr(sky.lib,'night_candidate_fit'):return
    cap=52000 if window else 4800
    if len(score)>cap:
        chosen=np.argpartition(-score,cap-1)[:cap]
        rows,gamma,score=rows[chosen],gamma[chosen],score[chosen]
    seeds=np.empty(len(rows),SEED)
    seeds['row'],seeds['gamma']=rows,gamma
    q=np.quantile(score,[.25,.5,.75])
    seeds['score']=np.clip((score-q[1])/max(.65,q[2]-q[0]),-5.,5.)
    with LOCK:
        holder=control.setdefault('intersections',{'batches':[],'serial':0,'done':-1,'at':0.,'job':None})
        holder['serial']+=1
        seeds['serial']=holder['serial']
        holder['batches'].append((holder['serial'],window,seeds))
        # Retain independent window starts and a rolling measured-arc bank.
        for kind,limit in ((False,380000),(True,95000)):
            count=0;keep=[]
            for serial,w,batch in reversed(holder['batches']):
                if w==kind:
                    count+=len(batch)
                    if count>limit:continue
                keep.append((serial,w,batch))
            holder['batches']=keep[::-1]
        old=holder['done']
        new=sum(len(x) for s,w,x in holder['batches'] if s>old)
        if (new<(4000 if window else 42000) or (holder['job'] is not None and not holder['job'].done())
                or time.monotonic()-holder['at']<(12. if window else 36.)):return
        parts=[a for s,w,a in holder['batches']]
        holder['at']=time.monotonic();holder['done']=holder['serial']
        def work():
            from shutter_continuation import confirm
            trials,stats=intersections(parts,sky,old,min(11.,.16*left))
            results=0
            if len(trials) and not control['event'].is_set():
                seconds=min(5.5,G['hard_end']-time.time()-12.)
                if seconds>.25:
                    fits=confirm(sky,trials,seconds,1750)
                    results=len(fits)
                    if results:control['raw_results'].put((fits,[]))
            print('intersecting measured arcs:',stats,'prefits',len(trials),'confirmed',results,flush=True)
        pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='intersect-arcs')
        job=pool.submit(work);holder['job']=job
        pool.shutdown(wait=False)
        def finished(f):
            if f.exception() is not None:print('arc intersection:',repr(f.exception())[:160],flush=True)
        job.add_done_callback(finished)
