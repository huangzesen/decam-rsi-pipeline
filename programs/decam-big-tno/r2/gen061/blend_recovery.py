"""Check four-night predictions against independently estimated foreground light."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import time

import numpy as np
from survey_arcs import CERTIFIED

COMPONENT=np.dtype([('row','i4')]+[(k,'f4') for k in ('x','y','mag','sn','logbf')],align=True)
RESULT=np.dtype([('orbit',CERTIFIED),('component',COMPONENT),('merit','f8'),('core','i4'),('pad','i4')],align=True)


def search(sky,cores,depth,seconds,threads=1):
    out=np.empty(min(2800,len(cores)),RESULT)
    counts=np.zeros(5,'i8')
    if not len(cores):return out,counts
    fn=sky.lib.blended_orbits
    ptr,integer=ctypes.c_void_p,ctypes.c_int
    fn.argtypes=[ptr,ptr,integer,ptr,integer,ctypes.c_double,integer,ptr,ptr];fn.restype=integer
    depth=np.ascontiguousarray(depth,'f8')
    n=fn(sky.handle,cores.ctypes.data,len(cores),depth.ctypes.data,threads,seconds,
         len(out),out.ctypes.data,counts.ctypes.data)
    if n<0:raise RuntimeError('blended source follow-up failed')
    return out[:n].copy(),counts


def submit(F,cores):
    G=F.G;control=G['survey_control'];sky=control.get('original_sky')
    if sky is None or not hasattr(sky.lib,'blended_orbits') or control['event'].is_set():return
    job=control.get('blend_job')
    if job is not None and not job.done():return
    left=G['hard_end']-time.time()
    if left<18.:return
    # Include the tails of the four-night merit distribution. A field star
    # can spoil the best-fitting five-night continuation of a genuine seed.
    if len(cores)>8000:
        tail=np.linspace(3200,len(cores)-1,4800,dtype=int)
        cores=cores[np.r_[np.arange(3200),tail]]
    def work():
        out,stats=search(sky,cores,G['m5c_np'],min(44.,.30*left))
        control['blend_results'].put(out)
        F.log('foreground-subtracted follow-up:',stats.tolist(),len(out),'orbits')
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='blended-sources')
    job=pool.submit(work);pool.shutdown(wait=False);control['blend_job']=job
    def finished(f):
        if f.exception():F.log('foreground follow-up:',repr(f.exception())[:150])
    job.add_done_callback(finished)


def merge(F):
    import image_followup as images
    G=F.G;control=G['survey_control']
    from queue import Empty
    batches=[]
    while True:
        try:batches.append(control['blend_results'].get_nowait())
        except Empty:break
    if not batches:return []
    records=np.concatenate(batches)
    used=control.setdefault('blend_seen',set())
    links=[];states=[]
    for v in records[np.argsort(-records['merit'],kind='stable')]:
        r=v['orbit'];b=v['component'];d=r['row'][:r['count']].astype('i8')
        key=tuple(sorted(d))
        if key in used:continue
        used.add(key)
        extra={int(b['row']):tuple(float(b[s]) for s in ('x','y','mag','sn'))}
        links.append(dict(d=d,e=control['original_sky'].exposure[d],res=r['err'][:r['count']],
            x=r['x'],ref=r['ref'],penalty=float(np.clip(.30*(v['merit']-5.),-2.7,1.2)),
            original=dict(candidate=dict(det=d[:-1],image_misses=0,image_deep_hits=0)),components=extra))
        states.append((r['x'].copy(),float(r['ref'])))
        if len(links)>=420:break
    if not links:return []
    G['ob_np_follow']=G['ob'].cpu().numpy()
    reports=images.score_promotions(F,links,images.arrays(G['scen']))
    for c,link,state in zip(reports,links,states):
        key=tuple(sorted(c['det']))
        c['foreground_fit']=1
        G.setdefault('report_components',{})[key]=link['components']
        G.setdefault('survey_orbits',{})[key]=state
    return reports
