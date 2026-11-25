"""Compare same-night matches with local first-exposure source controls."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from scipy.special import expit,logit

INPUT=np.dtype([('r','f8'),('n','i4'),('rows','i4',32),('pad','i4')])
OUTPUT=np.dtype([(k,'f4') for k in ('odds','positive','missing')]+
                [(k,'i4') for k in ('tested','found','controls')])


def measure(sky,depth,reports,seconds,threads=2):
    entries=[];records=np.zeros(len(reports),INPUT)
    for c in reports:
        rows=c['det']
        radius=c.get('r',42.)
        if not 5<=len(rows)<=32 or not np.isfinite(radius) or radius<12.:continue
        rec=records[len(entries)]
        rec['rows'][:len(rows)]=rows
        rec['n'],rec['r']=len(rows),radius
        entries.append(c)
    if not entries:return [],np.empty(0,OUTPUT)
    finite=depth[np.isfinite(depth)]
    depth=np.ascontiguousarray(np.where(np.isfinite(depth),depth,np.median(finite) if len(finite) else 23.),'f8')
    result=np.zeros(len(entries),OUTPUT)
    p,i=ctypes.c_void_p,ctypes.c_int
    fn=sky.lib.repeat_control_odds
    fn.argtypes=[p,p,p,i,i,ctypes.c_double,p]
    fn.restype=i
    count=fn(sky.handle,depth.ctypes.data,records.ctypes.data,len(entries),threads,max(.05,seconds),result.ctypes.data)
    if count<0:raise RuntimeError('repeat-source controls failed')
    return entries[:count],result[:count]


def adjust(reports,result):
    for c,r in zip(reports,result):
        if not r['tested']:continue
        counted=max(min(4.,1.45*c.get('h3m',0)),.62*c.get('visit_repeat',0.))
        extra=max(0.,float(r['positive'])-counted)
        missing=min(0.,float(r['missing'])+.50*c.get('image_misses',0))
        # Some of the same visits informed source selection. Allow strong
        # differential repeats to raise a linkage, with bounded influence.
        delta=.82*np.clip(extra+missing,-6.,7.)
        c['paired_visits']=int(r['tested'])
        c['paired_detections']=int(r['found'])
        c['paired_controls']=int(r['controls'])
        c['paired_log_odds']=round(float(delta),4)
        c['p']=float(expit(logit(np.clip(c['p'],1.e-15,1.-1.e-15))+delta))


def start(F,reports,deadline):
    ctrl=F.G.get('survey_control') or {}
    sky=ctrl.get('original_sky')
    if sky is None or not hasattr(sky.lib,'repeat_control_odds'):return None
    seconds=min(8.,deadline-time.monotonic()-1.)
    if seconds<.35:return None
    selected=sorted(reports,key=lambda c:-c['p'])[:850]
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='paired-exposures')
    job=pool.submit(measure,sky,F.G['m5c_np'],selected,seconds)
    pool.shutdown(wait=False)
    return job
