"""Find a second apparition from a measured two-night arc."""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from original_recovery import integer, ptr, real
from transient_arcs import PROPOSAL, diversity, report

TRIANGLE=np.dtype([(k,'i4') for k in ('a','b','c')]+
                  [(k,'f4') for k in ('g','merit','short_merit')])


def measure(sky,depth,direction,small,full,threads,seconds,cap=300000):
    out=np.empty(cap,TRIANGLE);stats=np.zeros(4,'i8')
    fn=sky.lib.measure_seasonal_triangles
    fn.argtypes=[ptr,ptr,ptr,ptr,integer,ptr,integer,integer,real,integer,ptr,ptr]
    fn.restype=integer
    n=fn(sky.handle,depth.ctypes.data,direction.ctypes.data,small.ctypes.data,len(small),
         full.ctypes.data,len(full),threads,seconds,cap,out.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('seasonal triangle measurement failed')
    if not n:return out[:0],stats
    data=out[:n]
    row=np.sort(np.stack([data[k] for k in ('a','b','c')],axis=1),axis=1)
    _,first=np.unique(row,axis=0,return_index=True)
    data=data[np.sort(first)]
    # A heavily repeated field can provide many near-equivalent returns.
    # Preserve trials from each brightness range and exposure pair.
    ea,ec=sky.exposure[data['a']],sky.exposure[data['c']]
    group=(ea*len(sky.geometry)+ec)*20+np.clip((sky.mag[data['a']]-13.)*1.4,0,19).astype('i4')
    _,group=np.unique(group,return_inverse=True)
    rank=diversity(data['merit'],group,.85)
    return np.ascontiguousarray(data[np.argsort(-rank,kind='stable')]),stats


def extend(sky,depth,direction,small,full,proposals,threads,seconds,cap=50000):
    out=np.empty(cap,PROPOSAL);stats=np.zeros(1,'i8')
    fn=sky.lib.extend_seasonal_triangles
    fn.argtypes=[ptr,ptr,ptr,ptr,integer,ptr,integer,ptr,integer,integer,real,integer,ptr,ptr]
    fn.restype=integer
    n=fn(sky.handle,depth.ctypes.data,direction.ctypes.data,small.ctypes.data,len(small),
         full.ctypes.data,len(full),proposals.ctypes.data,len(proposals),threads,seconds,
         cap,out.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('seasonal triangle continuation failed')
    if not n:return out[:0],stats
    _,first=np.unique(np.sort(out['row'][:n],axis=1),axis=0,return_index=True)
    return np.ascontiguousarray(out[np.sort(first)]),stats


def launch(sky,depth,direction,small,full,control,seconds):
    def work():
        end=time.monotonic()+seconds
        triangles,counts=measure(sky,depth,direction,small,full,1,.47*seconds)
        if not len(triangles) or control['event'].is_set():return
        q,seen=extend(sky,depth,direction,small,full,triangles,1,max(.1,.61*(end-time.monotonic())))
        if not len(q) or control['event'].is_set():return
        hits,tested=report(sky,q,control,1,max(.1,end-time.monotonic()))
        print('independent seasonal returns:',counts.tolist(),len(triangles),'arcs',int(seen[0]),
              'continued',len(q),'quads',tested,'tested',hits,'confirmed',flush=True)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='seasonal-triangles')
    job=pool.submit(work)
    def finished(j):
        error=j.exception()
        if error:print('seasonal triangle search:',repr(error)[:190],flush=True)
    job.add_done_callback(finished)
    pool.shutdown(wait=False)
    return job
