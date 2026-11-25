"""Predict measured returns from arclets supplied by the regional window searches."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import time

import numpy as np

from attributables import SEED
from original_recovery import CORE, RECOVERY

PTR = ctypes.c_void_p
INT = ctypes.c_int
REAL = ctypes.c_double
TRIAL = np.dtype([('row','i4',5),('gamma','f4'),('score','f4')])


def forecast(sky,seeds,n,seconds):
    capacity=len(seeds)*(3 if n==3 else 1)
    out=np.empty(capacity,CORE)
    tried=INT()
    if n==3:
        fn=sky.lib.triple_predictions
        fn.argtypes=[PTR,PTR,INT,REAL,INT,PTR,PTR]
        fn.restype=INT
        count=fn(sky.handle,seeds.ctypes.data,len(seeds),seconds,
                 capacity,out.ctypes.data,ctypes.byref(tried))
    else:
        fn=sky.lib.quartet_predictions
        fn.argtypes=[PTR,PTR,INT,PTR,INT,REAL,PTR,PTR]
        fn.restype=INT
        count=fn(sky.handle,seeds.ctypes.data,len(seeds),None,len(sky.mag),seconds,
                 out.ctypes.data,ctypes.byref(tried))
    return out[:max(0,count)],tried.value


def confirm(sky,trials,seconds,cap):
    fn=sky.lib.night_candidate_fit
    fn.argtypes=[PTR,PTR,INT,REAL,INT,PTR]
    fn.restype=INT
    out=np.empty(min(len(trials),cap),RECOVERY)
    count=fn(sky.handle,trials.ctypes.data,len(trials),seconds,len(out),out.ctypes.data)
    return out[:max(0,count)].copy()


def run(G,control,native,gamma,rms,seconds):
    end=time.monotonic()+seconds
    sky=control.get('original_sky')
    if sky is None or not hasattr(sky.lib,'triple_predictions') or control['event'].is_set():
        return
    good=(native.min(axis=1)>=0)&(native.max(axis=1)<len(G['keep_np']))&np.isfinite(gamma+rms)
    native,gamma,rms=native[good],gamma[good],rms[good]
    if not len(native):return
    rows=G['keep_np'][native]
    t=sky.geometry['t'][sky.exposure[rows]]
    order=np.argsort(t,axis=1)
    mag=G['mag_np'][native];sn=G['snr_np'][native]
    merit=(.43*np.log(np.minimum(sn,50.)/4.5).mean(1)-.57*np.minimum(9.,(np.std(mag,axis=1)/.32)**2)
           -.65*(rms/.24)**2+.45*np.log1p(np.ptp(t,axis=1)/25.))
    take=np.argsort(-np.nan_to_num(merit,nan=-90.),kind='stable')[:55000]
    seed=np.zeros(len(take),SEED);seed['rows'][:,3]=-1
    seed['rows'][:,:3]=np.take_along_axis(rows[take],order[take],axis=1)
    seed['gamma']=gamma[take];seed['score']=merit[take]
    at=total=0;seen=set()
    while at<len(seed) and time.monotonic()+.3<end and not control['event'].is_set():
        core,done=forecast(sky,seed[at:at+384],3,min(2.2,max(.08,end-time.monotonic())))
        if not done:break
        at+=done
        if not len(core):continue
        selected=[]
        for j in np.argsort(-core['merit'],kind='stable'):
            key=tuple(sorted(core['rows'][j,:5]))
            if key not in seen:
                seen.add(key);selected.append(j)
        if not selected:continue
        core=core[selected];trials=np.empty(len(core),TRIAL)
        trials['row']=core['rows'][:,:5];trials['gamma']=core['x'][:,2];trials['score']=core['merit']
        fits=confirm(sky,trials,min(2.0,max(.06,end-time.monotonic())),1200)
        if len(fits):control['raw_results'].put((fits,[]));total+=len(fits)
    print('three measured nights in original images:',at,'starts,',total,'verified fits',flush=True)


def begin(G,control,native,gamma,rms,seconds):
    if seconds<2. or control.get('original_sky') is None:return
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='shutter-arclets')
    job=pool.submit(run,G,control,native,gamma,rms,seconds);pool.shutdown(wait=False)
    def done(f):
        if f.exception() is not None:print('three-night raw return:',repr(f.exception())[:160],flush=True)
    job.add_done_callback(done)
    return job
