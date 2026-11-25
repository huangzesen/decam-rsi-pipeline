"""Recover two independent catalogue measurements from a measured triangle."""
from concurrent.futures import ThreadPoolExecutor
import time
from threading import Lock

import numpy as np
from original_recovery import CORE


def cores(sky,rows,gamma,merit):
    result=np.zeros(len(rows),CORE)
    result['count']=3;result['rows'][:,:3]=rows
    result['ref']=sky.geometry['t'][sky.exposure[rows]].mean(1)
    result['x'][:,:2]=sky.pos[rows].mean(1);result['x'][:,2]=gamma
    result['merit']=np.clip(merit,-2.,5.)
    return result


def submit(control,rows,gamma,seconds,priority=None):
    gate=control.setdefault('three_gate',Lock())
    if not gate.acquire(blocking=False):return
    try:
        return prepare(control,rows,gamma,seconds,priority)
    finally:
        gate.release()


def prepare(control,rows,gamma,seconds,priority):
    sky=control.get('original_sky')
    if sky is None or not len(rows) or control['event'].is_set():return
    previous=control.get('three_job')
    if previous is not None and not previous.done():return
    n=min(len(rows),80000);rows=rows[:n];gamma=gamma[:n]
    # Rank physical triangles by measurement quality, photometric consistency
    # and the curvature of the observer's projected path across their dates.
    ex=sky.exposure[rows];geo=sky.geometry
    ts=geo['t'][ex];order=np.argsort(ts,axis=1);ts=np.take_along_axis(ts,order,axis=1)
    e=np.take_along_axis(ex,order,axis=1)
    span=ts[:,-1]-ts[:,0]
    alpha=(ts[:,1]-ts[:,0])/np.maximum(.01,span)
    bend=np.zeros(n)
    for k in ('ox','oy'):
        v=geo[k][e]
        bend+=(v[:,1]-v[:,0]-alpha*(v[:,-1]-v[:,0]))**2
    m=sky.mag[rows];sn=sky.arrays[0][rows]
    merit=(.62*np.mean(np.log(np.clip(sn,4.5,40.)/5.),1)-2.7*np.std(m,1)
           +.32*np.log1p(span/22.)+.28*np.log(.03+np.sqrt(bend)))
    if priority is not None:merit+=.25*np.clip(priority[:n],-6.,6.)
    ok=np.isfinite(merit)&np.isfinite(gamma)&(gamma>1./9000.)&(gamma<1./13.)
    ii=np.flatnonzero(ok);ii=ii[np.argsort(-merit[ii],kind='stable')]
    seen=control.setdefault('three_tried',set())
    if len(seen)>180000:seen.clear()
    choice=[];endpoints={}
    for i in ii:
        key=tuple(sorted(rows[i]))
        if key in seen or min(endpoints.get(r,0) for r in key)>7:continue
        seen.add(key);choice.append(i)
        for r in key:endpoints[r]=endpoints.get(r,0)+1
        if len(choice)>=16000:break
    if not choice:return
    q=cores(sky,rows[choice],gamma[choice],merit[choice])
    def follow():
        start=time.monotonic()
        out,tested=sky.recover(q,seconds,cap=1000)
        control['raw_results'].put((out,[]))
        print('original three-night search:',tested,'of',len(q),'triangles,',len(out),
              'orbits in',round(time.monotonic()-start,1),'s',flush=True)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='original-triangles')
    control['three_job']=pool.submit(follow);pool.shutdown(wait=False)
    def finished(f):
        if f.exception():print('original triangles:',repr(f.exception())[:180],flush=True)
    control['three_job'].add_done_callback(finished)
