"""Competing local trajectories measured in the original exposure lists."""
import ctypes
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit,logit


QUERY=np.dtype([('row','i4'),('phase','i4'),('range','f8')])
VALUE=np.dtype([('log_odds','f4'),('speed','f4'),('count','i4'),('nights','i4')])


def queries(reports, seen, limit):
    result={}
    for c in reports:
        if c.get('nights',5)>14:
            continue
        distance=np.clip(c.get('r',42.),15.,40000.)
        if not np.isfinite(distance):
            continue
        for r in c['det']:
            if r not in seen and r not in result:
                result[r]=distance
        if len(result)>=limit:
            break
    q=np.zeros((len(result),3),QUERY)
    if result:
        q['row']=np.fromiter(result,'i4')[:,None]
        q['range']=np.fromiter(result.values(),float)[:,None]
        q['phase']=np.arange(q.shape[1])
    return q


def measure(sky, q, depth, pole, seconds, threads=1):
    if not len(q):
        return np.empty((0,3),VALUE)
    ptr,integer=ctypes.c_void_p,ctypes.c_int
    f=sky.lib.survey_foregrounds
    f.argtypes=[ptr,ptr,integer,ptr,ptr,ctypes.c_double,integer,ptr]
    f.restype=integer
    output=np.zeros(q.shape,VALUE)
    depth=np.ascontiguousarray(depth,'f8')
    pole=np.ascontiguousarray(pole,'f8')
    done=f(sky.handle,q.ctypes.data,q.size,depth.ctypes.data,pole.ctypes.data,
           seconds,threads,output.ctypes.data)
    if done<0:
        raise RuntimeError('local foreground orbit verification failed')
    return output[:done//q.shape[1]]


def submit(G,reports):
    control=G['survey_control']
    sky=control.get('original_sky')
    if sky is None or not len(reports) or control['event'].is_set():
        return
    table=control.setdefault('foreground_scores',{})
    previous=control.get('foreground_job')
    if previous is not None:
        if not previous.done():
            return
        control.pop('foreground_job')
        if previous.exception() is None:
            table.update(previous.result())
    remaining=G['hard_end']-time.time()
    if remaining<17. or len(table)>=6500:
        return
    cand=sorted(reports,key=lambda c:-c['p'])
    selection=cand[:420]+cand[420::max(1,len(cand[420:])//240)]
    q=queries(selection,table,3000)
    if len(q)<15:
        return
    tilt=np.deg2rad(23.43928)
    pole=G['frame_np']@np.array([0.,-np.sin(tilt),np.cos(tilt)])
    def work():
        result=measure(sky,q,G['m5c_np'],pole,min(45.,remaining*.20))
        return dict(zip(q['row'][:len(result),0].tolist(),result))
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='foreground-motion')
    control['foreground_job']=pool.submit(work)
    pool.shutdown(wait=False)


def odds(rows, observed):
    # A random translation has the same accessible alternative velocities,
    # filters and magnitude. Measure the tail for each level of shutter and
    # night support. Shrink the empirical distribution in a sparse field.
    values=observed['log_odds']+1.35*np.maximum(0,observed['count']-3)+1.45*(observed['nights']>=2)
    strength=values[:,0]
    negative=np.sort(values[:,1:].ravel())
    real=np.sort(strength)
    null=(len(negative)-np.searchsorted(negative,strength,'left')+.8)/(len(negative)+2.)
    mixed=(len(real)-np.searchsorted(real,strength,'left')+1.)/(len(real)+2.)
    logratio=np.maximum(0.,np.log(mixed/null))
    # Also require an absolute chance-odds margin. A small control catalogue
    # cannot exclude clusters of real foreground sources by itself.
    clear=np.clip((strength-4.)/8.,0.,1.)
    return dict(zip(rows,np.minimum(4.,logratio)*clear))


def apply(reports,table):
    if len(table)<45:
        return 0
    rows=list(table)
    score=odds(rows,np.stack(list(table.values())))
    changed=0
    for c in reports:
        if c.get('nights',5)>14:
            continue
        z=sorted((score.get(r,0.) for r in c['det']),reverse=True)
        if not z or z[0]<.15:
            continue
        penalty=min(5.5,.88*z[0]+.30*sum(z[1:3]))
        c['foreground_motion']=round(penalty,3)
        c['p']=float(expit(logit(np.clip(c['p'],1.e-14,1.-1.e-14))-penalty))
        changed+=1
    return changed


def finish(G,outdir):
    control=G.get('survey_control') or {}
    table=control.get('foreground_scores',{})
    job=control.get('foreground_job')
    if job is not None and job.done() and not job.exception():
        table.update(job.result())
    if not table:
        return
    path=Path(outdir)/'candidates.json'
    reports=json.loads(path.read_text())
    changed=apply(reports,table)
    if changed:
        reports.sort(key=lambda c:-c['p'])
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(reports));os.replace(tmp,path)
    print('independent foreground tracks:',len(table),'positions,',changed,'adjusted reports',flush=True)
