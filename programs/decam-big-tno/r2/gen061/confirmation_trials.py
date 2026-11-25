"""Independent-night confirmation against the input survey's displaced tracks."""
import ctypes
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import expit,logit

REPORT=np.dtype([('gamma','f8'),('n','i4'),('rows','i4',12),('pad','i4')],align=True)
RESULT=np.dtype([(k,'f4',12) for k in ('support','false_alarm','controls_hit','error')])


def measure(sky,depth,candidates,seconds):
    records=np.zeros(len(candidates),REPORT)
    order=[]
    nights=sky.geometry['night']
    for c in sorted(candidates,key=lambda x:-x['p']):
        rows=np.asarray(c['det'],'i4')
        if not 5<=len(rows)<=64:continue
        exp=sky.exposure[rows]
        ranked=np.argsort(-sky.arrays[0][rows],kind='stable')
        _,ix=np.unique(nights[exp[ranked]],return_index=True)
        use=rows[ranked[ix]]
        if not 5<=len(use)<=12:continue
        record=records[len(order)]
        record['n']=len(use);record['rows'][:len(use)]=use
        radius=float(c.get('r',43.))
        if not np.isfinite(radius) or not 13.<radius<9000.:continue
        record['gamma']=np.sqrt(1.+np.sum(sky.pos[use].mean(0)**2))/radius
        order.append(c)
    records=records[:len(order)]
    out=np.full((len(order),48),np.nan,'f4').view(RESULT).reshape(-1)
    if not len(order):return [],out,records
    depth=np.asarray(depth)
    finite=depth[np.isfinite(depth)]
    depth=np.ascontiguousarray(np.where(np.isfinite(depth),depth,np.median(finite) if len(finite) else 23.),'f8')
    fn=sky.lib.confirmation_trials_run
    ptr,integer=ctypes.c_void_p,ctypes.c_int
    fn.argtypes=[ptr,ptr,integer,ptr,integer,ctypes.c_double,ptr];fn.restype=integer
    n=fn(sky.handle,records.ctypes.data,len(order),depth.ctypes.data,2,max(.05,seconds),out.ctypes.data)
    if n<0:raise RuntimeError('held-out survey controls failed')
    return order[:n],out[:n],records[:n]


def adjust(order,out,records):
    for c,measurement,record in zip(order,out,records):
        n=int(record['n'])
        evidence=measurement['support'][:n]
        good=np.isfinite(evidence)
        if good.sum()<max(3,n-2):continue
        # A body must have five independent measurements. A single weak
        # assignment is less pivotal when many other nights agree.
        evidence=np.sort(evidence[good])
        lower=evidence[min(len(evidence)-1,max(0,n-5))]
        typical=np.median(evidence)
        value=.55*lower+.45*typical
        c['confirmation_log_ratio']=round(float(value),4)
        c['shifted_false_alarm']=round(float(np.median(measurement['false_alarm'][:n][good])),6)
        c['shifted_controls']=round(float(np.mean(measurement['controls_hit'][:n][good])),4)
        # This test shares astrometry and photometry with the orbit evidence.
        # Only pool a bounded contribution from the blind confirmation trials.
        shift=np.clip(.65*(value-2.),-3.1,3.1)
        c['p']=float(expit(logit(np.clip(c['p'],1.e-14,1.-1.e-14))+shift))


def start(F,reports,seconds):
    sky=(F.G.get('survey_control') or {}).get('original_sky')
    if sky is None or not hasattr(sky.lib,'confirmation_trials_run'):return
    selected=sorted(reports,key=lambda c:-c['p'])[:740]
    def work():
        order,out,records=measure(sky,F.G['m5c_np'],selected,seconds)
        adjust(order,out,records)
        return {tuple(sorted(c['det'])):{name:c[name] for name in
                ('confirmation_log_ratio','shifted_false_alarm','shifted_controls')}
                for c in order if 'confirmation_log_ratio' in c}
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='held-out-trials')
    job=pool.submit(work);pool.shutdown(wait=False)
    return job


def apply(F,reports,deadline):
    sky=(F.G.get('survey_control') or {}).get('original_sky')
    if sky is None or not hasattr(sky.lib,'confirmation_trials_run'):return
    job=F.G.get('heldout_job')
    cached=job.result() if job is not None and job.done() else {}
    selected=sorted(reports,key=lambda c:-c['p'])[:620]
    pending=[];tested=0
    for c in selected:
        evidence=cached.get(tuple(sorted(c['det'])))
        if evidence is None:
            pending.append(c);continue
        c.update(evidence)
        shift=np.clip(.65*(c['confirmation_log_ratio']-2.),-3.1,3.1)
        c['p']=float(expit(logit(np.clip(c['p'],1.e-14,1.-1.e-14))+shift))
        tested+=1
    if pending and time.monotonic()+.3<deadline:
        order,out,records=measure(sky,F.G['m5c_np'],pending,deadline-time.monotonic())
        adjust(order,out,records)
        tested+=sum('confirmation_log_ratio' in c for c in order)
    F.log('held-out survey controls:',tested,'reports with independent-night evidence')
