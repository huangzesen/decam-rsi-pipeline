"""Use independently measured short arcs to test foreground contamination."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import time

import numpy as np
from scipy.special import expit,logit


RESULT=np.dtype([('odds','f4',3),('span','f4'),('other','i4',3),('images','i4')])


def measure(sky,rows,radius,pole,depth,seconds):
    rows=np.ascontiguousarray(rows,'i4')
    r=np.maximum(9.,np.asarray(radius))
    speed=np.ascontiguousarray(.017203*206264.806/(24.*(r-1.))*(1.+np.sqrt(2./r)),'f4')
    pole=np.ascontiguousarray(pole,'f8')
    depth=np.ascontiguousarray(depth,'f8')
    out=np.zeros(len(rows),RESULT)
    fn=sky.lib.foreground_tracks
    p,i,d=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    fn.argtypes=[p,p,p,i,p,p,d,p];fn.restype=i
    n=fn(sky.handle,rows.ctypes.data,speed.ctypes.data,len(rows),
         pole.ctypes.data,depth.ctypes.data,max(.01,seconds),out.ctypes.data)
    if n<0:raise RuntimeError('foreground arc search failed')
    return rows[:n],out[:n]


def schedule(G,reports):
    control=G.get('survey_control') or {}
    sky=control.get('original_sky')
    if sky is None or not hasattr(sky.lib,'foreground_tracks') or not reports:
        return
    job=control.get('foreground_job')
    if job is not None and not job.done():return
    left=G['hard_end']-time.time()
    if left<22.:return
    table=control.setdefault('foreground_rows',{})
    cand=sorted(reports,key=lambda c:-c['p'])
    pending={}
    for c in cand[:600]+cand[600::7]:
        r=float(c.get('r',43.))
        if not (12.<r<1800. and 5<=c.get('nights',5)<=13):continue
        for row in c['det']:
            if row not in table:pending.setdefault(row,r)
    if not pending:return
    rows=np.fromiter(pending,'i4');radii=np.fromiter(pending.values(),'f8')
    eps=np.deg2rad(23.43928)
    pole=G['frame_np']@np.array([0.,-np.sin(eps),np.cos(eps)])
    def work():
        a,out=measure(sky,rows,radii,pole,G['m5c_np'],min(26.,.29*left))
        table.update(zip(a.tolist(),out))
        print('foreground arc measurements:',len(a),'of',len(rows),'new rows;',len(table),
              'in the field',flush=True)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='foreground-arcs')
    job=pool.submit(work)
    control['foreground_job']=job
    pool.shutdown(wait=False)
    def finished(f):
        if f.exception() is not None:
            print('foreground motion search:',str(f.exception())[:170],flush=True)
    job.add_done_callback(finished)


def contamination(values):
    """Estimate the false-track share at each strength using matched displaced queries."""
    observed=values['odds'].astype(float)
    good=(values['images']>=2)&(values['other'][:,1]>=0)
    true=np.sort(observed[values['images']>=2,0])
    control=np.sort(observed[values['images']>=2,1:].ravel())
    if not len(true):return np.zeros(len(values))
    score=observed[:,0]
    ntrue=len(true)-np.searchsorted(true,score)
    nfalse=.5*(len(control)-np.searchsorted(control,score))
    # Both foreground and control searches were conditioned on an actual
    # first pair. A two-visit chance association thus pays the same trials.
    purity=np.clip(1.-(nfalse+.35)/(ntrue+.7),0.,.995)
    separation=score-np.maximum(0.,np.max(observed[:,1:],axis=1))
    p=expit(score-4.)*expit((separation-2.2)*1.15)
    return np.where(good,np.clip(p*(.36+.64*purity),0.,.995),0.)


def apply(F,reports,deadline):
    ctrl=F.G.get('survey_control') or {}
    cache=ctrl.get('foreground_rows')
    if not cache:return
    snapshot=cache.copy()
    probability=contamination(np.fromiter(snapshot.values(),RESULT,count=len(snapshot)))
    prob=dict(zip(snapshot,probability))
    ex=ctrl['original_sky'].exposure
    night=F.G['night_e_np']
    changed=removed=0;recheck=[]
    for c in reports:
        rows=np.asarray(c['det'],'i8')
        val=np.asarray([prob.get(i,0.) for i in rows])
        if np.max(val)<.18:continue
        n=night[ex[rows]]
        risk=np.array([np.min(val[n==k]) for k in np.unique(n)])
        # A singly contaminated measurement need not invalidate a long
        # orbit. Five clean nights can support it without that measurement.
        penalty=np.log1p(-np.minimum(.995,risk))
        secure=np.sum(risk<.18)
        scale=1. if secure<5 else .36
        adjustment=scale*max(-10.,penalty.sum())
        c['foreground_probability']=round(float(np.max(val)),4)
        c['foreground_log_odds']=round(float(adjustment),4)
        c['p']=float(expit(logit(np.clip(c['p'],1.e-14,1.-1.e-14))+adjustment))
        changed+=1
        keep=val<.93
        if not keep.all() and len(np.unique(n[keep]))>=5 and len(rows)<=30:
            recheck.append(dict(c,det=rows[keep].tolist(),_parent=c))
    if recheck and time.monotonic()+.6<deadline:
        import image_followup as I
        G=F.G;cat=I.arrays(G['scen'])
        G['ob_np_follow']=G['ob'].cpu().numpy()
        fit=I.prepare(F,recheck,cat)
        for lo in range(0,len(fit),64):
            if time.monotonic()+.4>=deadline:break
            batch=fit[lo:lo+64];I.model(F,batch,refit=True)
            for v in batch:
                c=v['candidate']['_parent'];x=v['x'];g=x[2]
                if not .00010<g<.09:continue
                energy=x[3:]@x[3:]*np.sqrt(1.+x[:2]@x[:2])/(F.MU*g**3)
                rms=float(np.sqrt(np.mean(v['res']**2)))
                if energy>=2. or rms>max(.19,min(.36,c.get('rms',.23)+.03)):continue
                c.update(det=v['d'].tolist(),nights=len(set(night[v['e']])),rms=round(rms,3))
                removed+=1
    F.log('independent fast tracks:',changed,'confidence adjustments,',removed,'membership refits')
