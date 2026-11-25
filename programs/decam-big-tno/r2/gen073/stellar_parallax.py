"""Measure very small parallaxes in sources observed repeatedly in place."""
import ctypes
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
from scipy.special import expit

from annual_parallax import REPORT
from companion_graph import build_library

VIEW=np.dtype([(k,'f8') for k in ('t','ox','oy','oz','depth')]+[('night','i4'),('band','i4')],align=True)


def measure(xy,mag,sn,sharp,flag,ex,vis,seconds,cap=6500):
    end=time.monotonic()+seconds
    lib=build_library('stellar_parallax');run=lib.stellar_tracks
    ptr,integer,real=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    run.argtypes=[ptr]*6+[integer,ptr,integer,real,ptr,ptr,integer,ptr,ptr]
    run.restype=integer
    pos=np.empty(cap,REPORT);neg=np.empty(cap,REPORT);stats=np.zeros(6,'i8');n=np.zeros(2,'i4')
    remaining=end-time.monotonic()
    if remaining<=0.:return pos[:0],neg[:0],stats
    done=run(*(v.ctypes.data for v in (xy,mag,sn,sharp,flag,ex)),len(mag),vis.ctypes.data,len(vis),
             remaining,pos.ctypes.data,neg.ctypes.data,cap,n.ctypes.data,stats.ctypes.data)
    if done<0:raise RuntimeError('small annual displacement search failed')
    return pos[:n[0]].copy(),neg[:n[1]].copy(),stats


def independent(records):
    used=set();out=[]
    for v in records[np.argsort(-records['merit'],kind='stable')]:
        rows=v['rows'][:v['count']]
        if sum(int(row) in used for row in rows)>=2:continue
        used.update(map(int,rows));out.append(v)
    return np.asarray(out,dtype=REPORT)


def reports(pos,neg):
    pos,neg=independent(pos),independent(neg)
    if not len(pos):return []
    background=np.sort(neg['merit'])
    base=np.quantile(background,.72) if len(background)>=7 else 12.
    tail=background[background>=base]-base
    # Shrink a small observed control tail toward a conservative slope.
    scale=(15.+tail.sum())/(4.+len(tail))
    prev=1.-1.e-12;out=[]
    for j,r in enumerate(pos):
        merit=r['merit']
        false=max(float(len(background)-np.searchsorted(background,merit)),
                  (1.5+len(tail))*np.exp(-max(0.,merit-base)/np.clip(1.25*scale,2.2,12.)))
        confidence=min(prev,expit(.64*(merit-19.)),max(np.exp(-false),1.-false/(j+1.)))
        if confidence<.05:break
        prev=confidence;n=int(r['count'])
        out.append(dict(det=r['rows'][:n].tolist(),p=float(confidence),nights=n,
                        rms=round(float(r['rms']),3),r=round(float(1./r['g']),2),
                        arc=round(float(r['span']),1),parallax_sigma=round(float(r['significance']),2),
                        stellar_parallax=True))
    return out


def measurements(projected,path,t,observer,night,depth,bands,seconds):
    view=np.empty(len(depth),VIEW)
    view['t'],view['night']=t,night
    view['ox'],view['oy'],view['oz']=observer.T
    view['depth']=depth;view['band']=np.unique(bands,return_inverse=True)[1]
    raw=lambda k:np.load(Path(path)/('det_'+k+'.npy'),mmap_mode='r')
    return measure(*projected,raw('snr'),raw('sharp'),raw('flag'),raw('exp'),view,seconds)


def begin(projected,path,t,observer,night,depth,bands,control,seconds):
    def work():
        started=time.monotonic()
        p,n,stat=measurements(projected,path,t,observer,night,depth,bands,seconds)
        result=reports(p,n)
        control['stellar_parallax_reports']=result
        print('differential stellar astrometry:',stat.tolist(),len(p),'positive,',len(n),
              'reversed and',len(result),'retained in',round(time.monotonic()-started,1),'s',flush=True)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='stellar-parallax')
    job=pool.submit(work);pool.shutdown(wait=False)
    def report(future):
        exc=future.exception()
        if exc is not None:print('stellar parallax:',type(exc).__name__,str(exc)[:140],flush=True)
    job.add_done_callback(report)
    return job


def finish(G,outdir,deadline):
    from reference_field import project
    from survey_reports import unique
    ctrl=G.setdefault('survey_control',{})
    job=ctrl.get('stellar_parallax_job')
    remaining=lambda:deadline-time.monotonic()
    if job is None and remaining()>4.:
        path=Path(G['scen'])
        def raw(k):return np.load(path/('det_'+k+'.npy'),mmap_mode='r')
        ra,dec,flux,ex=[raw(k) for k in ('ra','dec','flux','exp')]
        if remaining()<4.+len(ra)/1500000.:return
        xy=np.empty((len(ra),2),'f4');mag=np.empty(len(ra),'f4');zero=G['mz_np']-G['cb_np']
        for lo in range(0,len(ra),600000):
            if remaining()<1.:return
            s=slice(lo,lo+600000)
            xy[s]=project(ra[s],dec[s],G['frame_np'])
            mag[s]=zero[ex[s]]-2.5*np.log10(np.maximum(1.e-9,flux[s]))
        job=begin((xy,mag),path,G['et_np'],G['ob'].cpu().numpy(),G['night_e_np'],
                  G['m5c_np'],G['band_np'],ctrl,max(.2,.92*remaining()))
        ctrl['stellar_parallax_job']=job
    if job is not None and not job.done():
        try:job.result(timeout=max(0.,remaining()))
        except TimeoutError:return
    fresh=ctrl.get('stellar_parallax_reports')
    if not fresh:return
    path=Path(outdir)/'candidates.json'
    prior=json.loads(path.read_text())
    merged=unique(prior+fresh,2000)
    tmp=path.with_suffix('.parallax');tmp.write_text(json.dumps(merged));os.replace(tmp,path)
