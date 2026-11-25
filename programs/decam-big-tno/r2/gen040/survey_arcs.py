"""Continuous distance and radial-velocity intersection from one measured pair."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

import range_search as R
from reference_field import Reference, local_depth, project

CERTIFIED=np.dtype([('x','f8',6),('ref','f8'),('merit','f8'),('count','i4'),('nights','i4'),
                    ('row','i4',64),('err','f4',64)],align=True)


def certify(lib,p,g,w,target,ex,threads,seconds,limit=12000):
    if not len(p):return np.empty(0,CERTIFIED)
    if p.shape[1]==3:p=np.column_stack((p,np.full(len(p),-1)))
    joined=np.ascontiguousarray(np.column_stack((p,w)),dtype='i4')
    g=np.ascontiguousarray(g,'f8')
    done=np.empty(limit,CERTIFIED)
    k=lib.certify_orbits;ptr=ctypes.c_void_p;integer=ctypes.c_int;real=ctypes.c_double
    k.argtypes=[ptr,integer,ptr,integer,ptr,ptr,integer,integer,real,integer,ptr];k.restype=integer
    count=k(target.ctypes.data,len(target),ex.ctypes.data,len(ex),joined.ctypes.data,g.ctypes.data,
            len(p),threads,seconds,len(done),done.ctypes.data)
    if count<0:raise RuntimeError('native Kepler verification failed')
    return done[:count].copy()


def register_measurements(F,control,arrays):
    """Give confirmed original-image measurements stable reporting indices."""
    G=F.G;extra=control.get('additional')
    if extra is None or not arrays:return
    offset=extra['offset'];book=extra['book']
    used=[]
    for arr in arrays:
        if not len(arr):continue
        d=arr['row']
        valid=(np.arange(d.shape[1])<arr['count'][:,None])&(d>=offset)
        if valid.any():used.append(d[valid]-offset)
    if not used:return
    selected=np.unique(np.concatenate(used))
    add=selected[book[selected]<0]
    if len(add):
        import torch
        A=G.get('aux')
        if A is None:
            A=dict(th_np=np.empty((0,2)),de_np=np.empty(0,'i8'),
                   et=G['et'].cpu(),ob=G['ob'].cpu());G['aux']=A
        G.setdefault('raw_start',len(G['keep_np']))
        G['raw_background']=extra['hist']
        book[add]=np.arange(len(add))+len(G['keep_np'])
        for k,src in (('keep_np','rows'),('snr_np','sn'),('mag_np','mag'),('lev_np','lev')):
            G[k]=np.concatenate((G[k],extra[src][add]))
        A['th_np']=np.concatenate((A['th_np'],extra['xy'][add]))
        A['de_np']=np.concatenate((A['de_np'],extra['ex'][add]))
        A['n']=len(A['de_np'])
        A['index']=F.Index(A['th_np'],A['de_np'],8.*F.AS,'cpu')
        A['has_e']=torch.as_tensor(np.bincount(A['de_np'],minlength=len(G['et_np']))>0)
        G.pop('completion_catalogue',None);G.pop('evidence_density',None)
        F.log('original-image recovery:',len(add),'confirmed measurements added to reporting catalogue')
    for arr in arrays:
        d=arr['row'];valid=(np.arange(d.shape[1])<arr['count'][:,None])&(d>=offset)
        d[valid]=book[d[valid]-offset]


def native_links(G,arr):
    nk=G['nk'];ae=G['aux']['de_np'] if G.get('aux') else np.empty(0,int)
    ne=G['night_e_np'];links=[]
    for v in arr:
        count=int(v['count'])
        d=v['row'][:count].astype('int64')
        main=d<nk;e=np.empty(count,dtype=np.int64)
        e[main]=G['de_np'][d[main]]
        if not main.all():e[~main]=ae[d[~main]-nk]
        order=np.argsort(e)
        links.append(dict(det=d[order],exps=e[order],res=v['err'][:count][order].astype('f8'),
                          x=v['x'].copy(),tref=float(v['ref']),nn=int(v['nights']),
                          nax=int((~main).sum()),nn0=len(np.unique(ne[e[main]])),
                          n0=4,sn=4,sp=1,src=0))
    return links


def source_priority(G,src,ids,field):
    ex=G['de_np'][ids]
    xy=G['th_np'][ids]
    m=src['mag']
    shadow=field.features(xy,ex,m,G['et_np'],G['lev_e'])
    visible=local_depth(G['cov'],G['m5c_np'],G['night_e_np'],xy,ex,m)
    return (src['quality']+.65*np.minimum(3.,visible)
            -1.20*shadow[:,0]-.50*shadow[:,1]-.80*shadow[:,2])


def run_kernel(lib,a,ids,quality,target,ex,epro,nthreads,seconds,span,cap,ns,nt,mode):
    ns=min(ns,len(a));nt=min(nt,len(a))
    sel=np.argpartition(-quality,nt-1)[:nt];sel.sort()
    a=a[sel];ids=ids[sel];a['quality']=quality[sel]
    si=np.argpartition(-a['quality'],ns-1)[:ns].astype('i4');si.sort()
    ptr=ctypes.c_void_p;integer=ctypes.c_int;real=ctypes.c_double;k=lib.arc_join
    k.argtypes=[ptr,integer,integer,ptr,integer,integer,real,real,ptr,integer,ptr,integer,
                real,integer,integer,integer,ptr,ptr,integer];k.restype=integer
    result=np.empty(cap,R.RESULT);stats=np.zeros(7,np.int64)
    n=k(a.ctypes.data,len(a),int(ex['night'].max())+1,si.ctypes.data,len(si),1900000,*map(float,epro),
        target.ctypes.data,len(target),ex.ctypes.data,len(ex),seconds,nthreads,span,cap,
        result.ctypes.data,stats.ctypes.data,mode)
    if n<0:raise RuntimeError('orbital strip intersection failed')
    out=result[:n]
    return ids[out['ids']],out['gamma'].copy(),out['witness'].copy(),stats


def work(G,ob,epro,available,control):
    start=time.monotonic()
    raw=Reference(G['scen'],G['frame_np'],G['mz_np'],G['cb_np'])
    try:
        src,ids=R.sources(G,ob,len(G['seed2_np']))
        if not len(ids):return
        q=source_priority(G,src,ids,raw)
        scen=Path(G['scen'])
        with (scen/'exposures.json').open() as fp:E=json.load(fp)
        limit=min(1000000,max(25000,int(.14*len(G['keep_np']))))
        rows,_=raw.candidates(scen,[G['keep_np']],E,limit)
        L=lambda k:np.load(scen/('det_'+k+'.npy'),mmap_mode='r')[rows]
        er=L('exp').astype('i8');xy=project(L('ra'),L('dec'),G['frame_np'])
        mag=(G['mz_np']-G['cb_np'])[er]-2.5*np.log10(np.maximum(L('flux'),1.e-9))
        mb=np.clip(np.nan_to_num((mag-10.)*4,nan=79).astype('i4'),0,79)
        hist=np.bincount(er*80+mb,minlength=len(ob)*80).reshape(len(ob),80)
        extra=dict(rows=rows,ex=er,xy=xy,mag=mag,lev=G['lev_e'][er],sn=L('snr'),
                   offset=len(G['keep_np']),book=np.full(len(rows),-1,'i8'),hist=hist)
        control['additional']=extra
    finally:
        raw.close()
    if control['event'].is_set():return
    a=G.get('aux')
    n=G['nk'];nt=n+(a['n'] if a else 0)
    target=np.empty(nt+len(rows),R.TARGET)
    target['x'][:n],target['y'][:n]=G['th_np'].T
    target['exposure'][:n]=G['de_np']
    if a:
        target['x'][n:nt],target['y'][n:nt]=a['th_np'].T
        target['exposure'][n:nt]=a['de_np']
    target['mag'][:nt]=G['mag_np'][:nt]
    target['x'][nt:],target['y'][nt:]=xy.T
    target['exposure'][nt:]=er;target['mag'][nt:]=mag
    ex=np.zeros(len(ob),R.EXPOSURE);ex['t'],ex['night']=G['et_np'],G['night_e_np']
    ex['ox'],ex['oy'],ex['oz']=ob.T
    depths=G['m5c_np'];depths=depths[np.isfinite(depths)]
    center=np.median(depths) if len(depths) else np.median(src['mag'])
    with tempfile.TemporaryDirectory(prefix='survey_arcs_') as build:
        compiler=next((p for name in ('c++','g++','clang++') if (p:=shutil.which(name))),None)
        if not compiler:raise RuntimeError('orbital strip search needs a C++17 compiler')
        cpp=Path(__file__).with_name('arc_search.cpp');path=Path(build)/'arcs.so'
        subprocess.run([compiler,'-std=c++17','-O3','-fPIC','-shared','-pthread',str(cpp),'-o',str(path)],
                       capture_output=True,check=True,timeout=45)
        lib=ctypes.CDLL(str(path));stop=lib.range_join_stop;stop.argtypes=[];stop.restype=None
        control['arcs_stop']=stop
        cores=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else 4
        threads=max(1,min(3,cores-2))
        control['certifier']=(lib,target,ex,threads)
        # Use two depth intervals to keep relatively faint pairs represented.
        # The distant worker explores its own range without a disk restriction.
        for offset,span,share,mode in ((-.18,800,.39,0),(.42,440,.52,3),(-.10,3900,1.,2)):
            remain=available-(time.monotonic()-start)
            if remain<8. or control['event'].is_set():break
            x=(src['mag']-center-offset)/.44
            score=q-1.75*np.log1p(x*x)-.8*np.maximum(0.,np.abs(x)-2.)
            seed_cap,target_cap=(2300000,3300000) if mode==3 else (1200000,2900000)
            p,g,w,stats=run_kernel(lib,src,ids,score,target[:nt] if mode==0 else target,
                                   ex,epro,threads,remain*share,span,40000,seed_cap,target_cap,mode)
            if not control['event'].is_set():
                try:
                    result=certify(lib,p,g,w,target,ex,threads,min(48.,max(7.,available*.09)))
                    control['certified'].put(result)
                    print('orbital strips',mode,stats.tolist(),'fits',len(p),'confirmed',len(result),flush=True)
                except Exception:
                    control['strip_results'].put((p,g,np.where(w<nt,w,-1),stats))
                    raise


def finish(F,control,deadline):
    from queue import Empty
    lists=F.G.setdefault('strip_pending',[])
    native=[];arrays=[]
    while True:
        try:arrays.append(control['certified'].get_nowait())
        except Empty:break
    register_measurements(F,control,arrays)
    for arr in arrays:native+=native_links(F.G,arr)
    if native:
        # Different seed queues often rediscover identical measurements.
        # Share one expensive photometric/footprint evaluation per orbit.
        native=select_native(F,native)
        F.G.setdefault('aux_completed',[]).extend(native)
        F.log('native Kepler confirmation:',len(native),'orbits submitted for scoring')
    while True:
        try:
            p,g,w,stats=control['strip_results'].get_nowait()
        except Empty:break
        F.log('single-pair orbital search:',stats.tolist(),len(p),'distinct four-night fits')
        if len(p):
            lists.append([p,g,0])
            hit=(w>=0)
            if hit.any():lists.append([np.column_stack((p[hit],w[hit])),g[hit],0])
    if not lists:return native
    output=[];stats=np.zeros(5,np.int64)
    # Each sweep visits every cadence interval and both the original quartet
    # and its independently measured fifth point. Keep progress at checkpoints.
    while lists and time.time()+1.<deadline:
        for item in list(lists):
            if time.time()+1.>=deadline:break
            p,g,i=item
            step=3072 if F.G['dev']!='cpu' else 384
            end=min(len(p),i+step)
            output.extend(R.confirm(F,(p[i:end],g[i:end],stats),deadline,quiet=True))
            item[2]=end
        lists[:]=[item for item in lists if item[2]<len(item[0])]
    F.G.setdefault('aux_completed',[]).extend(output)
    F.log('measured orbit confirmation:',len(output),'five-night orbits,',sum(len(p)-i for p,g,i in lists),'fits pending')
    return output+native


def select_native(F,links):
    G=F.G
    existing=G.setdefault('native_seen',set());unique={}
    for link in links:
        key=link['det'].astype('int64',copy=False).tobytes()
        if key not in existing:unique.setdefault(key,link)
    links=list(unique.values())
    if not links:return []
    # The marginalized likelihood includes the exposure-wise background,
    # the range of velocities allowed by the observations, and optical color.
    # Use it before the expensive repeated-checkpoint footprint bookkeeping.
    import orbit_evidence
    for lo in range(0,len(links),500):
        group=links[lo:lo+500]
        scores=orbit_evidence.values(F,group)
        for l,score in zip(group,scores):
            key=np.sort(l['det']).astype('int64',copy=False).tobytes()
            G.setdefault('orbit_evidence_cache',{})[key]=float(score)
            l['_native_order']=float(score) if np.isfinite(score) else -150.
    links.sort(key=lambda l:(l['nn']>=8,l['_native_order']),reverse=True)
    links=F.merge(links,key=lambda l:(-(l['nn']>=8),-l['_native_order']),frac=.68)
    links=links[:1800]
    for l in links:existing.add(l['det'].astype('int64',copy=False).tobytes())
    return links


def begin(F,seconds,control):
    G=F.G
    control['survey']=True
    keys=('nk','seed2_np','mag_np','lev_np','snr_np','scen','keep_np','de_np','th_np',
          'et_np','night_e_np','nights_u','season_np','cov','m5c_np','frame_np','mz_np','cb_np','lev_e')
    snapshot={k:G[k] for k in keys}
    if G.get('aux'):
        snapshot['aux']={k:G['aux'][k] for k in ('n','th_np','de_np')}
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='survey-orbits')
    future=pool.submit(work,snapshot,G['ob'].cpu().numpy().copy(),G['epro'].cpu().numpy().copy(),seconds,control)
    def report(done):
        err=done.exception()
        if err is not None:F.log('orbital strip worker:',type(err).__name__,str(err)[:220])
    future.add_done_callback(report)
    pool.shutdown(wait=False)
    return future


def request(control,p,g,seconds):
    if not len(p) or 'certifier' not in control or control['event'].is_set():return
    executor=control.get('follow_pool')
    if executor is None:
        executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='orbit-continuation')
        control['follow_pool']=executor
    lib,target,ex,threads=control['certifier']
    def follow():
        if control['event'].is_set():return
        result=certify(lib,p,g,np.full(len(p),-1),target,ex,min(threads,2),seconds)
        control['certified'].put(result)
    job=executor.submit(follow)
    def report(done):
        if not done.cancelled() and done.exception() is not None:
            print('orbital continuation:',str(done.exception())[:160],flush=True)
    job.add_done_callback(report)
    return job


def replay(control,proposal):
    surplus=control.pop('extra_orbits',None)
    if surplus is not None:
        _replay(control,proposal,65.,75.)
        _replay(control,surplus,80.,70.)
    else:
        _replay(control,proposal,65.,75.)


def _replay(control,proposal,quad_seconds,triangle_seconds):
    p,g,_=proposal
    if not len(p):return
    repeated=(p[:,1:]==p[:,:-1]).any(1)
    if (~repeated).any():request(control,p[~repeated],g[~repeated],quad_seconds)
    if repeated.any():
        tr=p[repeated]
        unique=np.column_stack((np.ones(len(tr),bool),tr[:,1:]!=tr[:,:-1]))
        request(control,tr[unique].reshape(-1,3),g[repeated],triangle_seconds)
