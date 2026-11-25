"""Search other nights with the common covariance of two orbital forecasts."""
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import time

import numpy as np

from original_recovery import CORE, RECOVERY, integer, ptr, real


def interleave(groups):
    groups=list(groups)
    if not any(len(v) for v in groups):return np.empty(0,groups[0].dtype if groups else CORE)
    groups=[v for v in groups if len(v)]
    if len(groups)==1:return groups[0]
    rank=np.concatenate([np.arange(len(v)) for v in groups])
    return np.concatenate(groups)[np.argsort(rank,kind='stable')]


def prepare(sky, proposals):
    rows=proposals['row']
    seeds=np.zeros(len(rows),CORE)
    if not len(rows):return seeds
    valid=rows>=0;size=valid.sum(1)
    r=np.where(valid,rows,rows[:,:1])
    p=sky.pos[r].astype('f8');e=sky.geometry[sky.exposure[r]]
    g=proposals['g'].astype('f8');h=proposals['h'].astype('f8')
    ref=(e['t']*valid).sum(1)/size
    dt=e['t']-ref[:,None]-1./(173.14463268467*g[:,None])
    obs=np.stack((e['ox'],e['oy']),axis=2)
    u=p*(1.+h[:,None]*dt-g[:,None]*e['oz'])[:,:,None]+g[:,None,None]*obs
    mean=(dt*valid).sum(1)/size
    t=(dt-mean[:,None])*valid
    v=np.einsum('nki,nk->ni',u,t)/np.maximum(1.e-8,(t*t).sum(1))[:,None]
    seeds['x'][:,:2]=(u*valid[:,:,None]).sum(1)/size[:,None]-v*mean[:,None]
    seeds['x'][:,2]=g;seeds['x'][:,3:5]=v;seeds['x'][:,5]=h
    seeds['ref']=ref;seeds['rows'][:,:4]=rows;seeds['count']=size
    seeds['merit']=np.clip(.25*(proposals['merit']-8.),-4.,6.)
    return seeds


def joint(sky,cores,depth,seconds,threads=2,cap=5000):
    if not len(cores):return np.empty(0,RECOVERY),0
    depth=np.ascontiguousarray(depth,'f8')
    cores=np.ascontiguousarray(cores,CORE)
    out=np.empty(cap,RECOVERY);tested=integer()
    fn=sky.lib.joint_return_fits
    fn.argtypes=[ptr,ptr,ptr,ptr,integer,integer,real,integer,ptr,ptr];fn.restype=integer
    from ctypes import byref
    n=fn(sky.handle,getattr(sky,'paired_returns',None),depth.ctypes.data,cores.ctypes.data,len(cores),
         threads,max(.05,seconds),cap,out.ctypes.data,byref(tested))
    if n<0:raise RuntimeError('two-night forecast search failed')
    return out[:n].copy(),tested.value


def submit_windows(F,candidates):
    G=F.G;control=G.get('survey_control') or {}
    sky=control.get('original_sky')
    if sky is None or not hasattr(sky.lib,'joint_return_fits') or control['event'].is_set():return
    prior=control.get('window_return_job')
    if prior is not None and not prior.done():return
    remaining=G['hard_end']-time.time()
    if remaining<12.:return
    def run():
        seen=control.setdefault('tested_window_returns',set());bank=[]
        deadline=time.time()+min(30.,.12*remaining)
        for k,c in enumerate(candidates):
            if k%256==0 and (control['event'].is_set() or time.time()>deadline):break
            if c['nights'] not in (3,4) or len(c['det'])>12 or not (1./9000.<c['g']<1./15.):continue
            d=c['det']
            if np.max(d)>=len(G['keep_np']):continue
            rows=G['keep_np'][d]
            key=tuple(sorted(rows))
            if key in seen:continue
            seen.add(key)
            st=c['st'];x=[st[0],st[1],c['g'],st[2],st[3],c.get('gd',0.)]
            e=sky.exposure[rows];arc=np.ptp(sky.geometry['t'][e]);m=sky.mag[rows]
            if not np.isfinite(m).all() or arc<4.:continue
            merit=(.57*np.log1p(arc/25.)+.67*np.log(np.median(np.minimum(35.,sky.arrays[0][rows])))
                   -1.75*np.std(m)-1.9*min(.7,c['rms'])+.65*(c['nights']-3))
            bank.append((merit,rows,x,c['tref'],key))
            if len(bank)>=80000:break
        if not bank or control['event'].is_set():return
        bank.sort(key=lambda c:-c[0])
        arr=np.zeros(len(bank),CORE)
        for r,(merit,rows,x,ref,_) in zip(arr,bank):
            r['x']=x;r['ref']=ref;r['count']=len(rows);r['rows'][:len(rows)]=rows;r['merit']=min(4.,merit-2.)
        budget=min(48.,.25*(G['hard_end']-time.time()))
        if budget<=.2 or control['event'].is_set():return
        hits,n=joint(sky,arr,G['m5c_np'],budget,1 if G['dev']=='cpu' else 2,cap=4200)
        control['raw_results'].put((hits,[]))
        for row in bank[n:]:seen.discard(row[-1])
        print('joint returns from window arcs:',n,'cores,',len(hits),'confirmed',flush=True)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='window-returns')
    job=pool.submit(run);control['window_return_job']=job
    job.add_done_callback(lambda j:F.log('window orbit returns:',str(j.exception())[:165]) if j.exception() else None)
    pool.shutdown(wait=False)


def reduced_reports(F,reports,control):
    G=F.G;sky=control['original_sky'];seen=control.setdefault('tested_reduced_reports',set())
    states=G.get('report_orbits',{});survey=G.get('survey_orbits',{})
    nmax=360
    bank=[];unique=set()
    for c in sorted(reports,key=lambda v:-v['p'])[:nmax]:
        rows=np.asarray(c['det'],'i4');key=tuple(sorted(rows))
        if key in seen or len(rows)>16:continue
        orbit=states.get(key,survey.get(key))
        if orbit is None:continue
        x,ref=orbit;e=sky.exposure[rows];n=sky.geometry['night'][e]
        dates=np.unique(n)
        if not 5<=len(dates)<=7:continue
        seen.add(key)
        # Two withheld dates prevent a single unrelated season from fixing
        # the preliminary range. Its replacements can be in any exposure.
        keep=min(4,len(dates)-2)
        values=[]
        for subset in combinations(dates,keep):
            r=rows[np.isin(n,subset)]
            if len(r)>12:continue
            k=tuple(sorted(r))
            if k in unique:continue
            arc=np.ptp(sky.geometry['t'][sky.exposure[r]])
            merit=np.log1p(arc/45.)+.42*np.log(np.median(sky.arrays[0][r]))+.57*np.log1p(len(r)-keep)
            unique.add(k);values.append((merit,r))
        values.sort(key=lambda v:-v[0])
        for merit,r in values[:16]:bank.append((r,x,ref,merit))
    cores=np.zeros(len(bank),CORE)
    for c,(r,x,ref,merit) in zip(cores,bank):
        c['x']=x;c['ref']=ref;c['merit']=np.clip(.42*(merit-4.),-2.,3.)
        c['count']=len(r);c['rows'][:len(r)]=r
    return cores
