"""Leave-night-out predictive confidence against translated orbits."""
import ctypes
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import expit,logit

VALIDATION=np.dtype([('count','i4'),('rows','i4',48),('pad','i4'),('x','f8',6),('ref','f8')],align=True)
NIGHT=np.dtype([(k,'i4') for k in ('night','visits','controls','exceeded')]+
               [(k,'f8') for k in ('odds','expected','empirical','residual','area','analytic')],align=True)


def measure(sky, reports, times, states, seconds, threads=2, trials=24,exclude=()):
    records=[];owners=[]
    for i,c in enumerate(reports):
        rows=np.asarray(c['det'],'i4')
        if not 5<=len(rows)<=48:continue
        nights=c.get('nights')
        if nights is None:nights=len(np.unique(np.floor(times[sky.exposure[rows]]-.5)))
        if not 5<=nights<=14:continue
        r=c.get('r') or 42.
        if not 13.<r<9200.:continue
        key=tuple(sorted(rows))
        if key in exclude:continue
        state=states.get(key)
        if state is None:
            pos=np.mean(sky.pos[rows],axis=0)
            x=np.r_[pos,np.sqrt(1.+pos@pos)/r,0.,0.,0.]
            t=np.mean(times[sky.exposure[rows]])
        else:x,t=state
        records.append((rows,x,t));owners.append(i)
    batch=np.zeros(len(records),VALIDATION)
    for r,(rows,x,t) in zip(batch,records):
        r['count']=len(rows);r['rows'][:len(rows)]=rows;r['x']=x;r['ref']=t
    result=np.zeros((len(batch),14),NIGHT)
    if not len(batch):return [],result
    fn=sky.lib.offset_predictions;p,i,d=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    fn.argtypes=[p,p,i,i,i,d,p];fn.restype=i
    n=fn(sky.handle,batch.ctypes.data,len(batch),trials,threads,max(.01,seconds),result.ctypes.data)
    if n<0:raise RuntimeError('offset orbit prediction failed')
    return np.asarray(owners[:n],dtype=int),result[:n]


def evidence(test):
    t=test[test['controls']>0]
    if len(t)<3:return 0.,0.,0
    prob=np.clip(t['empirical'],1.e-7,1.)
    # Incomplete physical arcs can generate convincing orbits by chance.
    # The most ambiguous independent night matters, while several precise
    # predictions provide positive evidence against an all-chance track.
    lp=-np.log(prob)
    weak=float(np.min(lp));typical=float(np.median(lp))
    evidence=.49*np.clip(weak-2.4,-3.,4.)+.28*np.clip(typical-5.,-5.,4.)
    return float(evidence),weak,len(t)


def update(reports,owners,results,weight=1.):
    for i,test in zip(owners,results):
        delta,weak,number=evidence(test)
        if not number:continue
        c=reports[i]
        c['p']=float(expit(logit(np.clip(c['p'],1.e-14,1.-1.e-14))+weight*delta))
        c['offset_nights']=number
        c['offset_log_odds']=round(weight*delta,3)
        c['offset_weakest']=round(weak,3)


def alternatives(reports,hypotheses,sky,times):
    # Reconsider an ambiguous individual night only when the other nights
    # contain exactly the same catalogue measurements in both hypotheses.
    night=np.floor(times-.5).astype('i4')
    primary=[];owner={};choices={}
    for i,c in enumerate(reports):
        ids=set(c['det'])
        if not 5<=len(ids)<=20 or not 5<=(c.get('nights') or 5)<=9:continue
        primary.append((i,ids))
        for row in ids:owner.setdefault(row,[]).append(i)
    primary=dict(primary)
    for c in hypotheses:
        rows=set(c['det']);matches={}
        if not 5<=len(rows)<=20:continue
        for row in rows:
            for i in owner.get(row,()):matches[i]=matches.get(i,0)+1
        for i,count in matches.items():
            if count<4:continue
            ids=primary[i];a=ids-rows;b=rows-ids
            if not a or not b:continue
            def nights(values):return set(night[sky.exposure[list(values)]])
            na,nb=nights(a),nights(b)
            if len(na)!=1 or len(nb)!=1:continue
            same=nights(ids&rows)
            if len(same)<4 or same&(na|nb):continue
            prior=reports[i];old=prior.get('evidence',0.);new=c.get('evidence',old)
            if new<old-13.:continue
            rank=.36*np.clip(new-old,-12.,9.)-.8*c.get('rms',.2)
            group=choices.setdefault(i,{})
            key=tuple(sorted(rows))
            if key not in group or rank>group[key][0]:group[key]=(rank,c,na.pop(),nb.pop())
    result=[]
    for i in sorted(choices):
        for _,c,n1,n2 in sorted(choices[i].values(),key=lambda v:-v[0])[:3]:result.append((i,c,n1,n2))
    return result


def assignments(reports,choices,tests,alternative_tests,times,background=None):
    # The native tests index nights by their chronological rank in the survey.
    dates=np.unique(np.floor(times-.5).astype('i4'))
    def held(test,day):
        if test is None:return None
        use=test[(test['controls']>0)&(test['night']==np.searchsorted(dates,day))]
        return use[0] if len(use) else None
    best={}
    for j,(i,c,n1,n2) in enumerate(choices):
        a=held(tests.get(i),n1);b=held(alternative_tests.get(j),n2)
        if a is None or b is None:continue
        gain=b['odds']-a['odds']
        tail=np.log(max(1.e-8,a['empirical'])/max(1.e-8,b['empirical']))
        if gain<1.7 or tail<1.65 or b['empirical']>.025:continue
        old=reports[i].get('evidence',0.)
        evidence=.68*gain+.33*min(7.,tail)+.16*np.clip(c.get('evidence',old)-old,-12.,8.)
        if evidence>3.2 and evidence>best.get(i,(-np.inf,))[0]:best[i]=(evidence,c,j)
    for i,(likelihood,c,j) in best.items():
        parent=reports[i];new=dict(c);new['p']=parent['p']
        new['alternative_night_gain']=round(float(likelihood),3)
        # Other independent nights, image completeness and source priors are
        # substantially shared with the parent. Preserve its rank initially.
        if background is not None and 'local_density_odds' in parent:
            spatial=background.correction(c['det'])
            old=background.correction(parent['det'])
            new['p']=float(expit(logit(np.clip(parent['p'],1.e-14,1.-1.e-14))+spatial-old))
            new['local_density_odds']=round(spatial,3)
        reports[i]=new;tests[i]=alternative_tests[j]
    return len(best)


def observations(G,sky,reports,deadline):
    states=dict(G.get('survey_orbits',{}));states.update(G.get('report_orbits',{}))
    limit=min(850,len(reports));times=G['et_np']
    exclude=G.get('report_components',{})
    choices=alternatives(reports[:min(360,limit)],G.get('competing_orbits',{}).values(),sky,times)[:550]
    remaining=deadline-time.time()-.5
    use,result=measure(sky,reports[:limit],times,states,remaining*(.77 if choices else 1.),exclude=exclude)
    opt_tests={}
    if choices and time.time()+.75<deadline:
        opts,r=measure(sky,[c for _,c,_,_ in choices],times,states,deadline-time.time()-.5,exclude=exclude)
        opt_tests=dict(zip(opts,r))
    keys=[tuple(sorted(c['det'])) for c in reports[:limit]]
    return keys,dict(zip(use,result)),choices,opt_tests


def begin(F,directory,deadline):
    G=F.G;control=G.get('survey_control') or {};sky=control.get('original_sky')
    if sky is None or time.time()+1.>=deadline:return
    with (Path(directory)/'candidates.json').open() as fp:reports=json.load(fp)
    reports.sort(key=lambda c:-c['p'])
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='offset-orbits')
    control['offset_job']=pool.submit(observations,G,sky,reports,deadline)
    pool.shutdown(wait=False)


def run(F,directory,deadline):
    G=F.G;control=G.get('survey_control') or {};future=control.get('offset_job')
    if future is None or time.time()+.3>=deadline:return
    try:keys,original,choices,opt_tests=future.result(timeout=max(0.,deadline-time.time()-.35))
    except TimeoutError:return
    path=Path(directory)/'candidates.json'
    with path.open() as fp:reports=json.load(fp)
    owners={tuple(sorted(c['det'])):i for i,c in enumerate(reports)}
    tests={owners[keys[i]]:v for i,v in original.items() if keys[i] in owners}
    changed=0
    if opt_tests:
        current=[];verified={}
        for j,(i,c,n1,n2) in enumerate(choices):
            if j in opt_tests and keys[i] in owners:
                verified[len(current)]=opt_tests[j];current.append((owners[keys[i]],c,n1,n2))
        job=control.get('local_background')
        background=job.result() if job is not None and job.done() and job.exception() is None else None
        changed=assignments(reports,current,tests,verified,G['et_np'],background)
    update(reports,tests.keys(),tests.values())
    reports.sort(key=lambda c:-c['p'])
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as fp:json.dump(reports,fp)
    os.replace(tmp,path)
    F.log('offset orbit controls:',sum(c.get('offset_nights',0)>0 for c in reports),
          'tracks measured,',changed,'night assignments revised')
