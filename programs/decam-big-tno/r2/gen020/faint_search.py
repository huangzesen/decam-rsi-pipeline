"""Deep seasonal linking where at least five observable nights are available.

Faint peaks are useful only if the cadence and exposure depth can support a
complete seasonal arc. That condition is imposed before expensive pair hashes.
"""
import math
import os
import time
import numpy as np


def search(F,deadline):
    G=F.G;start=time.time();out=[];seen=set()
    plans=[]
    # Cadence requirements apply locally, using the actual exposure footprints
    # and measured depths. They do not depend on catalogue identities.
    for h,step in ((32.,.0025),(64.,.00125)):
        key='deep_season_seed'
        G[key]=np.asarray(G['seed2_np']).copy()
        P=dict(h=h,dtmin=3. if h==32 else 6.,dtmax=2*h+.5,gstep=step,
               cell=8.,reach=20.,reach2=40.,tolfit=.65,tolext=1.5,gdot=1,
               minn=5,seedkey=key,maxseeds=0,maxpairs=G.get('maxp',2e6),
               useful_nights=5,deadline=deadline)
        wins=F.make_windows_days(P)
        windows=[F.prep_window(w,P) for w in wins]
        windows=[w for w in windows if len(w['idx'])>=5]
        gs=1/(20*np.exp(step*np.arange(int(math.ceil(math.log(100/20)/step))+1)))
        plan=F.mkplan(gs,P)
        cut=G.get('plan_cut')
        if cut is not None:plan=plan[:cut]
        plans.append((P,windows,plan))
        F.log('deep seasonal pool:',int(h*2),'days,',len(windows),'windows,',
              sum(w['nuse'][0] for w in windows),'initial,',sum(len(w['idx']) for w in windows),'observable on five nights')
    # Alternate window lengths and distance bands, so any timed prefix covers
    # both compact and extended seasonal arcs.
    jobs=[]
    for j in range(max(len(v[2]) for v in plans)):
        for P,windows,plan in plans:
            if j<len(plan):jobs.append((P,windows,plan[j]))
    nsearch=0;previous=0.
    for P,windows,(gs,block,band) in jobs:
        if time.time()+previous>=deadline:break
        bt=time.time();P=dict(P,blk=block)
        for g in gs:
            for W in windows:
                if time.time()>=deadline:break
                try:
                    cs=F.search_all(W,float(g),P)
                    for c in cs:
                        if c['nights']<5:continue
                        if np.all(G['seed_np'][c['det']]):continue
                        key=tuple(sorted(c['det'].tolist()))
                        if key in seen:continue
                        seen.add(key);out.append(c)
                    nsearch+=1
                except Exception as ex:
                    F.log('deep seasonal window skipped:',type(ex).__name__,str(ex)[:120])
                    W.pop('pc',None)
                    if G['dev']!='cpu':
                        import torch
                        torch.cuda.empty_cache()
        previous=time.time()-bt
    F.log('deep seasonal search:',nsearch,'distance/windows,',len(out),'complete arcs;',round(time.time()-start,1),'s')
    return out
