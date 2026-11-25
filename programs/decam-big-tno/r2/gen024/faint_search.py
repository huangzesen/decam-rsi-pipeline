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
        if time.time()>=deadline:break
        key='deep_season_seed'
        G[key]=np.asarray(G['seed2_np']).copy()
        P=dict(h=h,dtmin=3. if h==32 else 6.,dtmax=2*h+.5,gstep=step,
               cell=8.,reach=20.,reach2=40.,tolfit=.65,tolext=1.5,gdot=1,
               minn=5,seedkey=key,maxseeds=0,maxpairs=G.get('maxp',2e6),
               useful_nights=5,deadline=deadline)
        wins=F.make_windows_days(P)
        windows=[]
        for w in wins:
            if time.time()>=deadline:break
            prepared=F.prep_window(w,P)
            if len(prepared['idx'])>=5:windows.append(prepared)
        gs=1/(20*np.exp(step*np.arange(int(math.ceil(math.log(100/20)/step))+1)))
        plan=F.mkplan(gs,P)
        cut=G.get('plan_cut')
        if cut is not None:plan=plan[:cut]
        plans.append((P,windows,plan))
        F.log('deep seasonal pool:',int(h*2),'days,',len(windows),'windows,',
              sum(w['nuse'][0] for w in windows),'initial,',sum(len(w['idx']) for w in windows),'observable on five nights')
    if time.time()>=deadline or not plans:return out
    # Alternate window lengths and distance bands, so any timed prefix covers
    # both compact and extended seasonal arcs.
    jobs=[]
    # Give each range block one search before spending its additional coarse
    # samples. Long-window blocks contain twice as many samples; running a
    # whole block at once made a short time budget cover only a few distances.
    layers=max((len(job[0]) for _,_,plan in plans for job in plan),default=0)
    for layer in range(layers):
        for j in range(max(len(v[2]) for v in plans)):
            for P,windows,plan in plans:
                if j<len(plan) and layer<len(plan[j][0]):
                    gg,block,band=plan[j]
                    jobs.append((P,windows,(gg[layer:layer+1],block,band)))
    nsearch=0;previous=0.
    for P,windows,(gs,block,band) in jobs:
        if time.time()+previous>=deadline:break
        bt=time.time();P=dict(P,blk=block)
        for g in gs:
            for W in windows:
                if time.time()>=deadline:break
                try:
                    cs=search_window(F,W,float(g),P)
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


def search_window(F,W,g,P):
    """Split a crowded field spatially before discarding its faint sources."""
    if not W.get('spatially_split'):
        try:
            result=F.search_window(W,g,P)
        except RuntimeError as ex:
            if 'out of memory' not in str(ex).lower():raise
            W['over']=True
            if F.G['dev']!='cpu':
                import torch
                torch.cuda.empty_cache()
            result=[]
        if not W.pop('over',False):return result
        W.pop('pc',None)
        W['spatially_split']=True
        F.log('faint window uses sky tiles:',len(W['idx']),'seeds retained')
    from sky_tiles import tiles
    result=[]
    for sub in tiles(F,W,P,P['blk']):
        if time.time()>=P['deadline']:break
        result.extend(F.search_all(sub,g,P))
        sub.pop('pc',None)
    return result


def complete(F,deadline):
    """Verify the supplement before merging it with checkpointed candidates."""
    G=F.G
    raw=search(F,time.time()+.72*max(0.,deadline-time.time()))
    if not raw:return []
    grown=F.grow(F.merge(raw),deadline)
    good=[]
    for L in grown:
        e=np.asarray(L['exps']);d=np.asarray(L['det']);x=np.asarray(L['x'])
        nn=len(np.unique(G['night_e_np'][e]))
        residual=np.asarray(L['res'])
        if nn<5 or not np.isfinite(residual).all() or np.sqrt(np.mean(residual**2))>=.35:continue
        if (not np.isfinite(x).all() or x[2]<=0 or
            np.sum(x[3:6]**2)*np.sqrt(1+np.sum(x[:2]**2))>=1.99*F.MU*x[2]**3):continue
        good.append(dict(L,nn=nn,nax=int((d>=G['nk']).sum()),
                         nn0=len(np.unique(G['night_e_np'][e[d<G['nk']]]))))
    G.setdefault('aux_completed',[]).extend(good)
    F.log('faint seasonal confirmation:',len(good),'verified arcs')
    return good
