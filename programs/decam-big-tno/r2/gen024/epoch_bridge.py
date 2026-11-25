"""Supplement seasonal searches with short-pair predictions in other years."""
import os
import time
import numpy as np
from auxiliary_completion import catalogue


def search(F, scen, deadline):
    G=F.G
    if time.time()>=deadline or 'box' not in G:return []
    started=time.time();nk=G['nk'];original=G['seed_np']
    sharp=np.load(os.path.join(scen,'det_sharp.npy'),mmap_mode='r')[G['keep_np'][:nk]]
    # Restrict the *proposal* catalogue to plausible point sources. Confirmation
    # subsequently uses the full primary and auxiliary astrometric catalogues.
    seed=original&(sharp>=.8)&(sharp<=1.3)
    settings=dict(SP_DTU=32.,SP_SN=5.2,SP_SNHI=1e9,SP_DEEP=0.,SP_MAX=700000,
                  SP_MAXU=6000000,SP_TU=.20,SP_PRE=0.,SP_HOT=0.,SP_COS=.3,SP_V0=.2)
    saved={k:getattr(F,k) for k in settings}
    out=[]
    try:
        G['seed_np']=seed;G['epoch_bridge']=True
        for k,v in settings.items():setattr(F,k,v)
        with catalogue(F):
            raw=F.sparse_pass(time.time()+.82*max(deadline-time.time(),0.),(25.,100.))
            # Record actual hypotheses, without any catalogue identities.
            G['bridge_trace']=[G['keep_np'][np.asarray(c['det'],dtype=np.int64)].tolist() for c in raw[:20000]]
            grown=F.grow(F.merge(raw),deadline) if raw else []
            for L in grown:
                d=np.asarray(L['det']);e=np.asarray(L['exps']);x=np.asarray(L['x'])
                nn=len(np.unique(G['night_e_np'][e]))
                if nn<5 or np.sqrt(np.mean(np.asarray(L['res'])**2))>=.35:continue
                if (not np.isfinite(x).all() or x[2]<=0 or
                    np.sum(x[3:6]**2)*np.sqrt(1+np.sum(x[:2]**2))>=1.99*F.MU*x[2]**3):continue
                out.append(dict(L,nn=nn,nax=int((d>=nk).sum()),
                                nn0=len(np.unique(G['night_e_np'][e[d<nk]]))))
    finally:
        G['seed_np']=original;G.pop('epoch_bridge',None)
        for k,v in saved.items():setattr(F,k,v)
    G.setdefault('aux_completed',[]).extend(out)
    F.log('cross-season bridge:',len(out),'verified arcs;',round(time.time()-started,1),'s')
    return out
