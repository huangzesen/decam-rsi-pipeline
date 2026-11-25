"""Confirm incomplete four-night orbits in the broader recovery catalogue.

Source selection is useful for blind linking, but an established orbit provides
an independent positional constraint. Every uncertain association is charged
against the density of the catalogue actually queried and all its exposures.
"""
import time
import numpy as np
import torch


def complete(F, deadline):
    G = F.G
    if not G.get('aux') or time.time() >= deadline:
        return 0
    ne, et = G['night_e_np'], G['et_np']
    seen = G.setdefault('quad_completed_seen', set())
    used = set()
    for v in G.get('grown', {}).values():
        if v is not None and v.get('nn', 0) >= 5 and np.sqrt(np.mean(v['res']**2)) < .25:
            used.update(v['det'].tolist())
    queue = []
    for v in G.get('grown', {}).values():
        if v is None or v.get('nn', 0) != 4:
            continue
        d = np.asarray(v['det'], dtype=np.int64)
        key = tuple(sorted(d.tolist()))
        if key in seen or sum(int(i) in used for i in d) >= 3:
            continue
        rms = float(np.sqrt(np.mean(v['res']**2)))
        if rms > .35 or np.ptp(et[v['exps']]) < 16:
            continue
        sn = np.maximum(G['snr_np'][d], 4.5)
        try:
            _, mag = F.redness(G['mag_np'][d], G['lev_np'][d], sn)
            phot = float(np.mean((mag-np.median(mag))**2/(.15**2+(1.09/sn)**2)))
        except Exception:
            phot = 0.
        err = np.sqrt(np.mean((1.1/sn)**2+.04**2))
        q = (rms/err)**2 + phot - .5*np.log(np.median(sn)/4.5)
        queue.append((q, key, v))
    queue.sort(key=lambda v:v[0])
    from auxiliary_completion import catalogue
    started = time.time(); count = 0; added = 0; previous = 0.
    batch = 512 if G['dev'] != 'cpu' else 64
    with catalogue(F):
        for lo in range(0,len(queue),batch):
            if time.time()+.8*previous >= deadline:
                break
            bt=time.time(); group=queue[lo:lo+batch]
            vs=[v for _,_,v in group]; n=len(vs)
            x=torch.as_tensor(np.array([v['x'] for v in vs]),device=G['dev'])
            refs=torch.as_tensor(np.array([v['tref'] for v in vs]),device=G['dev'])
            memb=torch.full((n,len(et)),-1,dtype=torch.long,device=G['dev'])
            for k,v in enumerate(vs):
                d=torch.as_tensor(v['det'],device=G['dev'])
                memb[k,G['de'][d]]=d
            chance=torch.full((n,),float('inf'),dtype=torch.float64,device=G['dev'])
            try:
                x,memb,found=F.covariance_extend(x,memb,refs,deadline,seed_n=4,chance_out=chance)
            except Exception as ex:
                F.log('four-night completion batch failed:',type(ex).__name__,str(ex)[:160])
                if G['dev']!='cpu':torch.cuda.empty_cache()
                break
            count+=n
            seen.update(key for _,key,_ in group)
            if found:
                x,memb,res=F.cfit(x,memb,refs,6,clip=.7)
                for k,(xx,mm,rr,cc) in enumerate(zip(x.cpu().numpy(),memb.cpu().numpy(),res.cpu().numpy()/F.AS,chance.cpu().numpy())):
                    ee=np.flatnonzero(mm>=0); nn=len(np.unique(ne[ee]))
                    if nn<5 or not np.isfinite(cc):continue
                    if np.sqrt(np.mean(rr[ee]**2))>.35:continue
                    v=vs[k]
                    result=dict(v,det=mm[ee],exps=ee,res=rr[ee],x=xx,nn=nn,
                                src=4,sn=4,srms=float(np.sqrt(np.mean(v['res']**2))),
                                fa=float(cc),n0=nn,nax=int((mm[ee]>=G['nk']).sum()),
                                nn0=len(np.unique(ne[ee[mm[ee]<G['nk']]])))
                    G.setdefault('aux_completed',[]).append(result);added+=1
            previous=time.time()-bt
    G['quad_completion_stat']=(count,len(queue),added)
    F.log('four-night completion:',count,'of',len(queue),'arcs,',added,'confirmed;',round(time.time()-started,1),'s')
    return added
