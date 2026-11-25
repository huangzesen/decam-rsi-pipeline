"""Confirm independent-night pair hypotheses by voting in range and radial speed."""
import ctypes
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from survey_arcs import CERTIFIED

UNIT=np.dtype([('a','i4'),('b','i4'),('score','f4'),('gamma','f4')])


def units(pairs):
    bank=np.empty(len(pairs),UNIT)
    bank['a'],bank['b']=pairs['row'][:,:2].T
    bank['score'],bank['gamma']=pairs['score'],pairs['g']
    return bank


def pair_support(index,pairs,lever,east,threads,seconds):
    from scipy.stats import qmc
    ptr,integer,real=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    fn=index.lib.orbital_pair_support
    fn.argtypes=[ptr,ptr,integer,ptr,ptr,real,ptr,integer,integer,real,ptr,ptr,ptr,ptr]
    fn.restype=integer
    bank=units(pairs)
    trial=qmc.Sobol(7,scramble=True,seed=191).random_base2(5)
    trial[:,0]=(trial[:,0]-.5)*.40
    trial[:,1]=(trial[:,1]-.5)*.50
    trial[:,2]=160.+280.*trial[:,2]
    trial[0,:2]=0.
    lever,east=(np.ascontiguousarray(x,'f8') for x in (lever,east))
    merit=np.empty((len(bank),2),'f4');partner=np.empty(len(bank),'i4')
    gamma=np.empty(len(bank),'f4');stats=np.zeros(4,'i8')
    n=fn(index.handle,bank.ctypes.data,len(bank),lever.ctypes.data,east.ctypes.data,
         np.median(index.geometry['t']),trial.ctypes.data,len(trial),threads,seconds,
         merit.ctypes.data,partner.ctypes.data,gamma.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('pair corroboration failed')
    priority=merit[:,0]-.58*np.maximum(0.,merit[:,1]-3.8)
    ranks=np.arange(len(bank))
    joined=np.argsort(-priority,kind='stable')[:n]
    adjusted=ranks.astype('f8')/.78
    adjusted[joined]=np.minimum(adjusted[joined],np.arange(n)/.22)
    return np.argsort(adjusted,kind='stable'),joined,priority,partner,gamma,stats


def search(index,pairs,depth,lever,east,signal,threads,seconds,extra=None):
    ptr,integer,real=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    fn=index.lib.range_band_search
    fn.argtypes=[ptr,ptr,integer,ptr,ptr,ptr,ptr,integer,integer,real,integer,ptr,ptr]
    fn.restype=integer
    bank=units(pairs)
    depth=np.asarray(depth)
    finite=depth[np.isfinite(depth)]
    depth=np.ascontiguousarray(np.where(np.isfinite(depth),depth,np.median(finite) if len(finite) else 23.),'f8')
    lever,east=(np.ascontiguousarray(x,'f8') for x in (lever,east))
    target=index.target
    mag=target['mag']
    # Strong image measurements from deep exposures can be too faint to
    # recur elsewhere. Retain point sources with useful survey-wide flux.
    quality=(.74*np.log(np.clip(signal,4.5,28.)/6.5)-
             .55*np.log1p(((mag-np.median(depth))/.75)**2)).astype('f4')
    if extra is not None:
        ids,q=extra
        quality[ids]+=.45*np.clip(q-np.nanmedian(q),-4.,4.)
    quality[~np.isfinite(mag)]=-np.inf
    out=np.empty(11700,CERTIFIED);stats=np.zeros(8,'i8')
    n=fn(index.handle,bank.ctypes.data,len(bank),quality.ctypes.data,depth.ctypes.data,lever.ctypes.data,
         east.ctypes.data,2400,threads,max(.2,seconds),len(out),out.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('range consensus search failed')
    return out[:n].copy(),stats


def begin(index,job,depth,lever,east,signal,ids,q,control,seconds,threads,raw_rows):
    def work():
        result=job.result()
        if result is None or control['event'].is_set():return
        pairs=result[1]
        if not len(pairs):return
        start=time.monotonic();end=start+seconds
        triples=result[2]
        if len(triples) and seconds>45.:
            import three_recovery
            from survey_arcs import certify
            take=triples[:18000]
            out=certify(index,take['row'],take['g'],np.full(len(take),-1,'i4'),threads,min(7.,.025*seconds))
            control['certified'].put(out[out['nights']>=5])
            sky=control.get('original_sky')
            if sky is not None and not control['event'].is_set():
                cores=three_recovery.cores(sky,raw_rows[take['row']],take['g'],.24*take['score'])
                recovered,nt=sky.recover(cores,min(12.,.04*seconds),cap=950)
                control['raw_results'].put((recovered,[]))
                print('three-night catalogue continuations:',nt,'seeds,',len(recovered),'verified',flush=True)
        if seconds>55. and hasattr(index.lib,'shared_night_chains') and not control['event'].is_set():
            from survey_arcs import certify
            import night_chains
            import three_recovery
            chains,counts=night_chains.search(index.lib,index.target,index.geometry,signal,pairs,threads,min(11.,.035*seconds))
            four=chains[chains['row'][:,3]>=0]
            if len(four):
                fits=certify(index,four['row'],four['g'],np.full(len(four),-1,'i4'),threads,min(12.,.045*seconds))
                control['certified'].put(fits[fits['nights']>=5])
            three=chains[chains['row'][:,3]<0]
            if len(three):
                three_recovery.submit(control,raw_rows[three['row'][:,:3]],three['g'],min(18.,.06*seconds),three['merit'])
            print('shared nightly pairs:',counts.tolist(),'pivots and proposals',flush=True)
        if seconds>115. and len(pairs)>2500:
            from survey_arcs import certify
            order,joined,priority,partner,gamma,st=pair_support(index,pairs,lever,east,threads,min(40.,.11*seconds))
            seen=set();first=[]
            for i in joined:
                j=partner[i];k=(min(i,j),max(i,j))
                if k in seen:continue
                seen.add(k);first.append(i)
                if len(first)==22000:break
            if first:
                a=np.asarray(first);b=partner[a]
                quartet=np.column_stack((pairs['row'][a,:2],pairs['row'][b,:2]))
                out=certify(index,quartet,gamma[a],np.full(len(a),-1,'i4'),threads,min(22.,.07*seconds))
                control['certified'].put(out[out['nights']>=5])
            print('cross-season pair corroboration:',len(joined),'supported pairs,',len(first),'four-night seeds,',
                  st.tolist(),'in',round(time.monotonic()-start,1),'s',flush=True)
            pairs=pairs[order]
        at=0
        for share in (.25,.48,1.):
            left=end-time.monotonic()
            if left<7. or control['event'].is_set() or at>=len(pairs):break
            out,stats=search(index,pairs[at:],depth,lever,east,signal,threads,share*left,(ids,q))
            control['certified'].put(out[out['nights']>=5])
            at+=max(1,int(stats[0]))
            tri=out[out['nights']==3]
            if len(tri) and not control['event'].is_set():
                import three_recovery
                three_recovery.submit(control,raw_rows[tri['row'][:,:3]],tri['x'][:,2],
                                      min(18.,max(2.,.05*seconds)),tri['merit'])
            four=out[out['nights']==4]
            if len(four) and not control['event'].is_set():
                from original_recovery import CORE
                sky=control.get('original_sky')
                if sky is not None:
                    four=four[four['count']<=12]
                    cores=np.zeros(len(four),CORE)
                    for i,r in enumerate(four):
                        n=r['count'];c=cores[i]
                        c['count']=n;c['rows'][:n]=raw_rows[r['row'][:n]]
                        c['ref']=r['ref'];c['x']=r['x'];c['merit']=np.clip(r['merit']*.14,-3.,6.)
                    follow=min(12.,.09*max(0.,end-time.monotonic()))
                    if len(cores) and follow>1.:
                        recovered,_=sky.recover(cores,follow,cap=1400)
                        control['raw_results'].put((recovered,[]))
                        if hasattr(sky.lib,'blended_orbits') and follow>3.:
                            import blend_recovery
                            blends,_=blend_recovery.search(sky,cores,depth,.55*follow)
                            control['blend_results'].put(blends)
            print('nightly range consensus:',stats.tolist(),int((out['nights']>=5).sum()),'orbits,',
                  len(four),'four-night cores in',round(time.monotonic()-start,1),'s',flush=True)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='nightly-range')
    future=pool.submit(work);pool.shutdown(wait=False)
    def done(f):
        if f.exception():print('range consensus:',repr(f.exception())[:200],flush=True)
    future.add_done_callback(done)
    return future
