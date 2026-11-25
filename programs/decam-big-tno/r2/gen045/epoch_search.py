"""Blind cross-season quartets from a common-epoch sky index."""
import ctypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from survey_arcs import CERTIFIED


def supplemental(G,reference,xy,er,rows,mag,sn,obs):
    """Admit a bounded sample of secondary measurements as quartet members."""
    from pathlib import Path
    import range_search as R
    from reference_field import local_depth
    initial=len(G['keep_np'])
    det=np.arange(initial,initial+len(rows),dtype='i4')
    a=G.get('aux');nk=G['nk']
    if a:
        det=np.r_[np.arange(nk,initial,dtype='i4'),det]
        rows=np.r_[G['keep_np'][nk:],rows]
        xy=np.vstack((a['th_np'],xy))
        er=np.r_[a['de_np'],er]
        sn=np.r_[G['snr_np'][nk:],sn]
        mag=np.r_[G['mag_np'][nk:],mag]
    raw=lambda k:np.load(Path(G['scen'])/('det_'+k+'.npy'),mmap_mode='r')
    sharp,flag=raw('sharp')[rows],raw('flag')[rows]
    ok=(sn>=4.55)&np.isfinite(mag)&((flag&1)==0)&(sharp!=0.)
    j=np.flatnonzero(ok)
    if not len(j):return np.empty(0,R.SOURCE),np.empty(0,'i4')
    sn=sn[j];m=mag[j];ei=er[j];h=sharp[j];ef=flag[j]
    band=np.clip(np.floor((m-10.)*4).astype('i4'),0,79)
    count=np.bincount(ei*80+band,minlength=len(obs)*80).reshape(-1,80)
    counts=np.array([np.convolve(x,[.1,.22,.36,.22,.1],'same') for x in count])
    area=np.bincount(G['cov'][1]>>42,minlength=len(obs))*(45./3600.)**2
    density=(counts[ei,band]+3.)/np.maximum(.01,area[ei])
    depths=np.sort(G['m5c_np'][np.isfinite(G['m5c_np'])])
    fraction=(len(depths)-np.searchsorted(depths,m-.12)+1.)/(len(depths)+1.)
    q=(1.1*np.log(np.minimum(sn,15.)/5.)-.42*np.minimum(12.,((h-1.06)/np.hypot(3.7/sn,.18))**2)
       -.36*((ef&6)!=0)+.62*(.42*np.log(10.)*(np.clip(m,15.,28.)-23.)-np.log(density)+1.4*np.log(fraction)))
    n=min(len(q),650000)
    ix=np.sort(np.argpartition(-q,n-1)[:n])
    j=j[ix];q=q[ix];m=m[ix];ei=ei[ix];sn=sn[ix];pos=xy[j]
    shadow=reference.features(pos,ei,m,G['et_np'],G['lev_e'])
    visible=local_depth(G['cov'],G['m5c_np'],G['night_e_np'],pos,ei,m)
    q+=.65*np.minimum(3.,visible)-1.2*shadow[:,0]-.50*shadow[:,1]-.8*shadow[:,2]-1.30
    prof=G.get('field_profile')
    if prof is not None and prof.done() and prof.exception() is None:
        q+=.62*prof.result().transient_odds(rows[j])
    n=min(max(15000,int(.085*nk)),300000,len(q))
    ix=np.sort(np.argpartition(-q,n-1)[:n])
    j=j[ix];e=ei[ix]
    src=np.empty(n,R.SOURCE)
    src['x'],src['y']=pos[ix].T
    src['ox'],src['oy'],src['oz']=obs[e].T
    src['t'],src['night'],src['season']=G['et_np'][e],G['night_e_np'][e],G['season_np'][e]
    src['mag'],src['color'],src['sn'],src['quality']=m[ix],G['lev_e'][e],sn[ix],q[ix]
    return src,det[j]


def motion_trials(n):
    z=qmc.Sobol(9,scramble=True,seed=293).random_base2(int(np.ceil(np.log2(n))))[:n]
    # Sample a dynamically cold component and a wider distribution of
    # prograde orbits. Cell widths are large compared to centroid errors.
    norm=ndtri(np.clip(z[:,:4],.004,.996))
    broad=(np.arange(n)%3==2)
    a=np.empty_like(z)
    a[:,0]=np.clip(43.5*np.exp(norm[:,0]*np.where(broad,.18,.065)),24.,78.)
    a[:,1]=1.+norm[:,1]*np.where(broad,.12,.040)
    a[:,2]=norm[:,2]*np.where(broad,.30,.071)
    a[:,3]=norm[:,3]*np.where(broad,.24,.095)
    a[:,4]=(105.+180.*z[:,4])*np.where(np.arange(n)%3==1,2.15,1.)
    a[:,5:7]=z[:,5:7]
    a[:,7]=.63+.46*z[:,7]
    a[:,8]=z[:,8]
    a[0,:4]=(44.,1.,0.,0.)
    return a


def distant_trials(n,span):
    m=(n+1)//2
    z=qmc.Sobol(9,scramble=True,seed=809).random_base2(int(np.ceil(np.log2(m))))[:m]
    a=z.copy()
    # Stratify inverse range over successive passes so an early stopping
    # time retains coverage throughout the outer Solar System.
    step=np.arange(m)
    level=(7*step)%15
    cycle=step//15
    a[:,0]=128.*np.exp(np.log(55.)*(level+.8*z[:,0])/15.)
    speed=np.sqrt(z[:,1])*1.30
    disk=(cycle+level)%4==1
    a[:,1]=np.where(disk,1.+.17*ndtri(np.clip(z[:,1],.01,.99)),speed)
    phase=(.618033989*cycle+.381966011*level+.04*(z[:,2]-.5))%1.
    a[:,2]=np.where(disk,.30*ndtri(np.clip(z[:,2],.01,.99)),2.*np.pi*phase)
    a[:,3]=(.95*z[:,3]-.475)*np.sqrt(np.maximum(.12,2.-a[:,1]**2))
    # The uncertain tangential velocity covers a smaller angle as r^-3/2.
    # Coarse cells propose a group. A fit to its measured endpoints solves
    # for continuous transverse velocity, inverse range and range rate.
    drift=206264.806*.0172021*np.minimum(4500.,span)/a[:,0]**1.5
    parallax=206264.806/a[:,0]
    wide=(cycle+level)%3==2
    a[:,4]=np.clip((.24+.50*z[:,4])*drift+.13*parallax,17.,np.where(wide,430.,235.))
    a[:,7]=.63+.42*z[:,7]
    # Interleave the uniformly spaced orientations with independent
    # low-discrepancy trials. Near a tile boundary the two can collect
    # different sets of measured nights even at nearly identical speeds.
    other=a.copy()
    disk=step%3==1
    other[:,1]=np.where(disk,1.+.17*ndtri(np.clip(z[:,1],.01,.99)),speed)
    other[:,2]=np.where(disk,.30*ndtri(np.clip(z[:,2],.01,.99)),2.*np.pi*z[:,2])
    other[:,3]=(.95*z[:,3]-.475)*np.sqrt(np.maximum(.12,2.-other[:,1]**2))
    other[:,4]=np.clip((.24+.50*z[:,4])*drift+.13*parallax,17.,255.)
    return np.stack((other,a),axis=1).reshape(-1,9)[:n]


def search(index,src,ids,pole,epoch,trials,threads,seconds,limit=8500):
    fn=index.lib.epoch_sweeps
    p=ctypes.c_void_p;i=ctypes.c_int;d=ctypes.c_double
    fn.argtypes=[p,p,i,p,p,d,p,i,i,d,i,p,p]
    fn.restype=i
    out=np.empty(limit,CERTIFIED);stats=np.zeros(8,'i8')
    ids=np.ascontiguousarray(ids,'i4');pole=np.ascontiguousarray(pole,'f8')
    trials=np.ascontiguousarray(trials,'f8')
    count=fn(src.ctypes.data,ids.ctypes.data,len(src),index.handle,pole.ctypes.data,
             epoch,trials.ctypes.data,len(trials),threads,seconds,limit,out.ctypes.data,stats.ctypes.data)
    if count<0:raise RuntimeError('common-epoch sky search failed')
    return out[:count].copy(),stats


def work(index,src,ids,quality,repeat,depths,pole,ex,threads,seconds,control,extra):
    if len(src)<5:return
    if hasattr(os,'setpriority'):
        try:
            tid=threading.get_native_id()
            os.setpriority(os.PRIO_PROCESS,tid,max(4,os.getpriority(os.PRIO_PROCESS,tid)))
        except OSError:
            pass
    from element_search import source_marks
    end=time.monotonic()+seconds
    q=source_marks(src,quality,repeat,depths)
    n=min(1900000,len(src))
    select=np.sort(np.argpartition(-q,n-1)[:n])
    pool=src[select];own=ids[select];pool['quality']=q[select]
    if extra is not None and len(extra[0]):
        background,newids=extra
        sq=source_marks(background,background['quality'],np.zeros((len(background),1),'f4'),depths)
        extra_pool=background.copy()
        extra_pool['quality']=sq-.7
        pool=np.concatenate((pool,extra_pool));own=np.r_[own,newids]
    epochs=np.sort(ex['t'])
    epoch=float(np.median(epochs))
    if seconds>=72. and epochs[-1]-epochs[0]>130.:
        bright=pool['quality']+.68*np.log(np.minimum(pool['sn'],42.)/7.)
        nfar=min(len(pool),900000)
        ix=np.sort(np.argpartition(-bright,nfar-1)[:nfar])
        bank=distant_trials(768,epochs[-1]-epochs[0])
        far,counts=search(index,pool[ix],own[ix],pole,epoch,bank,threads,
                          min(70.,.14*seconds),limit=4500)
        control['certified'].put(far)
        print('distant common-epoch sky',counts.tolist(),'five-night fits',int((far['nights']>=5).sum()),flush=True)
    trials=motion_trials(512)
    for k in range(8):
        left=end-time.monotonic()
        if left<3. or control['event'].is_set():break
        out,stats=search(index,pool,own,pole,epoch,trials[k*64:k*64+64],threads,
                         min(left,max(12.,seconds/5.)))
        control['certified'].put(out)
        import original_recovery
        original_recovery.native_cores(control,out,min(18.,max(2.,end-time.monotonic())))
        print('common-epoch sky',stats.tolist(),'five-night fits',int((out['nights']>=5).sum()),flush=True)
        if not stats[0] and not stats[1]:break


def begin(*args):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='common-epoch')
    future=pool.submit(work,*args)
    def report(job):
        exc=job.exception()
        if exc is not None:print('common-epoch worker:',type(exc).__name__,str(exc)[:160],flush=True)
    future.add_done_callback(report)
    pool.shutdown(wait=False)
    return future
