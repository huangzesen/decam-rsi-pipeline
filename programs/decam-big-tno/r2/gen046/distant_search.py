"""Common-epoch parallax tests including phase-selected slow detections."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

import range_search as R
from survey_arcs import CERTIFIED


def trials(span,count=384):
    z=qmc.Sobol(9,scramble=True,seed=809).random_base2(int(np.ceil(np.log2(count))))[:count]
    step=np.arange(count);level=(7*step)%15
    r=128.*np.exp(np.log(55.)*(level+.8*z[:,0])/15.)
    speed=np.sqrt(z[:,1])*1.30
    disk=(step//15+level)%4==1
    a=z.copy();a[:,0]=r;a[:,1]=np.where(disk,1.+.17*ndtri(np.clip(z[:,1],.01,.99)),speed)
    angle=(.618033989*(step//15)+.381966011*level+.04*(z[:,2]-.5))%1.
    a[:,2]=np.where(disk,.30*ndtri(np.clip(z[:,2],.01,.99)),2.*np.pi*angle)
    a[:,3]=(.95*z[:,3]-.475)*np.sqrt(np.maximum(.12,2.-a[:,1]**2))
    drift=206264.806*.0172021*min(4500.,span)/r**1.5
    a[:,4]=np.clip((.24+.50*z[:,4])*drift+.13*206264.806/r,17.,np.where((step//15+level)%3==2,430.,235.))
    a[:,7]=.63+.42*z[:,7]
    other=a.copy();disk=step%3==1
    other[:,1]=np.where(disk,1.+.17*ndtri(np.clip(z[:,1],.01,.99)),speed)
    other[:,2]=np.where(disk,.30*ndtri(np.clip(z[:,2],.01,.99)),2.*np.pi*z[:,2])
    other[:,3]=(.95*z[:,3]-.475)*np.sqrt(np.maximum(.12,2.-other[:,1]**2))
    other[:,4]=np.clip((.24+.50*z[:,4])*drift+.13*206264.806/r,17.,255.)
    return np.stack((other,a),axis=1).reshape(-1,9)


def search(index,src,ids,pole,ex,seconds,threads):
    bank=trials(np.ptp(ex['t']))
    owner=np.ascontiguousarray(ids,'i4');pole=np.ascontiguousarray(pole,'f8')
    src=np.ascontiguousarray(src,R.SOURCE)
    fn=index.lib.epoch_sweeps
    p,i,d=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    fn.argtypes=[p,p,i,p,p,d,p,i,i,d,i,p,p];fn.restype=i
    out=np.empty(6000,CERTIFIED);stats=np.zeros(8,'i8')
    n=fn(src.ctypes.data,owner.ctypes.data,len(src),index.handle,pole.ctypes.data,float(np.median(ex['t'])),
         bank.ctypes.data,len(bank),threads,seconds,len(out),out.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('distant orbit sweep failed')
    return out[:n].copy(),stats


def run(index,src,ids,q,ex,lever,slow,offset,extra_sn,pole,seconds,threads,control):
    if control['event'].is_set() or len(src)<4 or np.ptp(ex['t'])<25.:return
    merit=q+.80*np.log(np.minimum(src['sn'],45.)/7.)
    n=min(900000,len(src));choose=np.sort(np.argpartition(-merit,n-1)[:n])
    pool=src[choose];pool['quality']=merit[choose]
    own=ids[choose]
    if len(slow):
        row=np.arange(offset,offset+len(slow),dtype='i4')
        target=index.target[row];e=target['exposure']
        extra=np.empty(len(row),R.SOURCE)
        extra['x'],extra['y'],extra['t']=target['x'],target['y'],ex['t'][e]
        extra['ox'],extra['oy'],extra['oz']=ex['ox'][e],ex['oy'][e],ex['oz'][e]
        extra['night']=ex['night'][e];extra['season']=np.floor((extra['t']-np.min(ex['t']))/365.25)
        extra['mag']=target['mag'];extra['color']=lever[e];extra['sn']=extra_sn
        extra['quality']=np.median(merit[choose])-.6+.6*np.minimum(6.,slow['quality'])
        pool=np.r_[pool,extra];own=np.r_[own,row]
    if control['event'].is_set():return
    result,stats=search(index,pool,own,pole,ex,seconds,threads)
    control['certified'].put(result)
    print('distant parallax',len(slow),'additional sources',stats.tolist(),'confirmed',int((result['nights']>=5).sum()),flush=True)


def begin(*args):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='distant-parallax')
    future=pool.submit(run,*args)
    def finished(f):
        if f.exception() is not None:print('distant parallax:',str(f.exception())[:200],flush=True)
    future.add_done_callback(finished)
    pool.shutdown(wait=False)
    return future
