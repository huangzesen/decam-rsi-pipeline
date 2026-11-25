"""Overlapping time windows and depth maps for single-detection orbit votes."""
import ctypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from survey_arcs import CERTIFIED


def opportunities(src, exid, ex, depths, east):
    """Probability of enough independent nights along a plausible trajectory."""
    depth = np.asarray(depths)
    finite = depth[np.isfinite(depth)]
    pivot = np.median(finite) if len(finite) else np.nanmedian(src['mag'])
    depth = np.where(np.isfinite(depth),depth,pivot-1.)
    width = np.deg2rad(.23)
    # A covered tile is supported by a measurement in the input exposures.
    # Several exposures of one night contribute only their best depth.
    low = np.array([src['x'].min(),src['y'].min()])
    x = np.floor((src['x']-low[0])/width).astype('i4')
    y = np.floor((src['y']-low[1])/width).astype('i4')
    base = int(y.max())+1
    sw = (x.max()+1)*base
    tiles = np.unique(exid.astype('i8')*sw+x*base+y)
    ei,grid = tiles//sw,tiles%sw
    pos = np.column_stack(((grid//base+.5)*width+low[0],(grid%base+.5)*width+low[1]))
    result = np.zeros(len(src),'f4')
    night = ex['night'];nn=int(night.max())+1
    epoch = np.median(ex['t'])
    for radius,along,north in ((43.,1.,0.),(34.,1.,0.),(55.,1.,0.),(43.,.89,.22),
                               (43.,1.10,-.25),(38.,.98,-.34),(50.,.95,.29),(200.,.9,.25)):
        g=1./radius;vc=.0172021*g**1.5
        vx=(along*east[0]-north*east[1])*vc
        vy=(along*east[1]+north*east[0])*vc
        fx=1.-g*ex['oz']
        ax=g*ex['ox']-vx*(ex['t']-epoch)
        ay=g*ex['oy']-vy*(ex['t']-epoch)
        # Scattered source measurements need only a coarse occupancy map.
        # Include a one-tile margin to account for the focal-plane boundary.
        bx=np.floor((pos[:,0]*fx[ei]+ax[ei])/width).astype('i4')
        by=np.floor((pos[:,1]*fx[ei]+ay[ei])/width).astype('i4')
        x0,y0=bx.min()-2,by.min()-2
        ny=int(by.max()-y0+3);nx=int(bx.max()-x0+3)
        cell=(bx-x0)*ny+(by-y0)
        if nx*ny>4000000:continue
        keys=cell.astype('i8')*nn+night[ei]
        order=np.lexsort((-depth[ei],keys));v=keys[order]
        take=order[np.r_[True,v[1:]!=v[:-1]]]
        order=np.lexsort((-depth[ei[take]],cell[take]))
        ix=take[order];sc=cell[ix]
        unique,start,count=np.unique(sc,return_index=True,return_counts=True)
        deepest=np.full((nx,ny),pivot-3.,'f4')
        ok=count>=5
        deepest.ravel()[unique[ok]]=depth[ei[ix[start[ok]+4]]]
        for lo in range(0,len(src),250000):
            sl=slice(lo,lo+250000);e=exid[sl]
            ix=np.floor((src['x'][sl]*fx[e]+ax[e])/width).astype('i4')-x0
            iy=np.floor((src['y'][sl]*fx[e]+ay[e])/width).astype('i4')-y0
            ix=np.clip(ix,1,nx-2);iy=np.clip(iy,1,ny-2)
            limit=(.50*deepest[ix,iy]+.125*(deepest[ix-1,iy]+deepest[ix+1,iy]+
                                            deepest[ix,iy-1]+deepest[ix,iy+1]))
            result[sl]+=1./(1.+np.exp(np.clip((src['mag'][sl]-.12-limit)/.30,-30.,30.)))
    return np.log(.035+result/8.)


def settings(family, count):
    points=qmc.Sobol(10,scramble=True,seed=831+family).random_base2(int(np.ceil(np.log2(count))))[:count]
    z=ndtri(np.clip(points,.012,.988))
    r=np.empty((count,10))
    if family==0:
        r[:,0]=43.*np.exp(.118*z[:,0])
        r[:,1]=np.clip(1.+.078*z[:,1],.69,1.28)
        r[:,2]=.119*z[:,2];r[:,3]=.14*z[:,3]
        r[:,4]=55.+95.*points[:,4];r[:,5]=680.+1200.*points[:,5]
    elif family==1:
        r[:,0]=40.*np.exp(.245*z[:,0])
        r[:,1]=np.clip(.97+.185*z[:,1],.48,1.31)
        r[:,2]=.43*z[:,2];r[:,3]=.29*z[:,3]
        r[:,4]=55.+130.*points[:,4];r[:,5]=480.+1720.*points[:,5]
    elif family==2:
        r[:,0]=np.exp(np.log(115.)+points[:,0]*np.log(7500./115.))
        r[:,1]=.45+.75*points[:,1]
        r[:,2]=2.*np.pi*points[:,2];r[:,3]=.50*z[:,3]
        r[:,4]=13.+72.*points[:,4];r[:,5]=900.+4400.*points[:,5]
    elif family==4:
        broad=np.arange(count)%3!=0
        r[:,0]=np.exp(np.log(66.)+points[:,0]*np.log(410./66.))
        r[:,1]=np.where(broad,1.33*np.sqrt(points[:,1]),1.+.16*z[:,1])
        r[:,2]=np.where(broad,2.*np.pi*points[:,2],.30*z[:,2])
        r[:,3]=(.95*points[:,3]-.475)*np.sqrt(np.maximum(.08,2.-r[:,1]**2))
        r[:,5]=520.+np.minimum(3700.,5.8*r[:,0])*(.55+points[:,5])
        drift=206264.806*.0172021*r[:,5]/r[:,0]**1.5
        r[:,4]=np.clip((.19+.25*points[:,4])*drift+.048*206264.806/r[:,0],26.,275.)
    else:
        raise ValueError('unknown orbit family')
    r[:,6:10]=points[:,6:10]
    return np.ascontiguousarray(r)


def sweep(index,src,ids,pole,trials,threads,seconds):
    p,i,d=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    func=index.lib.local_orbit_charts
    func.argtypes=[p,p,i,p,p,p,i,i,d,i,p,p];func.restype=i
    trials=np.ascontiguousarray(trials,'f8')
    pole=np.ascontiguousarray(pole,'f8');ids=np.ascontiguousarray(ids,'i4')
    out=np.empty(8000,CERTIFIED);stats=np.zeros(8,'i8')
    n=func(src.ctypes.data,ids.ctypes.data,len(src),index.handle,pole.ctypes.data,
           trials.ctypes.data,len(trials),threads,seconds,len(out),out.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('moving sky cells failed')
    return out[:n].copy(),stats


def catalogues(src,ids,base,repeat,depths,chance):
    finite=depths[np.isfinite(depths)]
    pivot=np.median(finite) if len(finite) else np.median(src['mag'])
    q=base+.82*np.minimum(repeat[:,0],6.)+1.25*chance
    q=q-1.18*np.log1p(((src['mag']-pivot)/.70)**2)
    return q


def search(index,src,ids,base,repeat,depths,ex,pole,east,threads,seconds,control,extra=None):
    if hasattr(os,'setpriority'):
        try:
            tid=threading.get_native_id()
            os.setpriority(os.PRIO_PROCESS,tid,max(4,os.getpriority(os.PRIO_PROCESS,tid)))
        except OSError:
            pass
    end=time.monotonic()+seconds
    if len(src)<4:return
    q=catalogues(src,ids,base,repeat,depths,opportunities(src,index.target['exposure'][ids],ex,depths,east))
    secondary,secondary_rows = extra if extra is not None else (None,None)
    if secondary is not None and len(secondary):
        sq = catalogues(secondary,secondary_rows,secondary['quality'],
                        np.zeros((len(secondary),1),'f4'),depths,
                        opportunities(secondary,index.target['exposure'][secondary_rows],ex,depths,east))
        # Secondary measurements passed a different selection function.
        # Normalize their source marks against the edge of the primary pool.
        sq += np.quantile(q,.67)-np.median(sq)-.70
    else:
        sq = None
    import epoch_search
    common = hasattr(index.lib,'epoch_sweeps')
    for fam,cap,portion in ((0,900000,.23),(2,650000,.17),(4,700000,.14),(3,1650000,.24),
                            (0,1600000,.52),(1,1150000,1.)):
        remaining=end-time.monotonic()
        if remaining<4. or control['event'].is_set():break
        if fam==3 and not common:continue
        n=min(cap,len(ids))
        weight=q if fam!=2 else base+1.1*np.log(np.minimum(src['sn'],40.)/7.)
        ne=min(len(sq),int(n*.13)) if sq is not None else 0
        n=max(1,n-ne)
        selected=np.sort(np.argpartition(-weight,n-1)[:n])
        a=src[selected];a['quality']=weight[selected];owner=ids[selected]
        if ne:
            choose=np.sort(np.argpartition(-sq,ne-1)[:ne]);add=secondary[choose]
            add['quality']=sq[choose]
            a=np.concatenate((a,add));owner=np.r_[owner,secondary_rows[choose]]
        if common and fam in (2,3):
            trials=epoch_search.distant_trials(512,np.ptp(ex['t'])) if fam==2 else epoch_search.motion_trials(192)
            out,stats=epoch_search.search(index,a,owner,pole,float(np.median(ex['t'])),trials,threads,portion*remaining)
        else:
            trials=settings(fam,512)
            if cap>1000000:trials=trials[11:]
            out,stats=sweep(index,a,owner,pole,trials,threads,portion*remaining)
        control['certified'].put(out)
        import cadence_search
        cadence_search.remember(control,out,'charts')
        print('co-moving sky cells',fam,len(a),stats.tolist(),'reports',len(out),flush=True)
        if control.get('original_sky') is not None and not control['event'].is_set():
            import original_recovery
            original_recovery.verify_cores(control,out,min(6.,max(.25,end-time.monotonic())))


def begin(*args):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='orbit-charts')
    future=pool.submit(search,*args)
    def finished(job):
        if job.exception() is not None:print('co-moving sky cells:',str(job.exception())[:180],flush=True)
    future.add_done_callback(finished);pool.shutdown(wait=False)
    return future
