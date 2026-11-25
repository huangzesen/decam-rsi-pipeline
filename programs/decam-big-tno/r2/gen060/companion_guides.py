"""Schedule pair continuations by source rarity and measured angular rate."""
import ctypes
import numpy as np

from element_search import UNIT
from survey_arcs import CERTIFIED


def prioritize(index,pairs,ex,depth,ids,quality,repeat):
    t=index.target
    n=len(t)
    ng=len(ex)
    counts=np.zeros((ng,68))
    hist=lambda m:np.nan_to_num((m-11.)*4.,nan=67.).clip(0,67).astype('i4')
    for lo in range(0,n,450000):
        a=t[lo:lo+450000]
        counts+=np.bincount(a['exposure']*68+hist(a['mag']),minlength=ng*68).reshape(ng,68)
    counts=np.array([np.convolve(r,[.10,.23,.34,.23,.10],'same') for r in counts])
    observed=counts.sum(1)
    # Use the local population of transients at this flux as a reference.
    # Mixed-depth visits may differ by orders of magnitude in peak density.
    rate=(counts+1.)/np.maximum(.2,np.minimum(1.,observed/np.maximum(200.,np.median(observed))))[:,None]
    marks=np.empty(n,'f4')
    for lo in range(0,n,450000):
        a=t[lo:lo+450000];m=a['mag'];e=a['exposure']
        q=.45*np.log(10.)*np.clip(m-23.,-8.,5.)-np.log(rate[e,hist(m)]+.25)
        delta=np.where(np.isfinite(m)&np.isfinite(depth[e]),m-depth[e]-.20,-4.)
        q-=np.logaddexp(0.,delta/.29)
        marks[lo:lo+len(m)]=np.nan_to_num(q,nan=-18.)
    marks-=np.median(marks)
    if ids is not None and len(ids):
        q=np.clip(quality-np.median(quality),-5.,6.)
        marks[ids]+=.74*q+.52*np.minimum(7.,repeat[:,0])+1.
    marks[~np.isfinite(t['mag'])]=-32.
    ab=pairs['row'][:,:2];e=t['exposure'][ab]
    span=np.abs(ex['t'][e[:,0]]-ex['t'][e[:,1]])
    m=np.mean(t['mag'][ab],axis=1)
    key=np.clip((np.log(np.maximum(.5,span))/.62+1.2).astype('i4'),0,10)*40
    key+=np.clip(((m-13.)*3.).astype('i4'),0,39)
    score=(.67*pairs['score']+.52*np.mean(marks[ab],axis=1)+.56*np.log1p(span))
    qbin=np.clip(((score+18.)*4.).astype('i4'),0,159)
    h=np.bincount(key*160+qbin,minlength=440*160).reshape(440,160)
    tail=np.cumsum(h[:,::-1],axis=1)[:,::-1]
    contrast=-np.log((tail[key,qbin]-.5*h[key,qbin]+2.)/(h.sum(1)[key]+4.))
    priority=.62*score+.82*contrast
    order=np.argsort(-priority,kind='stable')
    bank=pairs[order].copy();bank['score']=priority[order]
    # Rare remote measurements can enter even after the primary static veto.
    cap=min(n,max(50000,min(600000,int(1100*ng))))
    guide=np.argpartition(-marks,cap-1)[:cap]
    return bank,marks,np.sort(guide).astype('i4')


def search(index,pairs,marks,guide,depth,lever,east,threads,seconds):
    ptr,integer=ctypes.c_void_p,ctypes.c_int
    fn=index.lib.guided_cadence_orbits
    fn.argtypes=[ptr,ptr,integer,ptr,ptr,ptr,ptr,integer,ctypes.c_double,integer,ptr,ptr,ptr,integer]
    fn.restype=integer
    bank=np.empty(len(pairs),UNIT);bank['a'],bank['b']=pairs['row'][:,:2].T
    bank['gamma']=pairs['g'];bank['score']=pairs['score']
    depth=np.asarray(depth,'f8')
    depth=np.ascontiguousarray(np.where(np.isfinite(depth),depth,23.))
    lever,east=[np.ascontiguousarray(x,'f8') for x in (lever,east)]
    out=np.empty(10500,CERTIFIED);stats=np.zeros(8,'i8')
    n=fn(index.handle,bank.ctypes.data,len(bank),marks.ctypes.data,depth.ctypes.data,lever.ctypes.data,
         east.ctypes.data,threads,seconds,len(out),out.ctypes.data,stats.ctypes.data,guide.ctypes.data,len(guide))
    if n<0:raise RuntimeError('sparse-guide orbit search failed')
    return out[:n].copy(),stats


def local(index,pairs,marks,depth,lever,east,threads,seconds):
    from attributables import SEED
    ptr,integer=ctypes.c_void_p,ctypes.c_int
    fn=index.lib.local_companion_arcs
    fn.argtypes=[ptr,ptr,integer,ptr,ptr,ptr,ptr,integer,ctypes.c_double,integer,ptr,ptr]
    fn.restype=integer
    depth=np.ascontiguousarray(np.where(np.isfinite(depth),depth,23.),'f8')
    lever,east=[np.ascontiguousarray(x,'f8') for x in (lever,east)]
    out=np.empty(52000,SEED);stats=np.zeros(4,'i8')
    n=fn(index.handle,pairs.ctypes.data,len(pairs),marks.ctypes.data,depth.ctypes.data,
         lever.ctypes.data,east.ctypes.data,threads,seconds,len(out),out.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('local companion continuation failed')
    return out[:n].copy(),stats


def recover_original(index,seeds,control,keep,seconds):
    from original_recovery import CORE
    raw=control.get('original_sky')
    if raw is None or 'raw_results' not in control:return
    cores=np.empty(np.count_nonzero(seeds['rows'][:,3]>=0),CORE)
    if not len(cores):return
    ptr,integer=ctypes.c_void_p,ctypes.c_int
    fn=index.lib.companion_raw_cores
    fn.argtypes=[ptr,ptr,integer,ptr];fn.restype=integer
    n=fn(index.handle,seeds.ctypes.data,len(seeds),cores.ctypes.data)
    if not n:return
    cores=cores[:n];rows=cores['rows'][:,:4]
    extra=control['additional'];primary=rows<extra['offset']
    rows[~primary]=extra['rows'][rows[~primary]-extra['offset']]
    rows[primary]=keep[rows[primary]]
    result,tried=raw.recover(cores,seconds,cap=3500)
    control['raw_results'].put((result,[]))
    print('local companion original exposures:',tried,'four-night arcs,',len(result),'extensions',flush=True)
