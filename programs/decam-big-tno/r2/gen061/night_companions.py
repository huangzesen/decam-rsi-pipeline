"""Find improbable pairs and triples of transients across nearby nights."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

ARC = 180.*3600./np.pi
MU = .0002959122083*1.0013418
POINT = np.dtype([(k,'f4') for k in ('x','y','m','sn','mark')]+[('e','i4'),('id','i4')])
DATE = np.dtype([(k,'f8') for k in ('t','ox','oy','oz','depth','color')]+[('night','i4'),('pad','i4')])
VIEW = np.dtype([('a','i4'),('b','i4'),('at','f8',2),('mx','f8',2),('my','f8',2)]+
                [(k,'f8') for k in ('xx','xy','yy','g','sg','gx','gy','weight')])
HIT = np.dtype([('row','i4'),('score','f4'),('g','f4'),('sigma','f4')])
SEED = np.dtype([('row','i4',3),('score','f4'),('g','f4')])


def schedule(dates, lo, hi, frame, nominal, spread, along, across, fraction, days, limit):
    center=(hi+lo)/2.
    angle=np.deg2rad(23.43928)
    pole=frame@np.array([0.,-np.sin(angle),np.cos(angle)])
    t=dates['t']
    a,b=np.where((dates['night'][:,None]!=dates['night'])&
                 (np.abs(t[:,None]-t)<=days)&(np.abs(t[:,None]-t)>.45))
    if not len(a):return np.empty(0,VIEW)
    ob=np.column_stack([dates[k] for k in ('ox','oy','oz')])
    dt=t[b]-t[a]
    good=(np.linalg.norm(center[b]-center[a],axis=1)<
          np.linalg.norm((hi[b]-lo[b]+hi[a]-lo[a])*.5,axis=1)+
          1.4*np.linalg.norm(ob[b]-ob[a],axis=1)/nominal+.00012*np.abs(dt))
    a,b,dt=a[good],b[good],dt[good]
    if not len(a):return np.empty(0,VIEW)
    pos=center[a]
    def predict(xy, radius=nominal, east_rate=1., north_rate=0.):
        n=np.column_stack((xy,np.ones(len(xy))))
        n/=np.linalg.norm(n,axis=1)[:,None]
        dot=np.sum(n*ob[a],axis=1)
        rho=np.sqrt(radius**2-np.sum(ob[a]**2,axis=1)+dot**2)-dot
        r=n*rho[:,None]+ob[a]
        s=r/radius
        east=np.cross(pole,s)
        east/=np.maximum(np.linalg.norm(east,axis=1)[:,None],1.e-8)
        v=east_rate*east+north_rate*np.cross(s,east)
        phase=np.sqrt(MU/radius**3)*dt
        gpos=(s*np.cos(phase)[:,None]+v*np.sin(phase)[:,None])*radius-ob[b]
        return gpos[:,:2]/gpos[:,2,None]
    base=predict(pos)
    d=.002
    dr=(predict(pos,nominal/(1.+d))-predict(pos,nominal/(1.-d)))*(ARC*spread/(2*d))
    du=(predict(pos,east_rate=1.+along)-base)*ARC
    dv=(predict(pos,north_rate=across)-base)*ARC
    mx=(predict(pos+[d,0.])-predict(pos-[d,0.]))/(2*d)-[1.,0.]
    my=(predict(pos+[0.,d])-predict(pos-[0.,d]))/(2*d)-[0.,1.]
    offset=(base-pos-pos[:,0,None]*mx-pos[:,1,None]*my)*ARC
    cov=sum(z[:,:,None]*z[:,None,:] for z in (dr,du,dv))
    cov+=np.eye(2)*(.27**2+(.0018*np.abs(dt))**2)[:,None,None]
    inv=np.linalg.inv(cov)
    gamma=np.sqrt(1.+(pos*pos).sum(1))/nominal
    gain=(inv@dr[:,:,None])[:,:,0]
    sg=gamma*spread*np.sqrt(np.maximum(1.e-6,1.-(gain*dr).sum(1)))
    v=np.empty(len(a),VIEW)
    v['a'],v['b'],v['at'],v['mx'],v['my']=a,b,offset,mx,my
    v['xx'],v['xy'],v['yy']=cov[:,0,0],cov[:,0,1],cov[:,1,1]
    v['g'],v['sg']=gamma,np.hypot(sg,.003*gamma)
    v['gx'],v['gy']=(gain*(gamma*spread)[:,None]).T
    v['weight']=np.log(fraction)
    shift=base-pos
    lower=np.maximum(lo[a]+shift,lo[b]);upper=np.minimum(hi[a]+shift,hi[b])
    overlap=np.maximum(0.,upper-lower).prod(1)/np.maximum(1e-9,(hi[a]-lo[a]).prod(1))
    merit=np.log(.015+overlap)+.75*np.clip(dates['depth'][b]-dates['depth'][a],-2.,2.)-.80*np.log1p(np.abs(dt))
    order=np.lexsort((-merit,a))
    chosen=[]
    for group in np.split(order,np.flatnonzero(np.diff(a[order]))+1):
        nights={};count=0
        for k in group:
            if overlap[k]<.002:continue
            night=dates['night'][b[k]]
            # Pointings from the same night can cover different CCDs.
            if nights.get(night,0)>=3:continue
            chosen.append(k);nights[night]=nights.get(night,0)+1;count+=1
            if count>=limit:break
    return v[chosen]


def unique(records, rows, limit, priority=None):
    if not len(records):return records
    order=np.argsort(-(records['score'] if priority is None else priority),kind='stable')
    keys=np.sort(rows[order],axis=1)
    _,first=np.unique(keys.view(np.dtype((np.void,keys.dtype.itemsize*keys.shape[1]))).ravel(),
                      return_index=True)
    return records[order[np.sort(first)[:limit]]]


def build(scen,raw,th,e,m,sn,seed,frame,ob,et,night,depth,color,threads,aux=None):
    begin=time.monotonic()
    load=lambda k:np.load(Path(scen)/('det_'+k+'.npy'),mmap_mode='r')[raw]
    shape=load('sharp')
    flag=load('flag')
    ds=(shape-1.04)/np.hypot(.17,3.5/np.maximum(4.5,sn))
    use=(np.isfinite(m)&(shape!=0.)&((flag&1)==0)&((np.abs(ds)<2.9)|seed))
    ids=np.flatnonzero(use)
    if len(ids)<5:return None
    primary=len(raw)
    if aux is not None and len(aux['rows']):
        raw2=aux['rows'];s2=aux['snr']
        h2=np.load(Path(scen)/'det_sharp.npy',mmap_mode='r')[raw2]
        f2=np.load(Path(scen)/'det_flag.npy',mmap_mode='r')[raw2]
        d2=(h2-1.04)/np.hypot(.17,3.5/np.maximum(4.5,s2))
        q=np.flatnonzero((f2==0)&(h2!=0.)&(np.abs(d2)<1.75)&(s2>=5.0)&np.isfinite(aux['mag']))
        # Cells without deep independent stationary references can contain
        # whole star fields. Limit their contribution to each exposure.
        if len(q):
            e2=aux['de'];merit=np.log(np.minimum(30.,s2[q]))-.42*d2[q]**2
            rank=np.lexsort((-merit,e2[q]))
            groups=e2[q[rank]];starts=np.maximum.accumulate(np.where(
                np.r_[True,groups[1:]!=groups[:-1]],np.arange(len(q)),0))
            q=q[rank[np.arange(len(q))-starts<1500]]
            ids=np.r_[ids,primary+q]
            th=np.concatenate((th,aux['th']));e=np.r_[e,e2];m=np.r_[m,aux['mag']]
            sn=np.r_[sn,s2];ds=np.r_[ds,d2];flag=np.r_[flag,f2]
            seed=np.r_[seed,np.zeros(len(s2),bool)]
    n=len(ids)
    p=np.empty(n,POINT)
    p['x'],p['y']=(th[ids]*ARC).T
    p['m'],p['sn'],p['e'],p['id']=m[ids],sn[ids],e[ids],np.arange(n,dtype='i4')
    p['mark']=np.clip(.82*np.log(np.minimum(24.,sn[ids])/6.)-.42*ds[ids]**2-
                        .45*(~seed[ids])-.7*((flag[ids]&4)!=0),-4.,1.3)
    obs=np.zeros(len(et),DATE)
    obs['t'],obs['night'],obs['depth'],obs['color']=et,night,np.nan_to_num(depth,nan=23.),color
    obs['ox'],obs['oy'],obs['oz']=ob.T
    count=np.bincount(p['e'],minlength=len(et))
    order=np.argsort(p['e'],kind='stable') if np.any(np.diff(p['e'])<0) else slice(None)
    start=np.r_[0,np.cumsum(count)[:-1]][count>0]
    lo,hi=np.zeros((len(et),2)),np.zeros((len(et),2))
    for j,key in enumerate(('x','y')):
        v=p[key][order]
        lo[count>0,j]=np.minimum.reduceat(v,start)/ARC
        hi[count>0,j]=np.maximum.reduceat(v,start)/ARC
    # The cold and excited components cover classical and scattered orbits.
    # A separate slower component reaches visits outside the nearest season.
    maps=np.concatenate([schedule(obs,lo,hi,frame,*family) for family in
            ((43.,.14,.115,.20,.72,48.,25),(35.,.24,.28,.42,.22,30.,13),
             (160.,.53,.55,.67,.060,85.,16),(23.,.26,.45,.65,.045,19.,10))])
    score=np.zeros(n,'f4')
    pairs=np.zeros((n,3),HIT);pairs['row']=-1
    tri=np.zeros(n,SEED);tri['score']=-1.e20
    with tempfile.TemporaryDirectory(prefix='night_companions_') as tmp:
        compiler=next((s for k in ('c++','g++','clang++') if (s:=shutil.which(k))),None)
        if not compiler:return None
        path=Path(tmp)/'companion.so'
        subprocess.run([compiler,'-O3','-std=c++17','-fPIC','-shared','-pthread',
                        str(Path(__file__).with_suffix('.cpp')),'-o',str(path)],
                       check=True,capture_output=True,timeout=45)
        fn=ctypes.CDLL(str(path)).night_companion_alternatives
        ptr,integer=ctypes.c_void_p,ctypes.c_int
        fn.argtypes=[ptr,integer,ptr,integer,ptr,integer,integer,ptr,ptr,ptr];fn.restype=integer
        status=fn(p.ctypes.data,n,obs.ctypes.data,len(obs),maps.ctypes.data,len(maps),
                  threads,score.ctypes.data,pairs.ctypes.data,tri.ctypes.data)
    if status<0:raise RuntimeError('could not measure companions')
    keep,slot=np.where((pairs['row']>=0)&(pairs['score']> -6.4))
    selected=pairs[keep,slot]
    double=np.zeros(len(keep),SEED)
    double['row'][:,0]=ids[keep];double['row'][:,1]=ids[selected['row']];double['row'][:,2]=-1
    double['score']=selected['score'];double['g']=selected['g']
    # A longer measured displacement predicts remote nights more precisely,
    # even when the two-source likelihood alone has a large trials penalty.
    # Preserve the high-contrast tail within each duration and flux interval.
    if len(double):
        a,b=keep,selected['row']
        span=np.abs(et[p['e'][a]]-et[p['e'][b]])
        mbar=.5*(p['m'][a]+p['m'][b])
        group=np.clip((np.log(np.maximum(.45,span))/.72+1.25).astype('i4'),0,9)*24
        group+=np.clip(((mbar-15.)*2.).astype('i4'),0,23)
        base=double['score']+.63*np.log1p(span)
        quant=np.clip(((base+11.)*4.).astype('i4'),0,127)
        h=np.bincount(group*128+quant,minlength=240*128).reshape(240,128)
        tail=h[:,::-1].cumsum(1)[:,::-1]
        contrast=-np.log((tail[group,quant]-.5*h[group,quant]+2.)/(h.sum(1)[group]+4.))
        priority=.73*base+.72*contrast
        double=unique(double,double['row'][:,:2],900000,priority)
    tri=tri[tri['score']>1.5];tri['row']=ids[tri['row']]
    tri=unique(tri,tri['row'],95000)
    out=np.zeros(len(raw),'f4')
    rawpart=ids<primary
    out[ids[rawpart]]=np.clip(score[rawpart],0.,12.)
    print('independent-night companions:',n,'sources,',len(double),'pairs,',len(tri),
          'three-night fits in',round(time.monotonic()-begin,1),'s',flush=True)
    return out,double,tri


def start(*args):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='night-companions')
    job=pool.submit(build,*args)
    def report(f):
        if f.exception():print('independent-night companions:',repr(f.exception())[:200],flush=True)
    job.add_done_callback(report);pool.shutdown(wait=False)
    return job
