"""Marginal evidence for an orbit, including alternative shorter movers."""
import ctypes
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from scipy.special import expit, logit, logsumexp, ndtr


# Mixtures in distance and local velocity relative to circular speed.
# The last component has a uniform density in log distance.
FAMILIES=np.array([[.36,44.,.105,.99,.105,.135,.15,.095],
                   [.47,42.,.30,.94,.245,.38,.34,.32],
                   [.17,90.,-1.,.35,.63,.65,.63,1.]])


@lru_cache(maxsize=16)
def assignments(n):
    rows=[];bits=[]
    for size in range(2,n+1):
        for batch in combinations(range(n),size):
            m=np.zeros(n);m[list(batch)]=1.
            rows.append(m);bits.append(sum(1<<i for i in batch))
    mask=np.stack(rows);bits=np.array(bits,'u4');count=mask.sum(1)
    short=np.flatnonzero(count<5)
    a,b=np.triu_indices(len(short),1);a,b=short[a],short[b]
    disjoint=(bits[a]&bits[b])==0
    return mask,bits,count,a[disjoint],b[disjoint]


@lru_cache(maxsize=2)
def quadrature(order):
    z,w=np.polynomial.hermite.hermgauss(order)
    x,y=np.meshgrid(z*np.sqrt(2.),z*np.sqrt(2.))
    return np.column_stack((x.ravel(),y.ravel())),np.log(np.outer(w,w).ravel()/np.pi)


class Ranging:
    def __init__(self,lib=None):
        if lib is None or not hasattr(lib,'ranging_nodes'):
            compiler=next((s for k in ('c++','g++','clang++') if (s:=shutil.which(k))),None)
            if not compiler:raise RuntimeError('no native orbit integrator')
            with tempfile.TemporaryDirectory(prefix='orbit_integral_') as build:
                path=Path(build)/'ranging.so'
                subprocess.run([compiler,'-O3','-std=c++17','-fPIC','-shared',
                                str(Path(__file__).with_name('ranging_integral.cpp')),'-o',str(path)],
                               check=True,capture_output=True,timeout=25)
                lib=ctypes.CDLL(str(path))
        self.fn=lib.ranging_nodes
        ptr,integer=ctypes.c_void_p,ctypes.c_int
        self.fn.argtypes=[ptr,integer,ptr,integer,ptr,ptr,ptr,integer,ptr,ptr];self.fn.restype=integer

    def integrals(self,F,item,mask,bits,order):
        G=F.G;ix=item['take'];e=item['e'][ix];state=np.ascontiguousarray(item['x'],'f8')
        if not np.isfinite(state).all() or not 7.e-6<state[2]<.09:return None
        sigma=np.hypot(.065,item['errors'][ix])
        data=np.ascontiguousarray(np.column_stack((item['xy'][ix],G['et_np'][e]-item['ref'],
                                                 G['ob_np_follow'][e],sigma)))
        J=item['design'][ix]/sigma[:,None,None]
        res=item['resxy'][ix]/sigma[:,None]
        info=(mask@(J.transpose(0,2,1)@J).reshape(len(ix),36)).reshape(-1,6,6)
        rhs=mask@np.einsum('nij,ni->nj',J,res)
        pole=G['frame_np']@np.array([0.,-np.sin(.40909280),np.cos(.40909280)])
        pole=np.ascontiguousarray(pole,'f8')
        direction=np.r_[state[:2],1.];norm=np.linalg.norm(direction);direction/=norm
        east=np.cross(pole,direction);east/=max(np.linalg.norm(east),1.e-12)
        basis=np.stack((east,np.cross(direction,east),direction))
        g=state[2];r=norm/g;om=np.sqrt(F.MU/r)*g
        vel=basis@state[3:]/om
        z,logw=quadrature(order);arr=[]
        for fam in FAMILIES:
            _,radius,spread,speed,along,north,radial,_=fam
            scale=np.array([spread if spread>0 else 1.45,along,north,radial])
            penalty=np.r_[np.log(r/radius) if spread>0 else 0.,vel-[speed,0.,0.]]/scale
            prior=np.zeros((4,6))
            prior[0,2]=-1.
            prior[1:,2]=-1.5*vel;prior[1:,3:]=basis*item['scale'][3:]/om
            prior/=scale[:,None]
            A=info+prior.T@prior+np.diag([1.e-12,1.e-12,1.e-9,1.e-9,1.e-9,1.e-9])
            inv=np.linalg.inv(A)
            dx=-np.einsum('nij,nj->ni',inv,rhs+penalty@prior)
            cols=np.array([2,5])
            m=state[cols]+item['scale'][cols]*dx[:,cols]
            cov=inv[:,cols[:,None],cols]*np.outer(item['scale'][cols],item['scale'][cols])
            # A small enlargement covers the nonlinear tails of a sparse
            # arc. The integration always divides by this proposal density.
            chol=np.linalg.cholesky(cov)*1.12
            gh=np.ascontiguousarray(m[:,None,:]+z@chol.transpose(0,2,1))
            values=np.empty((len(m),len(z)),'f8')
            status=self.fn(data.ctypes.data,len(e),bits.ctypes.data,len(bits),state.ctypes.data,
                           fam.ctypes.data,pole.ctypes.data,len(z),gh.ctypes.data,values.ctypes.data)
            if status:raise RuntimeError('orbit integral failed')
            log_area=np.log(2.*np.pi*chol[:,0,0]*chol[:,1,1])
            arr.append(log_area+logsumexp(values+logw+.5*np.square(z).sum(1),axis=1))
        return logsumexp(arr,axis=0)


def photometry(item,mask,lever):
    ix=item['take'];sn=item['snr'][ix]
    obs=item['model_mag'][ix].astype(float)
    good=np.isfinite(obs)
    if good.sum()<2:return None
    # Truncation biases the magnitude of near-threshold measurements.
    # Use their local flux noise and a prior favouring more faint sources.
    center=np.median(obs[good])
    y=np.where(good,obs-center+1.3/np.maximum(sn,4.5)**2,0.)
    # Both slope and red-band curvature are shared by an object's nights.
    X=np.stack((np.ones(len(ix)),.27*lever,.105*np.minimum(lever,0.)),axis=1)
    components=[]
    for scatter,weight in ((.065,.46),(.165,.40),(.36,.14)):
        var=scatter**2+(1.086/np.maximum(sn,4.5))**2
        w=good/var
        normal=(mask@((X[:,:,None]*X[:,None,:])*w[:,None,None]).reshape(len(ix),9)).reshape(-1,3,3)
        normal+=np.diag([1.e-9,1.,1.])
        b=mask@(X*(w*y)[:,None])
        param=np.linalg.solve(normal,b[:,:,None])[:,:,0]
        chi=np.maximum(0.,mask@(w*y*y)-(b*param).sum(1))
        logz=-.5*(mask@(good*np.log(2.*np.pi*var))+np.linalg.slogdet(normal)[1]+chi)
        # One intercept has a flat prior per magnitude. The other two
        # parameters have normalized standard normal priors.
        components.append(logz+.5*np.log(2.*np.pi)+np.log(weight))
    z=logsumexp(components,axis=0)
    z[mask@good==0]=0.
    return z,center


def row_marks(G,item):
    d=item['d'][item['take']]
    out=np.zeros(len(d))
    for key,weight in (('shape_marks',.67),('stationary_marks',.68)):
        saved=G.get(key)
        if saved is None:continue
        rows,value=saved
        if not len(rows):continue
        j=np.searchsorted(rows,d).clip(0,len(rows)-1);valid=rows[j]==d
        out[valid]+=weight*value[j[valid]]
    return np.clip(out,-4.,2.5)


def assess(engine,F,item,order=3,select_members=False):
    n=len(item['take'])
    if not 5<=n<=8:return None
    mask,bits,size,left,right=assignments(n)
    ix=item['take'];e=item['e'][ix]
    phot=photometry(item,mask,F.G['lev_e'][e])
    if phot is None:return None
    astro=engine.integrals(F,item,mask,bits,order)
    if astro is None:return None
    marks=row_marks(F.G,item)
    density=item.get('null_density',np.full(n,.0007))
    logz=astro+phot[0]+mask@(marks-np.log(np.maximum(1.e-8,density)))
    logz+=.47*np.log(10.)*np.clip(phot[1]-23.,-7.,5.)-2.*np.log(3600.)
    # The alternative can contain a two-night arc. A remaining row may
    # come from another mover, or from the measured field background.
    d=F.G.get('m5c_np')
    mr=item['model_mag'][ix]
    pivot=phot[1]+item['color']*F.G['lev_e'][e]
    if d is None:pred=item['snr'][ix]*10.**np.clip(.4*(mr-pivot),-1.,1.)
    else:pred=5.*10.**np.clip(.4*(d[e]-pivot),-1.5,1.5)
    pd=ndtr((np.nan_to_num(pred,nan=5.)-4.5)/np.hypot(1.1,.18*np.nan_to_num(pred,nan=5.)))
    # Blending, chip holes and rotations can hide an otherwise detectable
    # mover. They should affect omitted rows once per proposed parent.
    logz+=(1.-mask)@np.log(.30+.59*(1.-pd))
    eligible=size>=5
    if np.max(astro[eligible])< -1.e12:return None
    split=logsumexp(np.r_[0.,logz[left]+logz[right]])
    def evidence(good):
        return float(logsumexp(logz[good])-np.logaddexp(split,logsumexp(logz[~good])))
    good=eligible&(size>=.8*n)
    odds=evidence(good)
    if not np.isfinite(odds):return None
    if not select_members:return odds
    full_odds=odds;kept=np.ones(n,bool)
    if n>=6:
        # Conditional on the presence of a five-night orbit, how often does
        # each night belong to it? Keep uncertain observations unless a
        # shorter membership explains the original measurements decisively.
        mass=np.exp(logz[eligible]-logsumexp(logz[eligible]))
        inclusion=mass@mask[eligible]
        proposed=inclusion>.105
        if 5<=proposed.sum()<n and np.min(inclusion[proposed])>.70:
            shared=mask@proposed
            alternate=evidence((shared>=5)&(shared>=.8*proposed.sum()))
            if alternate>-.4 and alternate>=odds:
                kept=proposed;odds=alternate
    return odds,kept,full_odds


def update_members(F,item,c,kept):
    import image_followup
    nights=F.G['night_e_np'][item['e']]
    retained=np.isin(nights,nights[item['take'][kept]])&np.isin(item['d'],c['det'])
    fit={k:item[k][retained] for k in ('d','e','xy','snr','mag')}
    fit.update(candidate={'_orbit':(item['x'],item['ref'])},ref=item['ref'])
    image_followup.model(F,[fit],refit=True)
    x=fit['x'];g=x[2]
    if not 1./150000.<g<1./12.:return False
    radius=np.sqrt(1.+x[:2]@x[:2])/g
    if not np.isfinite(fit['res']).all() or np.max(fit['res'])>1.04:return False
    if radius*(x[3:]@x[3:])>2.02*F.MU*g*g:return False
    times=F.G['et_np'][fit['e']]
    c.update(det=fit['d'].tolist(),nights=int(kept.sum()),r=round(float(radius),2),
             rms=round(float(np.sqrt(np.mean(fit['res']**2))),3),arc=round(float(np.ptp(times)),2),
             removed_nights=int((~kept).sum()))
    return True


def apply(F,items,deadline):
    lib=getattr((F.G.get('survey_control') or {}).get('certifier'),'lib',None)
    engine=Ranging(lib)
    count=0
    for item in sorted(items,key=lambda x:-x['candidate']['_report']['p']):
        if time.monotonic()+.12>=deadline:break
        c=item['candidate']['_report']
        if c['p']<.0005:break
        try:result=assess(engine,F,item,select_members=True)
        except (ValueError,np.linalg.LinAlgError):continue
        if result is None:continue
        value,kept,full=result
        if not np.all(kept):
            try:valid=update_members(F,item,c,kept)
            except (ValueError,np.linalg.LinAlgError):valid=False
            if not valid:value=full
        c['ranging_log_bf']=round(value,3);count+=1
        independent=np.clip(c.get('unused_visit_odds',0.),-8.,8.)*.54
        repeat=max(.63*c.get('visit_repeat',0.),min(3.4,1.4*c.get('h3m',0.)))
        other=independent+repeat-.7*c.get('image_misses',0.)+.50*c.get('sensor_log_odds',0.)
        z=logit(np.clip(c['p'],1.e-14,1.-1.e-14))
        c['p']=float(expit(z+np.clip(.48*(value+other-z),-6.,6.)))
    F.log('integrated sparse orbit evidence:',count,'reports')
