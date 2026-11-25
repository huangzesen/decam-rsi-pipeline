"""Use the measured first displacement to locate two more raw measurements."""
import copy
import ctypes
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import quartet_geometry as geo
from attributables import SEED
from quartet_footprints import routes

PAIR=np.dtype([('pos','f4',4),('mag','f4'),('quality','f4'),('a','i4'),('b','i4')])
PROBE=np.dtype([('dx','f8'),('dy','f8'),('scale','f8'),('e','i4'),('pad','i4')])
MODEL=np.dtype([('center','f8',4),('transform','f8',(4,4)),('poly','f8',(15,4)),
                ('first','f8',3),('slope','f8',(2,2)),('last','f8',3)])


def condition(samples):
    first=geo.pair_vector(samples[:,:,:2]);dest=samples[:,:,2:].reshape(len(samples),-1,4)
    z,center,T=geo.whitening(first)
    quad=geo.squared(z)
    feature=np.concatenate((z,quad),axis=-1)
    average=feature.mean(1);feature-=average[:,None,:]
    mean=dest.mean(1)
    ridge=np.diag([1.e-8]*4+[.009]*10)
    slope=np.linalg.solve(geo.covar(feature,feature)+ridge,geo.covar(feature,dest-mean[:,None,:]))
    output=np.empty(len(samples),MODEL)
    output['center']=center;output['transform']=T
    output['poly'][:,0]=mean-(average[:,None]@slope)[:,0]
    output['poly'][:,1:]=slope
    residual=dest-mean[:,None,:]-feature@slope
    cov=1.35*geo.covar(residual,residual)+.19**2*np.eye(4)
    a=cov[:,:2,:2];b=cov[:,:2,2:]
    conditional=np.linalg.solve(a,b)
    v=cov[:,2:,2:]-b.transpose(0,2,1)@conditional+.15**2*np.eye(2)
    output['first']=a[:,(0,0,1),(0,1,1)]
    output['slope']=conditional
    output['last']=v[:,(0,0,1),(0,1,1)]
    return output


def shutters(index):
    fields=index.fields;probes=[]
    for group in fields.member:
        # Use independent pointings in a dither sequence. Very long
        # sequences get a representative spread of the deepest shutters.
        by_depth=group[np.argsort(-fields.raw['depth'][group],kind='stable')]
        selected=by_depth[:4]
        p=np.zeros(len(selected),PROBE);p['e']=selected
        p['dx'],p['dy']=(fields.offset[selected]/geo.AS).T
        p['scale']=1.+fields.shear[selected]
        probes.append(p)
    return probes


def pack_pairs(pairs,rows):
    out=np.empty(len(pairs),PAIR)
    out['pos'],out['mag'],out['quality']=pairs['pos'],pairs['mag'],pairs['quality']
    out['a'],out['b']=rows[pairs['a']],rows[pairs['b']]
    return out


class Search:
    def __init__(self,scanner,sky,rowmap):
        # The image columns are immutable. Each search owns its pair cache.
        self.index=copy.copy(scanner)
        self.index.cache=OrderedDict();self.index.cache_bytes=0
        self.sky=sky;self.rows=rowmap;self.probes=shutters(scanner)
        self.fn=sky.lib.conditional_chord
        p,i,d=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
        self.fn.argtypes=[p,p,i,p,p,i,p,i,d,d,d,i,p,p];self.fn.restype=i
        self.gpu=None
        if scanner.gpu is not None:
            try:
                from raw_chord_device import Device
                self.gpu=Device(sky,scanner.actual_visits)
                print('raw displacement CUDA fixture verified',flush=True)
            except Exception as exc:
                print('raw catalogue host lookup:',str(exc)[:150],flush=True)

    def measure(self,pairs,model,third,fourth,lever,gamma,seconds,cap):
        if self.gpu is not None:
            try:
                return self.gpu.measure(pairs,model,third,fourth,lever,gamma,cap)
            except (RuntimeError,ValueError,MemoryError) as exc:
                print('continuing raw lookups on host:',str(exc)[:160],flush=True)
                self.gpu=None
        out=np.empty(cap,SEED);stat=np.zeros(4,'i8')
        n=self.fn(self.sky.handle,pairs.ctypes.data,len(pairs),model.ctypes.data,
                  third.ctypes.data,len(third),fourth.ctypes.data,len(fourth),lever,gamma,
                  seconds,cap,out.ctypes.data,stat.ctypes.data)
        if n<0:raise RuntimeError('conditional raw chord search failed')
        return out[:n],stat

    def scan(self,plans,pole,end,event):
        z=geo.quadrature(128);index=self.index;v=index.visits;rng=np.random.default_rng(4991)
        total=np.zeros(5,'i8')
        for off in range(0,len(plans),16):
            if time.monotonic()>=end or event.is_set():break
            batch=plans[off:off+16].copy()
            # Either observed displacement can be the conditioned pair.
            flip=rng.random(len(batch))<.5
            batch[flip,:4]=batch[flip][:,[2,3,0,1]]
            sample=index.fields.uncertainty(geo.sample(v,index.center,pole,batch,z),batch,z)
            models=condition(sample)
            gate=geo.pair_gate(sample[:,:,:2]);foot=routes(sample)[:,0]
            found=[]
            for k,(a,b,c,d,fam) in enumerate(batch):
                if time.monotonic()>=end or event.is_set():break
                initial=index.pairs(a,b,fam,gate[k],7500)
                selected=index.footprints.select(initial,foot[k],(c,d),6000 if self.gpu is not None else 1900)
                if not len(selected):continue
                # Allow sparse populations of ordinary quality and favour
                # the best pairs in fields with many chance associations.
                order=np.argsort(-selected['quality']+.45*rng.gumbel(size=len(selected)),kind='stable')
                pairs=pack_pairs(selected[order],self.rows)
                gamma=np.sqrt(1.+index.center[a]@index.center[a])/geo.FAMILIES[fam,0]
                lever=.5*(v['lever'][a]+v['lever'][b])
                seeds,stat=self.measure(pairs,models[k:k+1],self.probes[c],self.probes[d],lever,gamma,.095,
                                        290 if self.gpu is not None else 130)
                total+=np.r_[1,stat]
                if len(seeds):found.append(seeds)
            if found:yield np.concatenate(found),total.copy()


def work(scanner,sky,rowmap,plans,pole,end,control):
    from orbit_quartets import predicted_cores
    search=Search(scanner,sky,rowmap);last=time.monotonic();bank=[];size=0
    counts=np.zeros(5,'i8')
    def publish():
        nonlocal bank,size,last
        seeds=np.concatenate(bank)
        order=np.argsort(-seeds['score'],kind='stable')
        _,first=np.unique(seeds['rows'][order],axis=0,return_index=True)
        seeds=seeds[order[np.sort(first)]]
        cores,tested=predicted_cores(sky,seeds,None,min(3.4,max(.3,end-time.monotonic())))
        if len(cores):
            cores['merit']=np.clip(.27*cores['merit'],-2.,5.)
            result,checked=sky.recover(cores,min(3.2,max(.3,end-time.monotonic())),cap=1600,measured=True)
            control['raw_results'].put((result,[]))
            print('single displacement raw returns',counts.tolist(),'fifth-night predictions',tested,
                  'verified',len(result),'from',checked,flush=True)
        bank=[];size=0;last=time.monotonic()
    for seeds,counts in search.scan(plans,pole,end-5.,control['event']):
        bank.append(seeds);size+=len(seeds)
        if size>=5600 or time.monotonic()-last>22.:publish()
    if bank and not control['event'].is_set():publish()


def begin(*args):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='conditional-measurement')
    job=pool.submit(work,*args);pool.shutdown(wait=False)
    def done(f):
        if f.exception() is not None:print('conditional raw search:',repr(f.exception())[:180],flush=True)
    job.add_done_callback(done)
    return job
