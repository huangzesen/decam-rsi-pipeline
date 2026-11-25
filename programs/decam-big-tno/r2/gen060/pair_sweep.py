"""Join measured arcs at a common epoch after accounting for solar gravity."""
import ctypes
import time

import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import qmc

from survey_arcs import CERTIFIED

ARC=206264.80624709636
MU=.0002959122082855911*1.0013418
QUAD=np.dtype([('rows','i4',4),('g','f4'),('score','f4')])


class Sweep:
    def __init__(self,index,pairs,ex,east,device):
        self.device=device
        ids=np.asarray(pairs['row'][:,:2],dtype='i4')
        span=np.abs(np.diff(ex['t'][index.target['exposure'][ids]],axis=1)[:,0])
        # Very short arcs have high extrapolation errors. Keep a sample
        # across measured spans and brightnesses when limiting the catalogue.
        eligible=(span>.60)&np.isfinite(pairs['g'])&(pairs['g']>0.)
        self.ids=np.ascontiguousarray(ids[eligible])
        self.score=pairs['score'][eligible]
        t=lambda v:torch.as_tensor(np.require(v,requirements=['C','W']),device=device)
        points=index.target[self.ids];e=points['exposure']
        self.xy=t(np.stack((points['x'],points['y']),axis=-1).astype('f8'))
        self.t=t(ex['t'][e]-np.median(ex['t']))
        self.dt=self.t[:,1]-self.t[:,0]
        self.epoch=self.t.mean(1)
        self.mag=t(points['mag'].mean(1).astype('f8'))
        self.ob=t(np.stack([ex[k][e] for k in ('ox','oy','oz')],axis=-1))
        self.east=t(np.asarray(east,'f8'));self.north=torch.stack((-self.east[1],self.east[0]))
        self.xymean=self.xy.mean(1)
        self.approx=t(pairs['g'][eligible].astype('f8'))
        self.par=self.ob[:,:,:2]-self.xy*self.ob[:,:,2:3]
        self.dv=((self.xy[:,1]-self.xy[:,0])/self.dt[:,None]).contiguous()
        self.dp=((self.par[:,1]-self.par[:,0])/self.dt[:,None]).contiguous()
        self.norm=(1.+self.xymean.square().sum(1)).sqrt()
        self.max_time=float(np.quantile(np.abs(ex['t']-np.median(ex['t'])),.86))

    def matches(self,radius,radial,phase):
        # The prior bound is widened by radial displacement since the
        # reference epoch. The hash itself uses the actual pair astrometry.
        circ=np.sqrt(MU/radius**3)
        guess=self.norm/radius
        here=guess/(1.+self.epoch*circ*radial)
        velocity=self.dv+here[:,None]*self.dp
        tolerance=(.72/ARC)/self.dt.abs()+.15*circ
        speed=velocity.square().sum(1).sqrt()
        ok=(speed<1.55*circ*self.norm+tolerance)
        ix=torch.nonzero(ok,as_tuple=True)[0]
        if len(ix)<2:return np.empty((0,2),'i4')
        t=self.t[ix];xy=self.xy[ix];ob=self.ob[ix];g=guess[ix]
        v=velocity[ix]
        q=self.xymean[ix]+here[ix,None]*self.par[ix].mean(1)-self.epoch[ix,None]*v
        w2=MU/radius**3;u=radial*circ
        for _ in range(2):
            norm2=1.+q.square().sum(1);g=norm2.sqrt()/radius
            h=norm2*u-(q*v).sum(1)
            delay=(radius/173.1446326846693)*(1.+u*t)
            dt=t-delay
            z=w2*dt.square();z1=u*dt
            vsq=(v.square().sum(1)+h.square())/norm2
            f=1.-.5*z*(1.-z1)+z*((3.*vsq[:,None]-2.*w2)*dt.square()-15.*z1.square())/24.
            s=dt*(1.-z/6.+z*z1/4.+z*((9.*vsq[:,None]-8.*w2)*dt.square()-45.*z1.square())/120.)
            tau=s/f;gf=g[:,None]/f
            corrected=xy*(1.+tau*h[:,None]-gf*ob[:,:,2])[:,:,None]+gf[:,:,None]*ob[:,:,:2]
            v=(corrected[:,1]-corrected[:,0])/(tau[:,1]-tau[:,0])[:,None]
            q=corrected[:,0]-v*tau[:,0,None]
        # At opposition range errors mostly move an inferred rate along
        # the ecliptic. The less uncertain perpendicular velocity remains
        # discriminating even when a short arc is extrapolated several years.
        spread=(.66+.92*phase[0])
        along=spread*(.64*43./radius+.10)
        across=spread*(.23*43./radius+.09)
        timewidth=max(350.,min(2550.,self.max_time))
        params=[(q@self.east)*ARC/(along*timewidth+47.),
                (q@self.north)*ARC/(across*timewidth+25.),
                (v@self.east)*ARC/along,(v@self.north)*ARC/across,
                self.mag[ix]/(.65+.24*phase[1])]
        key=torch.zeros_like(ix)
        for col,shift,bits in zip(params,phase[2:7],(16,16,11,11,7)):
            key=(key<<bits)|(torch.floor(col+float(shift)).long()&((1<<bits)-1))
        order=torch.argsort(key,stable=True);key=key[order];i=ix[order]
        pairs=[]
        for stride in (1,2,3,4,6,9,14):
            if len(key)<=stride:break
            same=key[:-stride]==key[stride:]
            j=torch.nonzero(same,as_tuple=True)[0]
            a,b=i[j],i[j+stride]
            distinct=(self.epoch[a]-self.epoch[b]).abs()>70.
            if len(a):pairs.append(torch.stack((a[distinct],b[distinct]),1))
        if not pairs:return np.empty((0,2),'i4')
        return torch.cat(pairs).int().cpu().numpy()


def verify(index,proposals,threads,seconds):
    ptr,integer=ctypes.c_void_p,ctypes.c_int
    fn=index.lib.verify_companion_quads
    fn.argtypes=[ptr,ptr,integer,integer,ctypes.c_double,integer,ptr,ptr];fn.restype=integer
    out=np.empty(5200,CERTIFIED);stats=np.zeros(4,'i8')
    n=fn(index.handle,proposals.ctypes.data,len(proposals),threads,seconds,len(out),out.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('common-epoch verification failed')
    return out[:n].copy(),stats


def run(index,pairs,ex,east,control,threads,seconds,device):
    end=time.monotonic()+seconds
    samples=qmc.Sobol(9,scramble=True,seed=1811).random_base2(9)
    seen=set();pending=[];count=0;submitted=0;verified=0
    if device=='cuda' and torch.cuda.mem_get_info()[0]<2.5e9:device='cpu'
    sweep=Sweep(index,pairs,ex,east,device)
    for j,u in enumerate(samples):
        left=end-time.monotonic()
        if left<5. or control['event'].is_set():break
        if j%7==5:radius=72.*(3200./72.)**u[7]
        elif j%5==3:radius=26.*(74./26.)**u[7]
        else:radius=43.5*np.exp(.117*ndtri(np.clip(u[7],.003,.997)))
        radial=np.clip(.27*ndtri(np.clip(u[8],.001,.999)),-.72,.72)
        hits=sweep.matches(radius,radial,u[:7]);count+=1
        if len(hits):
            # Collision probability varies with the template. Only send
            # distinct quartets from different dates into an orbit solve.
            a,b=hits.T;keys=(np.minimum(a,b).astype('i8')<<32)|np.maximum(a,b)
            keep=np.fromiter((int(k) not in seen for k in keys),bool,count=len(keys))
            rows=np.column_stack((sweep.ids[a[keep]],sweep.ids[b[keep]]))
            if len(rows):
                keys=keys[keep];score=sweep.score[a[keep]]+sweep.score[b[keep]]
                if len(rows)>42000:
                    use=np.argpartition(-score,41999)[:42000]
                    keys,rows,score=keys[use],rows[use],score[use]
                if len(seen)<2200000:seen.update(map(int,keys))
                part=np.empty(len(rows),QUAD);part['rows']=rows;part['score']=score;part['g']=1./radius
                pending.append(part)
        if pending and (len(pending)>=5 or left<12. or sum(map(len,pending))>=60000):
            proposals=np.concatenate(pending);pending.clear()
            found,stats=verify(index,proposals,threads,max(1.,min(12.,.22*left)))
            submitted+=len(proposals);verified+=int(stats[-1])
            control['certified'].put(found[found['nights']>=5])
    print('common-epoch companion pairs:',device,count,'distance and speed trials,',submitted,
          'distinct quartets,',verified,'five-night orbits',flush=True)
