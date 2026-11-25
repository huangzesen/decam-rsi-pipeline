"""Conditional return positions for the camera fields reachable by measured arcs."""
from collections import OrderedDict

import numpy as np

import night_forecasts as F
import quartet_geometry as orbit

VIEWS=48
RETURN=np.dtype([('count','i4'),('views','i4',VIEWS),('step',F.STEP,VIEWS),
                 ('corr','f4',(VIEWS,VIEWS,4)),('logrange','f4',F.FEATURES),
                 ('crossrange','f4',(VIEWS,2))])

def grouped_apparent(pos,vel,dt,group,offset,observer):
    """Kepler states once per night with short-interval transport to each visit."""
    r=np.linalg.norm(pos,axis=1)[:,None]
    alpha=2./r-np.sum(vel*vel,axis=1)[:,None]/orbit.MU
    motion=np.sqrt(orbit.MU*alpha**3)
    s0=np.sum(pos*vel,axis=1)[:,None]*np.sqrt(alpha/orbit.MU)
    c0=1.-r*alpha
    mean=motion*dt
    anomaly=mean.copy()
    for _ in range(5):
        s,c=np.sin(anomaly),np.cos(anomaly)
        anomaly-=(anomaly-c0*s+s0*(1.-c)-mean)/(1.-c0*c+s0*s)
    s=np.sin(anomaly);u=2.*np.sin(.5*anomaly)**2
    x=(1.-u/(r*alpha))[:,:,None]*pos[:,None]+(dt+(s-anomaly)/motion)[:,:,None]*vel[:,None]
    rn=np.linalg.norm(x,axis=2)
    fd=-np.sqrt(orbit.MU/alpha)*s/(rn*r)
    gd=1.-u/(rn*alpha)
    velocity=fd[:,:,None]*pos[:,None]+gd[:,:,None]*vel[:,None]
    gravity=-.5*orbit.MU/rn**3
    # All offsets are within this individual observing night. Even at the
    # inner edge of the prior the neglected jerk is below a milliarcsecond.
    at=x[:,group,:]*(1.+gravity[:,group]*offset**2)[:,:,None]
    at+=velocity[:,group,:]*offset[None,:,None]
    at-=observer
    return at[:,:,:2]/at[:,:,2,None]


class Ephemerides:
    def __init__(self,cat,pole,draws):
        self.cat,self.pole,self.draws=cat,pole,draws
        self.stored=OrderedDict()
        v=cat.nights.visits
        self.ob=np.column_stack([v[k] for k in ('ox','oy','oz')])
        self.time=v['t']
        self.night=v['night']
        self.depth=v['depth']
        self.maps=cat.visibility
        origin,maps=self.maps
        self.lower=origin+270.*np.array([(x,y) for x,y,g in maps])
        self.upper=self.lower+270.*np.array([g.shape for x,y,g in maps])
        dims=np.array([g.shape for x,y,g in maps])
        self.width,self.height=dims.T
        self.starts=np.r_[0,np.cumsum(self.width*self.height)[:-1]]
        self.live_depth=np.concatenate([g.ravel() for x,y,g in maps])
        self.covered=np.array([np.any(g>-79.) for x,y,g in maps])
        _,self.night_index=np.unique(self.night,return_inverse=True)
        self.epochs=np.bincount(self.night_index,weights=self.time)/np.bincount(self.night_index)
        self.offset=self.time-self.epochs[self.night_index]

    def population(self,a,f):
        key=(a,int(f))
        if key in self.stored:
            self.stored.move_to_end(key)
            return self.stored[key]
        z=self.draws
        extent=getattr(self.cat,'extent',None)
        width=np.deg2rad(.66) if extent is None else extent[a]
        xy=self.cat.center[a]+width*z[:,:2]
        state=orbit.directions(xy,*orbit.variation(z,f),self.ob[a],self.pole)
        sky=grouped_apparent(*state,self.epochs-self.time[a],self.night_index,self.offset,self.ob)/orbit.AS
        jitter=np.hypot(.23,orbit.TRANSPORT_RATE[f]*
                        self.cat.nights.spread*7.7)
        # Bound the reachable camera list before building its conditional
        # ephemerides. Several neighbouring CCDs of slack cover sparse draws.
        margin=1800.
        eligible=np.all((sky.min(0)<self.upper+margin)&(sky.max(0)>self.lower-margin),axis=1)
        eligible &= self.covered
        loggamma=-np.log(state[0][:,2])
        self.stored[key]=(sky.astype('f4'),jitter,np.flatnonzero(eligible),loggamma)
        while len(self.stored)>1 and sum(v[0].nbytes for v in self.stored.values())>175000000:
            self.stored.popitem(last=False)
        return self.stored[key]

    def select(self,available,base,states,measured,prefix):
        # Score each camera footprint at the directions reached by measured
        # triples. A nearby simulated trajectory supplies only this coarse
        # footprint test. The astrometric search uses the full regression.
        if len(available)<=VIEWS:
            return available
        xy,mag,weight=measured
        loc=np.floor((xy[:,None,:]+states[:,available,:]-base[:,None,:]-self.lower[available])/270.).astype('i4')
        x,y=loc[:,:,0],loc[:,:,1]
        w,h=self.width[available],self.height[available]
        inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        address=self.starts[available]+np.clip(x,0,w-1)*h+np.clip(y,0,h-1)
        d=self.live_depth[address]-mag[:,None]
        prob=inside/(1.+np.exp(np.clip(-d/.38,-16.,16.)))
        # Promote complements that reach different parts of the starting
        # image, retaining a small contribution from the original simulated
        # path votes when the three-point orbit is particularly uncertain.
        original=np.isin(available,prefix)
        floor=.08*original+.002
        active=weight.copy();selected=[]
        peak=np.ones(len(available))
        boosted=prob+.15*(prob>.25)
        for _ in range(min(VIEWS//4,len(available)//4)):
            score=(active@boosted+floor)*peak
            chosen=np.argpartition(-score,3)[:4]
            selected.extend(chosen)
            active*=.68+.32*np.exp(-prob[:,chosen].sum(1))
            active+=.045*weight
            relevant=np.isin(self.night[available],self.night[available[chosen]])
            peak[relevant]*=.79
            peak[chosen]=0.
        return available[np.asarray(selected)]

    def build(self,model,f,triples,points=None):
        out=np.zeros(1,RETURN)
        if not len(triples):return out
        m=model[0]
        a,b,c=map(int,m['view'][:3])
        sky,jitter,near,loggamma=self.population(a,f)
        available=near[~np.isin(self.night[near],self.night[[a,b,c]])]
        if len(available)<2:return out
        z=self.draws
        init=sky[:,[a,b,c]].astype(float)+jitter[[a,b,c]][None,:,None]*z[:,6:12].reshape(-1,3,2)
        base=init[:,0].copy()
        obs=init.copy();obs[:,1:]-=base[:,None]
        measured=triples if len(triples)<=80 else triples[np.linspace(0,len(triples)-1,80,dtype='i4')]
        p=self.cat.points if points is None else points
        positions=np.stack([np.column_stack((p['x'][measured[k]],p['y'][measured[k]])) for k in ('a','b','c')],axis=1)
        origin=positions[:,0].copy()
        positions[:,1:]-=origin[:,None]
        def coordinates(raw):
            previous=np.empty((len(raw),0))
            for j in range(3):
                X=F.features(previous)
                step=m['step'][j]
                r=raw[:,j]-X@step['coef'][:X.shape[1]]
                lx,ly,lz=step['chol']
                u=r[:,0]/lx;v=(r[:,1]-ly*u)/lz
                previous=np.column_stack((previous,u,v))
            return previous
        coords=coordinates(obs)
        if len(available)>VIEWS:
            query=coordinates(positions)
            # Sky location is transported explicitly, leaving mostly the
            # four measured displacement components in this distance.
            metric=np.array([.15,.15,1.,1.,.80,.80])
            d=((query[:,None,:]-coords[None,:,:])*metric)**2
            where=np.argpartition(d.sum(2),1,axis=1)[:,:2].ravel()
            brightness=np.mean([p['mag'][measured[k]] for k in ('a','b','c')],axis=0)
            weight=np.ones(len(where))/len(where)
            available=self.select(available,base[where],sky[where],
                                  (np.repeat(origin,2,axis=0),np.repeat(brightness,2),weight),
                                  m['view'][3:3+m['count']])
        n=len(available);out['count']=n;out['views'][0,:n]=available
        X=F.features(coords)
        xt=X.T
        normal=xt@X+np.diag(np.r_[np.full(7,.00002),np.full(21,.055)])
        response=(sky[:,available].astype(float)-base[:,None,:]).reshape(len(base),-1)
        beta=np.linalg.solve(normal,xt@response)
        coeff=np.linalg.solve(normal,xt@(loggamma-np.log(float(m['gamma']))))
        out['logrange'][0]=coeff
        residual_range=loggamma-np.log(float(m['gamma']))-X@coeff
        res=(response-X@beta).reshape(len(base),n,2).transpose(1,0,2)
        den=len(base)-X.shape[1]
        cov=res.transpose(0,2,1)@res/den
        noise=.075**2+jitter[available]**2
        cov[:,0,0]+=noise;cov[:,1,1]+=noise
        L=np.linalg.cholesky(cov)
        out['step']['coef'][0,:n]=beta.reshape(-1,n,2).transpose(1,0,2)
        out['step']['chol'][0,:n]=L[:,[0,1,1],[0,0,1]]
        white=np.linalg.solve(L,res.transpose(0,2,1)).reshape(2*n,len(base))
        out['crossrange'][0,:n]=(white@residual_range/den).reshape(n,2)
        corr=(white@white.T).reshape(n,2,n,2).transpose(0,2,1,3)
        out['corr'][0,:n,:n]=(.998/den*corr).reshape(n,n,4)
        return out
