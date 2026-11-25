"""Choose additional third visits from measured two-night displacements."""
import numpy as np
from scipy.spatial import cKDTree

import measured_triples
import night_forecasts as forecast


class PairVisits:
    def __init__(self, ephemerides, plans):
        self.orbits=ephemerides
        self.tried=set()
        self.scheduled={}
        for (a,b,c,f),_ in plans[:1800]:
            self.scheduled.setdefault((min(a,b),max(a,b),f),set()).add(c)

    def probability(self, sky, base, rows, states, visits, xy, mag):
        p=self.orbits
        locations=xy[rows,None,None,:]+sky[states[:,:,None],visits[None,None,:]]-base[states][:,:,None,:]
        dx=np.floor((locations-p.lower[visits])/270.).astype('i4')
        x,y=dx[:,:,:,0],dx[:,:,:,1]
        w,h=p.width[visits],p.height[visits]
        ok=(x>=0)&(x<w)&(y>=0)&(y<h)
        address=p.starts[visits]+np.clip(x,0,w-1)*h+np.clip(y,0,h-1)
        d=p.live_depth[address]-mag[rows,None,None]
        return ok/(1.+np.exp(np.clip(-(d+.10)/.34,-16.,16.)))

    def propose(self,join,model,f):
        m=model[0]
        a,b=map(int,m['view'][:2]);key=(min(a,b),max(a,b),int(f))
        if key in self.tried:return []
        self.tried.add(key)
        pairs=join.pairs(model,f)
        if len(pairs)<8:return []
        point=join.cat.points
        if len(pairs)>3800:
            roots=point[pairs['a']]
            use=measured_triples.balanced(pairs['merit']+.42*roots['quality'],roots,3800)
            pairs=pairs[use]
        pred=self.orbits
        sky,jitter,near,_=pred.population(a,f)
        visits=near[~np.isin(pred.night[near],pred.night[[a,b]])]
        usable=join.cat.fields[visits]
        visits=visits[(usable['columns']>0)|(usable['pad']>0)]
        if len(np.unique(pred.night[visits]))<3:return []
        z=pred.draws
        observed=sky[:,[a,b]].astype('f8')+jitter[[a,b]][None,:,None]*z[:,6:10].reshape(-1,2,2)
        base=observed[:,0].copy()
        mean=m['step'][0]['coef'][0];L=m['step'][0]['chol']
        def whiten(v,L):
            x=v[:,0]/L[0];return np.column_stack((x,(v[:,1]-L[1]*x)/L[2]))
        start=whiten(base-mean,L)
        X=forecast.features(start)
        displace=observed[:,1]-base-X@m['step'][1]['coef'][:X.shape[1]]
        coords=np.column_stack((start,whiten(displace,m['step'][1]['chol'])))
        root=point[pairs['a']]
        xy=np.column_stack((root['x'],root['y']))
        mag=.5*(root['mag']+point['mag'][pairs['b']])
        actual=np.column_stack((whiten(xy-mean,L),pairs['u'],pairs['v']))
        metric=np.array([.20,.20,1.,1.])
        _,states=cKDTree(coords*metric).query(actual*metric,k=3)
        sample=np.arange(len(pairs)) if len(pairs)<=96 else np.linspace(0,len(pairs)-1,96,dtype=int)
        weight=self.probability(sky,base,sample,states[sample],visits,xy,mag)
        # Each sampled orbit must see at least three more independent
        # nights. Several pointings on the same date count as one chance.
        nights=pred.night[visits]
        expected=np.zeros(weight.shape[:2])
        for n in np.unique(nights):
            expected+=weight[:,:,nights==n].max(axis=2)
        weight*=np.clip((expected-1.1)/2.,.015,1.)[:,:,None]
        # Estimate positional confusion from the residuals of the pair's
        # conditional ephemeris, including range and angular-rate error.
        X=forecast.features(coords)
        normals=X.T@X+np.diag(np.r_[np.full(5,.00004),np.full(10,.05)])
        dy=(sky[:,visits,:]-base[:,None,:]).reshape(len(base),-1)
        residual=(dy-X@np.linalg.solve(normals,X.T@dy)).reshape(len(base),-1,2).transpose(1,0,2)
        cov=residual.transpose(0,2,1)@residual/(len(base)-X.shape[1])
        cov[:,0,0]+=jitter[visits]**2;cov[:,1,1]+=jitter[visits]**2
        area=np.sqrt(np.maximum(1.e-8,np.linalg.det(cov)))
        chance=area*join.cat.fields['density'][visits]
        strength=weight.mean(axis=(0,1))/(1.+.45*chance)**.61
        previous=self.scheduled.setdefault(key,set())
        allowed=~np.isin(visits,list(previous|{int(m['view'][2])}))
        if not np.any(allowed):return []
        result=[]
        for _ in range(2):
            utility=np.where(allowed,strength,-1.)
            j=int(np.argmax(utility))
            if utility[j]<.016:break
            c=int(visits[j]);previous.add(c)
            # Conditional on reaching C, choose the other two nights along
            # the same trajectories. All actual peaks still get an orbit fit.
            joint=(weight*weight[:,:,j,None]).sum(axis=(0,1))
            ranked=np.argsort(-joint,kind='stable')
            rest=[];seen={int(nights[j])}
            for k in ranked:
                n=int(nights[k])
                if n in seen or joint[k]<=.005:continue
                rest.append(int(visits[k]));seen.add(n)
                if len(rest)>=forecast.RETURNS:break
            if len(rest)>=2:
                every=np.arange(len(pairs))
                live=self.probability(sky,base,every,states,np.array([c]),xy,mag)
                admissible=live.max(axis=(1,2))>.035
                if admissible.any():result.append((((a,b,c,int(f)),rest),pairs[admissible]))
            allowed &= nights!=nights[j]
            if not np.any(allowed):break
        return result
