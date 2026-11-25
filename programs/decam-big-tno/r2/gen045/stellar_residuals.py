"""Relative CCD registration and nonlinear centroid response from repeat stars."""
import numpy as np
from scipy.ndimage import gaussian_filter

from reference_field import AS


class ResidualField:
    def __init__(self, ne, nc):
        self.nc=nc
        self.scale=1.
        self.affine=np.zeros((ne,nc,3,2),'f4')
        self.phase=np.zeros((4,5,8,2),'f4')
        self.gain=np.zeros((ne,nc,2),'f4')

    @staticmethod
    def coordinates(x,y):
        return np.column_stack((np.ones(len(x)),x/1024.-1.,y/2048.-1.))

    @staticmethod
    def marks(sharp,snr,ph):
        shape=np.digitize((sharp-1.06)/np.hypot(.17,3.5/np.maximum(snr,4.5)),[-1.,-.25,.40,1.18])
        phase=np.minimum(7,np.floor(np.abs(ph)/.080).astype('i4'))
        return shape,phase

    def predict(self,ex,ccd,x,y,sn,b,sharp,ph):
        shape,p=self.marks(sharp,sn,ph)
        d=self.affine[ex,ccd]
        corr=d[:,0]+d[:,1]*(x/1024.-1.)[:,None]+d[:,2]*(y/2048.-1.)[:,None]
        for k in (0,1):
            corr[:,k]+=np.sign(ph[:,k])*self.phase[b,shape,p[:,k],k]
        corr+=self.gain[ex,ccd]*ph
        return self.scale*np.clip(corr,-.8,.8)


def fit(pixel,matched,trees,reference_rows,et,sharp,signal_bin):
    a=pixel.data
    store=ResidualField(len(et),pixel.chips)
    cr=[]
    for tree,rows in zip(trees,reference_rows):
        pos=tree.data.copy()*AS
        pixel.correct(pos,rows)
        cr.append(pos/AS)
    samples=[];available=0
    for e,side,rows,xy,indices in matched:
        valid=indices<len(cr[side]);k=np.minimum(indices,len(cr[side])-1)
        refrows=reference_rows[side][k]
        times=(et[a['exp'][refrows]]-et[e])/365.25
        target=xy.copy()*AS
        pixel.correct(target,rows)
        delta=cr[side][k]-target[:,None,:]/AS
        # Use different nights to estimate a reference position. A weak
        # stellar proper-motion prior bounds extrapolations in time.
        weight=valid/(1.+(np.linalg.norm(delta,axis=2)/.38)**2)
        rn=np.floor(et[a['exp'][refrows]]-.5)
        rep=(rn[:,:,None]==rn[:,None,:])&valid[:,None,:]
        weight/=np.maximum(1.,rep.sum(2))
        sw=weight.sum(1);st=(weight*times).sum(1);stt=(weight*times**2).sum(1)+18.
        wn=weight*(stt[:,None]-st[:,None]*times)
        offset=(wn[:,:,None]*delta).sum(1)/np.maximum(.00001,sw*stt-st*st)[:,None]
        sn=a['snr'][rows];b=signal_bin(sn);chip=a['ccd'][rows]
        jac=pixel.table[e,chip,b,2:6].reshape(-1,2,2)/AS
        good=(sw>.18)&(np.max(np.abs(offset),axis=1)<.54)&(np.abs(np.linalg.det(jac))>.025)
        if not good.any():continue
        pix=np.einsum('ni,nij->nj',offset[good],np.linalg.inv(jac[good]))
        r=rows[good];x=a['x'][r];y=a['y'][r];c=chip[good];b=b[good]
        phase=np.column_stack((x-np.rint(x),y-np.rint(y)))
        phase=phase+np.clip(phase*pixel.table[e,c,b,:2],-.60,.60)
        # Keep the stars in a sky cell together for the validation split.
        sky=np.floor(xy[good]/8.).astype('i8')
        test=((sky[:,0]*2654435761+sky[:,1]*2246822519)%1000003)%5==0
        samples.append((e,r,c,b,phase,pix,test))
        available+=len(r)
    if available<500:return None
    sums=np.zeros(store.phase.shape);counts=np.zeros_like(sums)
    for e,r,c,b,ph,res,test in samples:
        train=~test;sn=a['snr'][r]
        Z=store.coordinates(a['x'][r],a['y'][r])
        # A CCD registration has an offset and a slowly varying distortion.
        # Share the exposure offset, and shrink slopes on sparsely sampled CCDs.
        pool=np.median(res[train],axis=0) if train.any() else np.zeros(2)
        prior=np.array([15.*pool,np.zeros(2),np.zeros(2)])
        eye=np.diag([18.,30.,30.])
        order=np.argsort(c,kind='stable')
        for ii in np.split(order,np.flatnonzero(np.diff(c[order]))+1):
            use=ii[train[ii]]
            if not len(use):continue
            weight=np.minimum(1.,.55/np.maximum(.001,np.linalg.norm(res[use],axis=1)))
            weight*=np.minimum(1.,(sn[use]/13.)**2)
            wz=Z[use]*weight[:,None]
            store.affine[e,c[ii[0]]]=np.linalg.solve(eye+wz.T@Z[use],prior+wz.T@res[use])
        # The compressed centroid phase also depends on how strongly peaked
        # the PSF measurement is. Calibrate its remaining bias in detector axes.
        resid=res-np.einsum('ni,nij->nj',Z,store.affine[e,c])
        shape,pb=store.marks(sharp[r],sn,ph)
        for axis in (0,1):
            key=(b[train]*5+shape[train])*8+pb[train,axis]
            wt=np.minimum(1.,.62/np.maximum(.001,np.abs(resid[train,axis])))
            sums[...,axis]+=np.bincount(key,wt*np.sign(ph[train,axis])*resid[train,axis],minlength=160).reshape(4,5,8)
            counts[...,axis]+=np.bincount(key,wt,minlength=160).reshape(4,5,8)
    store.phase[:]=gaussian_filter(sums,(.45,.50,.55,0))/(
                                  gaussian_filter(counts,(.45,.50,.55,0))+30.)
    before=[];after=[]
    for e,r,c,b,ph,res,test in samples:
        sn=a['snr'][r];x=a['x'][r];y=a['y'][r];h=sharp[r]
        delta=store.predict(e,c,x,y,sn,b,h,ph)
        for axis in (0,1):
            p=ph[:,axis];u=np.clip(res[:,axis]-delta[:,axis],-.8,.8)
            ct=np.bincount(c[~test],(p*p)[~test],minlength=store.nc)
            cross=np.bincount(c[~test],(p*u)[~test],minlength=store.nc)
            store.gain[e,:,axis]=np.clip(cross/(ct+5.),-.7,.7)
        before.append(np.abs(res[test]))
        after.append(np.abs((res-.65*store.predict(e,c,x,y,sn,b,h,ph))[test]))
    before=np.concatenate(before).ravel();after=np.concatenate(after).ravel()
    if len(after)<100:return None
    old,new=np.median(before),np.median(after)
    print('relative stellar field',len(after)//2,'held-out centroids, median absolute residual',
          round(float(old),4),'to',round(float(new),4),'pixels',flush=True)
    if new>.996*old or np.quantile(after,.90)>1.015*np.quantile(before,.90):return None
    # The comparison sample is dominated by isolated stars. Shrink its
    # residual solution when transferring to faint transient centroids.
    store.scale=.65
    return store
