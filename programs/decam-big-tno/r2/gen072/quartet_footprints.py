"""Use the other two visits' live area before truncating a measured pair list."""
import numpy as np
from scipy.ndimage import distance_transform_edt

import quartet_geometry as geometry

STEP = 240.
TILE = np.dtype([(k,'i4') for k in ('x','y','nx','ny','start')])
ROUTE = np.dtype([('center','f8',4),('offset','f8',4),('slope','f8',(4,4)),('radius','f8',2)])


def routes(samples):
    result=np.empty((len(samples),2),ROUTE)
    for role,(first,other) in enumerate(((slice(0,2),slice(2,4)),(slice(2,4),slice(0,2)))):
        v=geometry.pair_vector(samples[:,:,first])
        dest=samples[:,:,other].reshape(len(samples),-1,4)
        x,center,T=geometry.whitening(v)
        mean=dest.mean(1)
        slope=geometry.covar(x,dest-mean[:,None,:])
        res=(dest-mean[:,None,:]-x@slope).reshape(len(v),-1,2,2)
        # Use the longer axis for an inexpensive enclosing circle. Large
        # propagation uncertainties automatically admit a larger footprint.
        rr=np.sqrt(np.quantile(np.sum(res**2,axis=-1),.97,axis=1))
        r=result[:,role]
        r['center'],r['offset'],r['slope'],r['radius']=center,mean,T@slope,2.1*rr+1.55*STEP
    return result


class Footprints:
    def __init__(self,points,edges):
        self.tiles=np.zeros(len(edges)-1,TILE)
        pieces=[];start=0
        for e,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            x=np.floor(points['x'][lo:hi]/STEP).astype('i4')
            y=np.floor(points['y'][lo:hi]/STEP).astype('i4')
            if not len(x):
                self.tiles[e]=(0,0,0,0,start)
                continue
            x0,y0=x.min(),y.min()
            nx,ny=int(x.max()-x0)+1,int(y.max()-y0)+1
            if nx*ny>2000000:
                self.tiles[e]=(x0,y0,-nx,ny,start)
                continue
            occupied=np.zeros((nx,ny),bool);occupied[x-x0,y-y0]=True
            grid=np.minimum(255.,np.floor(4.*distance_transform_edt(~occupied))).astype('u1')
            self.tiles[e]=(x0,y0,nx,ny,start)
            pieces.append(grid.ravel());start+=grid.size
        self.grid=np.concatenate(pieces) if pieces else np.zeros(1,'u1')

    def accessible(self,xy,e,radius):
        tile=self.tiles[e];nx,ny=int(tile['nx']),int(tile['ny'])
        if nx<=0:
            return np.full(len(xy),nx<0)
        x=np.floor(xy[:,0]/STEP).astype('i4')-tile['x']
        y=np.floor(xy[:,1]/STEP).astype('i4')-tile['y']
        ix=np.clip(x,0,nx-1);iy=np.clip(y,0,ny-1)
        outside=np.hypot(x-ix,y-iy)
        distance=.25*self.grid[tile['start']+ix*ny+iy]
        return STEP*np.maximum(outside,distance-outside)<=radius

    def select(self,pairs,route,exposures,cap):
        if not len(pairs):
            return pairs
        forecast=(pairs['pos']-route['center'])@route['slope']+route['offset']
        allowed=(self.accessible(forecast[:,:2],exposures[0],route['radius'][0])&
                 self.accessible(forecast[:,2:],exposures[1],route['radius'][1]))
        take=np.flatnonzero(allowed)
        if len(take)>cap:
            take=take[np.argpartition(-pairs['quality'][take],cap-1)[:cap]]
        return pairs[take]
