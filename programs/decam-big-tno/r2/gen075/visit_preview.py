"""Shortlists of independent exposures reachable from each part of a shutter."""
import numpy as np

def schedules(cat, family):
    ex=cat.view;grid=cat.grid
    lo=np.column_stack((grid['bx'],grid['by']))*grid['cell'][:,None]
    size=np.column_stack((grid['nx'],grid['ny']))*grid['cell'][:,None]
    top=lo+size
    output=np.full((len(ex),9,24),-1,'i4')
    east=np.asarray(cat.direction);north=east[::-1]*[-1,1]
    observer=np.column_stack((ex['ox'],ex['oy']))
    radius=(44.,39.,85.,47.)[family];width=(.11,.235,.49,.24)[family]
    speeds=(.1,.236,.54,.64)[family],(.14,.39,.73,.69)[family]
    for e in np.flatnonzero(grid['nx']):
        dt=ex['t']-ex['t'][e]
        for s in range(9):
            p=lo[e]+size[e]*((np.array([s%3,s//3])+.5)/3.)
            norm=np.sqrt(1.+np.sum((p/206264.806)**2));g=norm/radius
            obs=206264.806*(observer-observer[e])-(ex['oz']-ex['oz'][e])[:,None]*p
            sp=.01720209895*206264.806*norm/radius**1.5
            vel=sp*dt[:,None]*east
            den=(1.-g*ex['oz'])[:,None];shift=(vel-g*obs)/den;center=p+shift
            der=(1.5*vel/g-obs+shift*ex['oz'][:,None])*g*width/den
            su=(sp*dt*speeds[0])**2;sv=(sp*dt*speeds[1])**2
            xx=der[:,0]**2+su*east[0]**2+sv*east[1]**2+.06
            yy=der[:,1]**2+su*east[1]**2+sv*east[0]**2+.06
            xy=der[:,0]*der[:,1]+(su-sv)*east.prod()
            area=np.sqrt(np.maximum(.0001,xx*yy-xy**2))
            sep=np.maximum(0.,np.maximum(lo-center,center-top))
            tile_noise=(size[e]/8.)**2
            chi=sep[:,0]**2/(xx+tile_noise[0])+sep[:,1]**2/(yy+tile_noise[1])
            expansion=np.maximum(1.,(np.sqrt(xx+tile_noise[0])*np.sqrt(yy+tile_noise[1]))/
                                   (np.maximum(800.,size[:,0])*np.maximum(800.,size[:,1])))
            score=-.62*np.log(.035+area*grid['density'])-.52*chi-.45*np.log(expansion)
            score+=.67*np.clip(ex['depth']-ex['depth'][e],-2.5,1.6)
            valid=(grid['nx']>0)&(ex['night']!=ex['night'][e])&(chi<10.)&(ex['depth']>10.)
            rank=np.flatnonzero(valid);rank=rank[np.argsort(-score[rank],kind='stable')]
            _,first=np.unique(ex['night'][rank],return_index=True);rank=rank[np.sort(first)]
            duration=abs(dt[rank]);bands=np.digitize(duration,[27.,185.,760.])
            reserv=np.concatenate([rank[bands==k][:num] for k,num in enumerate((6,6,4,4))])
            # The additional shutter peaks are reserved for the most selective dates.
            kept=np.isin(rank,reserv)
            choice=np.r_[rank[kept],rank[~kept]][:24]
            output[e,s,:len(choice)]=choice
    return output

def root_order(cat):
    rng=np.random.default_rng(17681)
    odds=cat.point['q']+.93*rng.gumbel(size=len(cat.point))
    return np.argsort(-odds,kind='stable').astype('i4')
