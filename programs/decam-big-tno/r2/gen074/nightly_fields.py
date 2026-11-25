"""Short, overlapping visits on a night transported to a shared epoch."""
import numpy as np

from quartet_geometry import AS,MU


class NightFields:
    def __init__(self, visits, center, counts, pole):
        self.raw = visits
        t,night = visits['t'],visits['night']
        groups = []
        assignment = np.zeros(len(t), 'i4')
        for n in np.unique(night):
            indices = np.flatnonzero(night==n)
            indices = indices[np.argsort(-visits['depth'][indices],kind='stable')]
            taken = np.zeros(len(indices),bool)
            for j,e in enumerate(indices):
                if taken[j]:
                    continue
                delta = center[indices]-center[e]
                # A survey tiling often steps several camera diameters
                # across one strip. Search it as one visit while transport
                # to a shared epoch remains accurate over the time window.
                nearby = (~taken)&(np.sum(delta**2,axis=1)<np.deg2rad(3.3)**2)&(abs(t[indices]-t[e])<=.125)
                times = np.sort(t[indices[nearby]])
                if times[-1]-times[0]>.125:
                    starts = times[(times<=t[e])&(times>=t[e]-.125)]
                    occupancy = np.searchsorted(times,starts+.125,side='right')-np.searchsorted(times,starts)
                    begin = starts[np.argmax(occupancy)]
                    nearby &= (t[indices]>=begin)&(t[indices]<=begin+.125)
                subset = indices[nearby]
                taken |= nearby
                assignment[subset] = len(groups)
                groups.append(subset)
        self.group = assignment
        self.member = groups
        chosen = np.array([ii[np.argmin(abs(t[ii]-np.median(t[ii])))] for ii in groups])
        self.visits = visits[chosen].copy()
        self.spread = np.array([np.ptp(t[ii]) for ii in groups])
        for k,ii in enumerate(groups):
            self.visits['depth'][k] = np.max(visits['depth'][ii])
        ob = np.column_stack([visits[k] for k in ('ox','oy','oz')])
        tc = t[chosen][assignment]
        diff = ob[chosen][assignment]-ob
        center = np.nan_to_num(center)
        direction = np.column_stack((center,np.ones(len(t))))
        norm = np.linalg.norm(direction,axis=1)
        east = np.cross(pole,direction)
        east /= np.maximum(np.linalg.norm(east,axis=1,keepdims=True),1.e-6)
        velocity = np.sqrt(MU/44.**3)*norm[:,None]*(east[:,:2]-center*east[:,2:])
        gamma = norm/44.
        self.shear = gamma*diff[:,2]
        self.offset = velocity*(tc-t)[:,None]-gamma[:,None]*diff[:,:2]

    def correct(self, xy, e):
        return np.asarray(xy + self.offset[e]+xy*self.shear[e,None], 'f4')

    def uncertainty(self, samples, plans, z):
        # Residual motion after the shared circular guide is much larger
        # in a multi-hour sequence for a distant or inclined trial orbit.
        family = plans[:,4]
        typical = np.array([.75,1.6,2.,3.2,3.6,2.8,2.7])[family]
        duration = 24.*self.spread[plans[:,:4]]*.32
        samples += (typical[:,None]*duration)[:,None,:,None]*z[None,:,6:].reshape(1,-1,4,2)
        return samples


def merge_measurements(xy, ex, group, ids, quality, mag, signal):
    """Keep one actual shutter when several give the same transported source."""
    # For short visits the unused shutters can independently confirm an
    # orbit later. Limit the multiplicity paid by the four-position join.
    cell = np.floor((xy-xy.min(axis=0))/(1.10*AS)).astype('i8')
    key = (group.astype('i8')<<44)|(cell[:,0]<<22)|cell[:,1]
    order = np.lexsort((-quality,key))
    edge = np.r_[0,1+np.flatnonzero(key[order[1:]]!=key[order[:-1]]),len(order)]
    chosen = order[edge[:-1]]
    bonus = np.zeros(len(chosen),'f4')
    repeat = np.flatnonzero(np.diff(edge)>1)
    second = order[edge[repeat]+1]
    a = chosen[repeat]
    dm = (mag[a]-mag[second])/np.hypot(.26,1.086/np.maximum(signal[a],4.5))
    dist = np.hypot(*(xy[a]-xy[second]).T)/AS
    bonus[repeat] = np.where(ex[a]!=ex[second],
                            1.5*np.exp(-.22*dm**2-.42*dist**2),0.)
    return chosen,quality[chosen]+bonus
