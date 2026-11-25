"""Reserve costly validation for informative arcs throughout the preliminary ranks."""
import numpy as np


def select(G,reports,head,tail):
    ordered=sorted(reports,key=lambda c:-c['p'])
    if len(ordered)<=head+tail:return ordered
    high,remaining=ordered[:head],ordered[head:]
    ctrl=G.get('survey_control') or {}
    sky=ctrl.get('original_sky')
    if sky is None:
        return high+[remaining[j] for j in np.linspace(0,len(remaining)-1,tail,dtype=int)]
    weights=np.empty(len(remaining))
    for j,c in enumerate(remaining):
        rows=np.asarray(c['det'],dtype='i4')
        e=sky.exposure[rows];n=G['night_e_np'][e]
        sn=sky.arrays[0][rows]
        rank=np.argsort(-sn,kind='stable')
        _,ii,counts=np.unique(n[rank],return_index=True,return_counts=True)
        take=rank[ii];mag=np.asarray(sky.mag[rows[take]],'f8')
        lev=G['lev_e'][e[take]]
        good=np.isfinite(mag)&np.isfinite(sn[take])
        if good.sum()<4:
            weights[j]=-1.e3;continue
        m=mag[good];x=lev[good];x=x-np.mean(x)
        colour=np.clip((m-m.mean())@x/(x@x+.43),-.75,.75)
        residual=m-colour*x
        scatter=np.median(np.abs(residual-np.median(residual)))
        signal=np.log(np.maximum(4.5,np.minimum(sn[take][good],75.))/4.5)
        spread=np.log1p(np.minimum(4,counts-1)).sum()
        precision=np.clip(c.get('rms',.24)/.23,.25,3.)
        # A poorly rated but independently visible source can change rank
        # substantially after its original-image foreground is measured.
        weights[j]=(1.15*min(9,len(take))+.65*np.median(signal)+.40*spread
                    -1.3*np.log1p((scatter/.20)**2)-.67*precision**2)
    priority=np.argsort(-weights,kind='stable')[:tail//2]
    include=np.ones(len(remaining),bool);include[priority]=False
    other=np.flatnonzero(include)
    controls=other[np.linspace(0,len(other)-1,tail-len(priority),dtype=int)]
    # Mix the informative sample and the ordinary tail if time runs short.
    choose=np.empty(tail,'i4');choose[1::2]=priority;choose[::2]=controls
    return high+[remaining[j] for j in choose]
