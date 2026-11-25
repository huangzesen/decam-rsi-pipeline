"""Marginal evidence for a common orbit, including partial-arc alternatives."""
from itertools import combinations
from pathlib import Path
import json
import os
import time

import numpy as np
from scipy.special import expit, logit, logsumexp

AS = np.deg2rad(1./3600.)
NM = 54


class TransientDensity:
    """Transient counts per magnitude per square arcsecond in each visit."""
    def __init__(self, directory, exposures, zp):
        path = Path(directory)
        load = lambda k: np.load(path/('det_'+k+'.npy'), mmap_mode='r')
        ra, dec, exp, flux, sharp, flag = map(load, ('ra','dec','exp','flux','sharp','flag'))
        ne = len(exposures)
        stride = max(1,len(ra)//3000)
        lon = np.deg2rad(ra[::stride])
        mid = np.rad2deg(np.arctan2(np.mean(np.sin(lon)),np.mean(np.cos(lon))))
        cosine = max(.05,np.cos(np.deg2rad(np.median(dec[::stride]))))
        samples = []
        # Sample sky area. All visits to a selected patch are present,
        # allowing a static counterpart to be measured at the actual depth.
        fraction = max(1,min(38,len(ra)//1200000))
        for lo in range(0,len(ra),600000):
            sl = slice(lo,lo+600000)
            x = ((ra[sl]-mid+180.)%360.-180.)*(3600.*cosine)
            y = dec[sl]*3600.
            ix,iy = np.floor(x/76.).astype('i8'),np.floor(y/76.).astype('i8')
            cell = ix*73856093 ^ iy*19349663
            take = np.flatnonzero((cell%fraction)==0)
            samples.append((take+lo,x[take],y[take],ix[take],iy[take]))
        if not any(len(s[0]) for s in samples):
            self.rho = np.full((ne,NM),.0005)
            return
        row,x,y,ix,iy = (np.concatenate(part) for part in zip(*samples))
        e = exp[row]
        dates = np.array([int(np.floor(v['mjd_mid']-.5)) for v in exposures])
        night = dates[e]
        small = (ix-ix.min())*(iy.max()-iy.min()+1)+iy-iy.min()
        size = int(small.max())+1
        patches = np.unique(e.astype('i8')*size+small)//size
        area = np.maximum(1,np.bincount(patches,minlength=ne))*76.**2
        # Count independent observing nights in offset astrometric cells.
        # Only the sampled interior contributes to the counts and area.
        persistent = np.zeros(len(row),bool)
        for dx,dy in ((.0,.0),(.5,.5)):
            gx = np.floor((x-x.min())/1.35+dx).astype('i8')
            gy = np.floor((y-y.min())/1.35+dy).astype('i8')
            skey = gx*(gy.max()+1)+gy
            order = np.lexsort((night,skey))
            key,nn = skey[order],night[order]
            first = np.r_[True,key[1:]!=key[:-1]]
            change = first | np.r_[False,nn[1:]!=nn[:-1]]
            start = np.flatnonzero(first)
            cnt = np.add.reduceat(change,start)
            persistent[order] |= np.repeat(cnt>=3,np.diff(np.r_[start,len(key)]))
        m = zp[e]-2.5*np.log10(np.maximum(1.e-12,flux[row]))
        mb = self.bin(m)
        inside = (np.abs(x/76.-np.rint(x/76.))>1.4/76.) & (np.abs(y/76.-np.rint(y/76.))>1.4/76.)
        usable = inside & np.isfinite(m) & ((flag[row]&1)==0) & (sharp[row]!=0.)
        key = e*NM+mb
        def counts(use):
            h = np.bincount(key[use],minlength=ne*NM).reshape(ne,NM)
            return np.array([np.convolve(v,[.10,.22,.36,.22,.10],'same') for v in h])
        raw = counts(usable)
        trans = counts(usable & ~persistent)
        # Rare bright foreground objects and incomplete spatial sampling
        # leave a finite background even when no local control passes.
        surfaces = area*(1.-2.8/76.)**2
        mean = (trans.sum(0)+1.)/(surfaces.sum()+1.)
        self.rho = 3.*((trans+.035*raw+1.2+6000.*mean)/(surfaces[:,None]+6000.))
        self.rho = np.maximum(self.rho,2.e-8)

    @staticmethod
    def bin(m):
        return np.clip(np.floor((np.nan_to_num(m,nan=29.)-11.)*3.).astype('i4'),0,NM-1)

    def density(self,e,m):
        return self.rho[e,self.bin(m)]


def families(F, v):
    """Local Gaussian components of an orbital phase-space prior."""
    x = v['x']
    g = x[2]
    radius = np.linalg.norm(np.r_[x[:2],1.])/g
    radial = np.r_[x[:2],1.]
    radial /= np.linalg.norm(radial)
    pole = F.G['frame_np'] @ np.array([0.,-.397778,.917482])
    east = np.cross(pole,radial)
    east /= max(.001,np.linalg.norm(east))
    axes = np.array([east,np.cross(radial,east),radial])
    speed_scale = np.sqrt(np.linalg.norm(np.r_[x[:2],1.]))
    speed = axes@x[3:]*speed_scale/np.sqrt(F.MU*g**3)
    latitude = np.arcsin(np.clip(pole@radial,-1.,1.))
    # Gradients in log inverse range and velocity in circular-speed units.
    derivative = np.zeros((3,6))
    derivative[:,2] = -1.5*speed
    derivative[:,3:] = speed_scale*axes
    models = ((.37,44.,.105,.060,.066,.11,1.),(.45,43.,.34,.21,.36,.38,.94),
              (.18,None,1.,.67,.74,.68,.48 if radius<110. else 0.))
    output = []
    for weight,center,sr,sa,si,sh,along in models:
        if center is None:
            log_range = -np.log(np.log(9000./12.))
            range_gradient = 0.
            range_precision = .10
        else:
            z = np.log(radius/center)/sr
            log_range = -.5*z*z-np.log(np.sqrt(2.*np.pi)*sr)
            range_gradient = -z/sr
            range_precision = 1./sr**2
        inv = 1./np.array([sa,si,sh])
        delta = (speed-[along,0.,0.])*inv
        A = derivative*inv[:,None]
        precision = A.T@A
        precision[2,2] += range_precision
        gradient = A.T@delta
        gradient[2] += range_gradient
        lat = 0. if center is None else -.5*(latitude/si)**2
        lp = (np.log(weight)+log_range+lat-.5*(delta@delta)
              -1.5*np.log(2.*np.pi)+np.log(inv).sum())
        output.append((lp,precision,gradient))
    return output


def calculate(F, v, density):
    """Integrate an orbit and a linear colour relation for each night subset."""
    indices = v['take']
    nights = len(indices)
    if nights<5 or nights>15 or not np.all(np.isfinite(v['x'])) or not 1./9400.<v['x'][2]<1./10.:
        return None
    ex = v['e'][indices]
    sn = v['snr'][indices]
    mag = v['mag'][indices]
    if not np.all(np.isfinite(mag)):
        return None
    # One independent astrometric position per night. The existing
    # within-night motion test supplies the separate repeat evidence.
    sigma = np.hypot(.063,v['errors'][indices])
    residual = v['resxy'][indices]/sigma[:,None]
    J = v['design'][indices]/sigma[:,None,None]
    if not np.isfinite(J).all() or not np.isfinite(residual).all():
        return None
    median = np.median(mag)
    color_axis = F.G['lev_e'][ex]
    sm = np.hypot(.16,1.086/np.maximum(sn,4.5))
    # Noisy flux estimates at a positive detection threshold are bright
    # on average. A steep but finite number-count slope debiases them.
    dm = (mag+1.45/np.maximum(sn,4.5)**2-median)/sm
    X = np.column_stack((np.ones(nights),color_axis))/sm[:,None]
    rho = density.density(ex,mag)
    const = (-np.log(2.*np.pi*sigma**2*rho)-.5*np.sum(residual**2,axis=1)
             -np.log(np.sqrt(2.*np.pi)*sm)-.5*dm**2)
    # A matching foreground source explains the measurement under the
    # omitted-night hypothesis. Local motion or flux excess can support
    # the orbit when measured independently of this row.
    if 'local_motion' in v:
        motion = v['local_motion'][indices]
        const += .65*np.clip(motion['slow']-3.,0.,5.)-.8*np.clip(motion['fast']-7.5,0.,7.)
    if 'static_flux' in v:
        star = v['static_flux']['star'][indices]
        const += .34*np.clip(-star-2.,0.,5.)-.75*np.clip(star-2.,0.,11.)
    # All eligible complete tracks include a five-night orbit. Competing
    # subsets represent a shorter real arc with unrelated catalogue peaks.
    omissions = [()]
    max_drop = 2 if nights<=9 else 1
    for k in range(1,max_drop+1):
        omissions.extend(combinations(range(nights),k))
    masks = np.ones((len(omissions),nights))
    for k,drop in enumerate(omissions):
        if drop:masks[k,list(drop)] = 0.
    mt = masks@masks[0]
    geom = (masks@np.einsum('nci,ncj->nij',J,J).reshape(nights,36)).reshape(-1,6,6)
    grad = masks@np.einsum('nci,nc->ni',J,residual)
    phot = (masks@np.einsum('ni,nj->nij',X,X).reshape(nights,4)).reshape(-1,2,2)
    phot[:,1,1] += 1./.27**2
    pg = masks@(-X*dm[:,None])
    pc = .5*np.einsum('ni,ni->n',pg,np.linalg.solve(phot,pg[:,:,None])[:,:,0])
    pc += .5*np.log(2.*np.pi)-np.log(.27)-.5*np.linalg.slogdet(phot)[1]
    sky = -2.*np.log(3600.)+.53*np.log(10.)*np.clip(median-23.,-5.5,3.5)
    base = masks@const+pc+sky+3.*np.log(2.*np.pi)
    components = []
    for lp,P,prior_grad in families(F,v):
        mat = geom+P
        rhs = grad+prior_grad
        sign,ld = np.linalg.slogdet(mat)
        dx = np.linalg.solve(mat,rhs[:,:,None])[:,:,0]
        # Far from the expansion point the Gaussian integral can extend
        # beyond physical distances. Cap the gain of those poorly fixed fits.
        gain = .5*np.einsum('ni,ni->n',rhs,dx)
        scale = np.minimum(1.,.42/(np.abs(dx[:,2])+1.e-12))
        gain *= scale*(2.-scale)
        components.append(np.where(sign>0,lp+gain-.5*ld,-1.e10))
    logz = base+logsumexp(components,axis=0)
    valid = (mt>=5)&(mt>=.80*nights-.01)
    # A partly observed object may leave no catalogued member on an
    # omitted night. Marginalizing a missing observation is an alternative
    # to insisting that a good four-night arc certifies its weakest member.
    logz -= .7*(nights-mt)
    support = logsumexp(logz[valid])-np.logaddexp(0.,logsumexp(logz[~valid]))
    if nights>5:
        eligible=mt>=5
        posterior=np.exp(logz[eligible]-logsumexp(logz[eligible]))
        v['night_membership']=posterior@masks[eligible]
    return float(logz[0]),float(support)


def decontaminate(F, items, cat):
    import image_followup as I
    replacements = []
    for k,v in enumerate(items):
        mark = v.get('static_flux')
        night = F.G['night_e_np'][v['e']]
        c = v['candidate']
        strong=np.zeros(len(night),bool)
        if mark is not None:strong=(mark['nights']>=3)&(mark['star']>8.)&(mark['excess']<2.)
        posterior=v.get('night_membership')
        if posterior is not None and np.any(posterior<.035):
            # A night should fail as a whole. An independently agreeing
            # exposure on that night preserves its original measurements.
            photerr=np.hypot(.18,1.086/np.maximum(v['snr'],4.5))
            photo=np.abs(v['mag']-v['mr']-v['color']*F.G['lev_e'][v['e']])/photerr
            ast=v['res']/np.hypot(v['errors'],.07)
            for nn in night[v['take'][posterior<.035]]:
                members=night==nn
                if np.all((photo[members]>2.5)|(ast[members]>2.35)):
                    strong[members]=True
        if c.get('_components'):
            strong &= ~np.isin(v['d'],list(c['_components']))
        if not strong.any():continue
        keep = ~strong
        if len(np.unique(night[keep]))<5:continue
        trial = dict(c,det=v['d'][keep].tolist(),_orbit=(v['x'],v['ref']))
        piece = I.prepare(F,[trial],cat)
        if piece:
            new = piece[0]
            for name in ('static_flux','local_motion'):
                if name in v:new[name] = v[name][keep]
            replacements.append((k,new,v,keep))
    if not replacements:return
    I.model(F,[new for _,new,_,_ in replacements],refit=True)
    for k,new,old,keep in replacements:
        if not np.isfinite(new['res']).all() or np.max(new['res'])>max(.75,np.max(old['res'])):continue
        items[k] = new
        c = new['candidate']['_report']
        n = len(new['take'])
        # The proposal confidence included the discarded observations.
        loss = .9*(len(old['take'])-n)+.2*np.count_nonzero(~keep)
        c['p'] = float(expit(logit(np.clip(c['p'],1.e-12,1.-1.e-12))-loss))
        c.update(det=new['d'].tolist(),nights=n,membership_removed=int(np.sum(~keep)),
                 rms=round(float(np.sqrt(np.mean(new['res']**2))),4))
        distance=np.linalg.norm(np.r_[new['x'][:2],1.])/new['x'][2]
        c.update(nfit=len(new['d']),r=round(float(distance),2),
                 arc=round(float(np.ptp(F.G['et_np'][new['e']])),1))
        F.G.setdefault('report_orbits',{})[tuple(sorted(c['det']))]=(new['x'],new['ref'])


def adjustment(c, full, support):
    # The astrometric/SED calculation shares data with the proposal score.
    # Missing exposures, static counterparts and repeated images are
    # evidence outside this likelihood and retain their separate terms.
    p = np.clip(float(c['p']),1.e-12,1.-1.e-12)
    old = logit(p)
    n = int(c.get('nights',5))
    ancillary = (1.5*min(3,c.get('h3m',0))-.95*c.get('ndm',0)
                 -.40*c.get('ndm2',0)+.62*c.get('field_log_odds',0.)
                 +.65*c.get('sensor_log_odds',0.)+.72*min(0.,c.get('visit_ll',0.)))
    physical = min(full, support+1.1*max(0,n-5))+ancillary
    return float(np.clip(.42*(physical-old),-6.,5.))


def assignments(F,candidates,items,priority,deadline):
    """Compare competing measured tracks before one claims shared rows."""
    job=F.G.get('association_rates')
    if job is None or not job.done() or job.exception() is not None:return priority
    models={id(v['candidate']):v for v in items if 'design' in v and 5<=len(v['take'])<=15}
    if not models:return priority
    base=priority or {id(c):float(logit(np.clip(c['p'],1.e-12,1.-1.e-12))) for c in candidates}
    owners={};pairs=[]
    for c in sorted(candidates,key=lambda x:-base[id(x)]):
        k=id(c)
        if k not in models:continue
        counts={}
        for row in c['det']:
            for other in owners.get(row,()):
                counts[other]=counts.get(other,0)+1
        for other,n in counts.items():
            if n>=max(2,.28*len(c['det'])) and abs(base[k]-base[other])<8.:
                pairs.append((other,k))
        for row in c['det']:
            used=owners.setdefault(row,[])
            if len(used)<5:used.append(k)
    values={};out=dict(base)
    for first,second in pairs:
        if time.time()+.12>=deadline:break
        for key in (first,second):
            if key in values:continue
            v=models[key]
            try:res=calculate(F,v,job.result())
            except (FloatingPointError,np.linalg.LinAlgError):res=None
            values[key]=adjustment(v['candidate'],*res) if res is not None and np.isfinite(res).all() else None
        a,b=values[first],values[second]
        if a is None or b is None:continue
        out[first]=base[first]+.6*a
        out[second]=base[second]+.6*b
    return out


def rank(F,outdir,deadline):
    import image_followup as I
    import heldout_astrometry
    G = F.G
    job = G.get('association_rates')
    if job is None or not job.done() or job.exception() is not None:
        return
    density = job.result()
    path = Path(outdir)/'candidates.json'
    reports = json.loads(path.read_text())
    if not reports:return
    import foreground_motion
    cat = I.arrays(G['scen'])
    G['ob_np_follow'] = G['ob'].cpu().numpy()
    states = dict(G.get('report_orbits',{}))
    states.update(G.get('survey_orbits',{}))
    measured = 0
    ordered = sorted(reports,key=lambda c:-c['p'])
    for lo in range(0,min(1000,len(ordered)),40):
        if time.time()+.55>deadline:break
        chunk = []
        for c in ordered[lo:lo+40]:
            if c.get('nights',5)<5 or len(c['det'])>80:continue
            v = dict(c,_report=c)
            key = tuple(sorted(c['det']))
            if key in states:v['_orbit'] = states[key]
            if key in G.get('report_components',{}):v['_components'] = G['report_components'][key]
            chunk.append(v)
        items = I.prepare(F,chunk,cat)
        if not items:continue
        I.model(F,items)
        raw=(G.get('survey_control') or {}).get('original_sky')
        foreground_motion.evaluate(F,items,raw,max(.05,min(.9,deadline-time.time()-.10)))
        for v in items:
            try:
                v['marginal_odds']=calculate(F,v,density)
            except (FloatingPointError,np.linalg.LinAlgError):
                v['marginal_odds']=None
        decontaminate(F,items,cat)
        for v in items:
            c = v['candidate']['_report']
            if 'holdout_weight' not in c:
                ix=v['take']
                surface=np.clip(1.5*density.density(v['e'][ix],v['mag'][ix]),.00015,.035)
                held=heldout_astrometry.confirmation(F,v,surface=surface)
                shift=heldout_astrometry.log_weight(held,len(ix))
                c.update(holdout_odds=round(held,3),holdout_weight=round(shift,3))
                c['p']=float(expit(logit(np.clip(c['p'],1.e-12,1.-1.e-12))+shift))
            try:
                vals = v['marginal_odds'] if 'marginal_odds' in v else calculate(F,v,density)
            except (FloatingPointError,np.linalg.LinAlgError):
                vals = None
            measured += 1
            if vals is None or not np.isfinite(vals).all():continue
            delta = adjustment(c,*vals)
            fast,slow=foreground_motion.odds(v)
            if fast or slow:
                c.update(foreground_support=round(fast,3),local_repeat_support=round(slow,3))
            blend=v.get('static_flux')
            if blend is not None and np.any(blend['nights']):
                c['flux_confusion']=round(float(blend['star'].max()),3)
                c['flux_excess']=round(float(max(0.,-blend['star'].min())),3)
            c.update(orbit_log_bf=round(vals[0],3),association_log_bf=round(vals[1],3),
                     association_shift=round(delta,4))
            c['p'] = float(expit(logit(np.clip(c['p'],1.e-12,1.-1.e-12))+delta))
    if measured:
        reports.sort(key=lambda c:-c['p'])
        distinct={}
        for c in reports:distinct.setdefault(tuple(sorted(c['det'])),c)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(list(distinct.values())));os.replace(tmp,path)
        F.log('marginal common-orbit likelihood:',measured,'reports')


def start(directory, exposures, zp):
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='transient-counts')
    future = pool.submit(TransientDensity,directory,exposures,zp.copy())
    pool.shutdown(wait=False)
    return future
