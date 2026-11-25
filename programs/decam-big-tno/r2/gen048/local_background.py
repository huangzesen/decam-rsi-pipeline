"""Spatial likelihood of a point source versus the local transient catalogue."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve1d, uniform_filter
from scipy.special import expit, logit, ndtr


def _sum(values, index, count):
    return np.bincount(index, weights=values, minlength=count)


def _bands(counts):
    return convolve1d(counts, [.16,.68,.16], axis=-1, mode='nearest')


class Background:
    """Local odds referenced to a uniform, equally bright moving population."""

    def __init__(self, directory, primary, auxiliary=(), stop=None):
        path = Path(directory)
        read = lambda k: np.load(path/('det_'+k+'.npy'), mmap_mode='r')
        e, x, y, chip, flux, ferr, sn = map(read, ('exp','x','y','ccd','flux','ferr','snr'))
        times = [v['mjd_mid'] for v in json.loads((path/'exposures.json').read_text())]
        self.night = np.floor(np.asarray(times)-.5)
        self.exp = e
        self.snr = sn
        self.score = np.zeros(len(e),'i1')
        self.validation = np.zeros(3)
        self.visits = 0
        bounds = np.searchsorted(e,np.arange(len(times)+1))
        primary = np.asarray(primary,dtype='i8')
        auxiliary = np.asarray(auxiliary,dtype='i8')
        if np.any(primary[1:]<primary[:-1]):primary=np.sort(primary)
        if np.any(auxiliary[1:]<auxiliary[:-1]):auxiliary=np.sort(auxiliary)
        pi = np.searchsorted(primary,bounds)
        ai = np.searchsorted(auxiliary,bounds)
        for ei,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
            if stop is not None and stop.is_set():break
            selected = np.r_[primary[pi[ei]:pi[ei+1]],auxiliary[ai[ei]:ai[ei+1]]]-lo
            if hi-lo<120 or len(selected)<48:continue
            selected = np.unique(selected)
            sl = slice(lo,hi)
            s = self.exposure(x[sl],y[sl],chip[sl],flux[sl],ferr[sl],sn[sl],selected)
            if s is None:continue
            delta, validation = s
            self.score[sl] = np.rint(25.*delta).astype('i1')
            self.validation += validation
            self.visits += 1

    @staticmethod
    def exposure(x,y,chip,flux,err,snr,sel):
        n = len(chip)
        # The layout is in detector pixels. Missing chips and partial visits
        # only contribute their observed area to the spatial selection model.
        good = np.isfinite(x)&np.isfinite(y)&np.isfinite(flux)&np.isfinite(err)&(err>0.)
        faint = good & (snr<12.)
        if faint.sum()<64:return None
        sky = np.median(err[faint])
        if not np.isfinite(sky) or sky<=0.:return None
        x = np.clip(np.nan_to_num(x,nan=0.).astype('i4'),0,2047)
        y = np.clip(np.nan_to_num(y,nan=0.).astype('i4'),0,4095)
        _, c = np.unique(chip,return_inverse=True)
        c=c.astype('i4');nc=int(c.max())+1
        cells=nc*8*16
        cell = (c*8+x//256)*16+y//256
        count = np.bincount(cell,minlength=cells).astype(float)
        cover = np.clip(count/3.,0.,1.)
        near = uniform_filter(cover.reshape(nc,8,16), size=(1,3,3),mode='nearest').ravel()
        area = np.maximum(cover,.65*near)
        noise = np.log(np.clip(np.nan_to_num(err/sky,nan=1.),.25,5.))
        cnt = _sum(None,cell[faint],cells)
        total = _sum(noise[faint],cell[faint],cells)
        # Shrink an individual patch towards its own amplifier's noise.
        cn = cnt.reshape(nc,2,-1).sum(2);ct = total.reshape(nc,2,-1).sum(2)
        amp = np.repeat(ct/(cn+6.),64)
        local = (total+9.*amp)/(cnt+9.)
        nb = 16
        dm = -2.5*np.log10(np.clip(np.nan_to_num(flux,nan=1.e-12)/(5.*sky),1.e-10,1.e10))
        band = np.clip(np.floor((dm+5.75)*2),0,nb-1).astype('i4')
        mids = -.5*np.arange(nb,0,-1)+2.25+.25
        # Average over unresolved variability and threshold scatter to avoid
        # attributing every fluctuation near the limit to camera sensitivity.
        signal = 5.*10.**(-.4*mids)[None,:]/np.exp(local[:,None])
        detectable = .10+.90*ndtr((signal-4.7)/1.65)
        exposure_mean = (area[:,None]*detectable).sum(0)/max(1.,area.sum())
        relative = detectable/np.maximum(.02,exposure_mean)
        # The comparison catalogue is split within visits. The density at a
        # reported measurement is always estimated without that measurement.
        h = np.arange(n,dtype='u4')
        h ^= h>>16; h *= np.uint32(0x7feb352d); h ^= h>>15
        fold = h&1
        delta = np.zeros(n);validation = np.zeros(3)
        for k in range(2):
            use = sel[(fold[sel]!=k)&good[sel]]
            if len(use)<16:continue
            h0 = np.bincount(cell[use]*nb+band[use],minlength=cells*nb).reshape(nc,8,16,nb)
            hist = _bands(h0.astype(float))
            # Global, chip, half-chip, and local sky. At high count the
            # measured rate controls each scale; sparse cells borrow strength.
            expected = (hist.sum((0,1,2))+1.)/(area.sum()+1.)
            A = area.reshape(nc,8,16)
            chiprate = (hist.sum((1,2))+10.*expected)/(
                A.sum((1,2))[:,None]+10.)
            counts4 = hist.reshape(nc,2,4,4,4,nb).sum((2,4))
            area4 = A.reshape(nc,2,4,4,4).sum((2,4))
            prior = chiprate[:,None,None,:]
            # Pool in units of counts so a rare bright bin has an adequate
            # prior even if few sources were measured on its particular CCD.
            strength = 2.5/(prior+.01)+2.
            rate4 = (counts4+strength*prior)/(area4[:,:,:,None]+strength)
            sm = uniform_filter(hist,size=(1,3,3,1),mode='nearest')
            sa = uniform_filter(A,size=(1,3,3),mode='nearest')
            prior = np.repeat(np.repeat(rate4,4,axis=1),4,axis=2)
            weight = 2./(prior+.025)
            rate = ((1.8*hist+6.*sm+weight*prior)/
                    (1.8*A[:,:,:,None]+6.*sa[:,:,:,None]+weight))
            rate = rate.reshape(cells,nb)/np.maximum(expected,.0001)
            # Count thin column and row structures within an individual
            # amplifier. Only significant positive excesses carry a penalty.
            # Distinct magnitudes share this readout-pattern measurement.
            col = c*128+x//16
            row = (c*2+x//1024)*128+y//32
            def stripe(key):
                size = int(key.max())+1
                obs = np.bincount(key[use],minlength=size)
                ref = np.bincount(key,minlength=size).astype(float)
                pad = (-size)%16
                def pool(v):
                    a=np.pad(v,(0,pad)).reshape(-1,16).sum(1)
                    return np.repeat(a,16)[:size]
                pred = (ref+3.)*(pool(obs)+3.)/(pool(ref)+48.)
                return np.log1p(np.maximum(0.,obs-pred-1.6*np.sqrt(pred+1.))/(pred+3.))
            colr, rowr = stripe(col),stripe(row)
            selk = np.flatnonzero(fold==k)
            loc, mb = cell[selk],band[selk]
            bg = np.clip(rate[loc,mb],.14,40.)
            prob = np.clip(relative[loc,mb],.11,4.)
            delta[selk] = (np.log(prob/bg)-.65*(colr[col[selk]]+rowr[row[selk]]))
            validation[0] += np.log(np.maximum(.03,rate[cell[sel[fold[sel]==k]],band[sel[fold[sel]==k]]])).sum()
            validation[1] += len(sel[fold[sel]==k])
            validation[2] += len(use)
        delta[~good]=0.
        return np.clip(delta,-4.6,2.8),validation

    def values(self,rows):
        return self.score[rows].astype('f4')*.04

    def correction(self,rows,weight=.72):
        rows=np.asarray(rows,'i8')
        order=np.argsort(-self.snr[rows],kind='stable')
        _,first=np.unique(self.night[self.exp[rows[order]]],return_index=True)
        odds=self.values(rows[order[first]])
        return float(weight*np.clip(odds.sum(),-11.,8.))

    def update(self,reports,weight=.72):
        for c in reports:
            # The same region may be sampled in several exposures of a
            # season. Temper their spatial dependence in the final odds.
            delta=self.correction(c['det'],weight)
            z=logit(np.clip(c['p'],1.e-14,1.-1.e-14))+delta
            c['p']=float(expit(np.clip(z,-37.,37.)))
            c['local_density_odds']=round(float(delta),3)
        reports.sort(key=lambda c:-c['p'])


def begin(G,control):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='local-background')
    future=pool.submit(Background,G['scen'],G['keep_np'][:G['nk']],G['keep_np'][G['nk']:],control['event'])
    control['local_background']=future
    pool.shutdown(wait=False)
    return future


def finish(F,directory):
    job=(F.G.get('survey_control') or {}).get('local_background')
    if job is None or not job.done():return
    model=job.result()
    if not model.visits:return
    path=Path(directory)/'candidates.json'
    with path.open() as fp:reports=json.load(fp)
    model.update(reports)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as fp:json.dump(reports,fp)
    os.replace(tmp,path)
    F.log('spatial background:',model.visits,'visits, held-out log density gain',
          round(float(model.validation[0]/max(1.,model.validation[1])),3))
