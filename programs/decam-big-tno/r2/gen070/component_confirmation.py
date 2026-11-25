"""Fit independently measured residual-flux centroids in the survey orbit index."""
import numpy as np


class Measurements:
    def __init__(self,kept,control):
        self.index=control['certifier']
        more=control['additional']
        offset=more['offset']
        begin=more['diff_start']
        self.corrected=np.asarray(more['rows'][begin-offset:],'i4')
        self.order=np.argsort(self.corrected,kind='stable').astype('i4')
        self.corrected=self.corrected[self.order]
        self.order+=begin
        self.kept=np.asarray(kept)
        self.primary=None
        if np.any(self.kept[1:]<self.kept[:-1]):
            self.primary=np.argsort(self.kept,kind='stable').astype('i4')
            self.kept=self.kept[self.primary]

    @staticmethod
    def map(rows,keys,values):
        if not len(keys):return np.full(rows.shape,-1,'i4')
        ix=np.minimum(np.searchsorted(keys,rows),len(keys)-1)
        return np.where(keys[ix]==rows,ix if values is None else values[ix],-1).astype('i4')

    def run(self,proposals,control,threads,seconds):
        import survey_arcs
        if not len(proposals) or control['event'].is_set():return 0
        p=proposals[:75000]
        base=self.map(p['row'],self.kept,self.primary)
        comp=self.map(p['row'],self.corrected,self.order)
        present=(comp>=0)&(base>=0)
        if not np.any(present):return 0
        any_good=(base>=0).all(1)
        # Test one measured component at a time, plus a joint solution for
        # multiple credible corrections. The flux correction must be compatible
        # with the independently observed brightness at the other epochs.
        mag=self.index.target['mag'];rawmag=mag[np.maximum(base,0)]
        anchor=np.median(rawmag,axis=1)
        cmag=mag[np.maximum(comp,0)]
        close=present&(np.abs(cmag-anchor[:,None])<1.12)
        ids=[];trials=[];merits=[]
        for j in range(4):
            take=np.flatnonzero(any_good&close[:,j])
            if not len(take):continue
            m=base[take].copy();m[:,j]=comp[take,j]
            ids.append(take);trials.append(m)
            gain=np.abs(rawmag[take,j]-anchor[take])-np.abs(cmag[take,j]-anchor[take])
            merits.append(p['merit'][take]+1.3*gain)
        take=np.flatnonzero(any_good&(close.sum(1)>1))
        if len(take):
            ids.append(take);trials.append(np.where(close[take],comp[take],base[take]))
            merits.append(p['merit'][take]-.30)
        # These same corrected sources are potential *new* nights for raw
        # four-position fits, even when none of the four measurements is blended.
        take=np.flatnonzero(any_good&~present.any(1))[:6200]
        if len(take):
            ids.append(take);trials.append(base[take]);merits.append(p['merit'][take]-2.6)
        if not ids:return 0
        ids=np.concatenate(ids)
        order=np.argsort(-np.concatenate(merits),kind='stable')[:36000]
        ids=ids[order];positions=np.ascontiguousarray(np.concatenate(trials)[order])
        w=self.map(p['witness'][ids],self.kept,self.primary)
        fits=survey_arcs.certify(self.index,positions,p['g'][ids],w,threads,seconds,limit=2800)
        fits=fits[fits['nights']>=5]
        if len(fits):control['certified'].put(fits)
        return len(fits)
