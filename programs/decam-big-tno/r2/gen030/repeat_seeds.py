"""Use repeated nightly observations to prioritize source proposals.

The orbit fit retains each exposure's own astrometry, flux and uncertainty.
Only proposal selection benefits from the independent detection evidence.
"""
import numpy as np
from night_repeats import flags


def apply(F):
    G=F.G;nk=G['nk']
    rep=flags(F,rate=8.,floor=.65,max_sep=6.,photometry=True)
    sn=np.asarray(G['snr_np'][:nk],dtype=np.float64)
    # At most three observations' information enters a proposal. This bounds
    # the influence of long sequences and multiple possible associations.
    strength=np.minimum(G['rep_strength'],np.sqrt(3.)*sn)
    effective=np.where(rep,np.maximum(sn,strength),sn)
    original=G['seed_np'].copy()
    G['seed_np'] |= G['seed2_np'] & rep & (effective>=F.SNR_SEED)
    for key in ('seedb_np','seedc_np'):
        mask=G[key]
        if not mask.any():continue
        threshold=float(np.min(sn[mask]))
        mask |= G['seed_np'] & (effective>=threshold)
    G['proposal_snr']=effective
    F.log('repeat-supported proposals:',int((G['seed_np']&~original).sum()),
          'new seeds;',int((G['seed_np']&(sn<8)&(effective>=8)).sum()),
          'additional bright-pool sources')
