import numpy as np, json, collections, sys
Z=np.load('/tmp/w/cache_p4.npz'); keep=Z['keep']; de=Z['de']; snr=Z['snr_k'].astype(float); clean=Z['seed2']; seed=Z['seed']
E=json.load(open('/tmp/p4/exposures.json')); et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
T=np.load('/tmp/p4/truth.npz'); bod=-np.ones(int(T['ntot']),np.int64); bod[T['rows']]=T['body']; kb=bod[keep]
bn=collections.defaultdict(set)
for b,e in zip(T['body'],T['exp']): bn[int(b)].add(int(night[e]))
scored=[b for b in bn if len(bn[b])>=5]
found=set(np.load(sys.argv[1]).tolist()) if len(sys.argv)>1 else set()
o=np.argsort(kb,kind='stable'); kbs=kb[o]
tab=collections.Counter(); tabf=collections.Counter()
def maxwin(nts, w):
    nts=np.sort(np.array(sorted(nts))); 
    return max((np.searchsorted(nts, t+w, side='right')-i) for i,t in enumerate(nts)) if len(nts) else 0
for b in scored:
    ii=o[np.searchsorted(kbs,b):np.searchsorted(kbs,b,side='right')]
    kn=set(night[de[ii]].tolist()); sn=set(night[de[ii[seed[ii]]]].tolist()); bnn=set(night[de[ii[snr[ii]>=6.5]]].tolist())
    nk=len(kn)
    m128=maxwin(kn,128); s128=maxwin(sn,128)
    # seasons split at gaps >120
    nb=np.array(sorted(kn)); parts=np.split(nb,np.flatnonzero(np.diff(nb)>120)+1) if len(nb) else []
    per=sorted([len(p) for p in parts],reverse=True)
    key=('kept<5' if nk<5 else 'win128>=5' if m128>=5 else 'win128=%d'%m128)
    key2=key+' | seedwin128=%d'%min(s128,5) + ' | n2seasons=%d'%min(sum(v>=2 for v in per),3)
    tab[key]+=1; tabf[key]+= b in found
    tab[key2]+=1; tabf[key2]+= b in found
for k in sorted(tab): print(f'{k:60s} {tabf[k]:4d} / {tab[k]:4d}')
