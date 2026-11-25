import numpy as np, json, pickle, collections, sys
Z=np.load('/tmp/w/cache_p4.npz'); keep=Z['keep']; de=Z['de']; snr=Z['snr_k'].astype(float); clean=Z['seed2']
E=json.load(open('/tmp/p4/exposures.json')); et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
T=np.load('/tmp/p4/truth.npz'); bod=-np.ones(int(T['ntot']),np.int64); bod[T['rows']]=T['body']; kb=bod[keep]
found=set(np.load(sys.argv[1]).tolist())
bn=collections.defaultdict(set)
for b,e in zip(T['body'],T['exp']): bn[int(b)].add(int(night[e]))
scored=[b for b in bn if len(bn[b])>=5]
grown=pickle.load(open('/tmp/w/base/grown_full.pkl','rb'))
arcs=collections.defaultdict(list)
for v in grown.values():
    if v is None or v.get('src'): continue
    det=np.asarray(v['det']); b=kb[det]
    if (b>=0).all() and (b==b[0]).all():
        sn=np.maximum(snr[det],4.5); rr=float(np.sqrt(np.mean(v['res']**2)))/float(np.sqrt(2*np.mean((1.1/sn)**2+0.04**2)))
        arcs[int(b[0])].append((len(np.unique(night[de[det]])),det,rr))
o=np.argsort(kb,kind='stable'); kbs=kb[o]
res=collections.Counter()
for b in scored:
    if b in found: continue
    ii=o[np.searchsorted(kbs,b):np.searchsorted(kbs,b,side='right')]
    nk=len(np.unique(night[de[ii]]))
    if nk<5: res['kept nights <5']+=1; continue
    if b not in arcs: res['no pure grown arc']+=1; continue
    na,det,rr=max(arcs[b],key=lambda a:a[0])
    an=set(night[de[det]].tolist())
    far=[i for i in ii if night[de[i]] not in an]
    nb7=len(set(night[de[i]] for i in far if snr[i]>=7 and clean[i])); nb55=len(set(night[de[i]] for i in far if snr[i]>=5.5 and clean[i])); nc=len(set(night[de[i]] for i in far if clean[i])); na_=len(set(night[de[i]] for i in far))
    key='arc%d %s | far nights: %s' % (min(na,5), 'fit ok ' if rr<=0.9 else 'fit bad', 'bright>=1' if nb7>=1 else 'clean5.5>=1' if nb55>=1 else 'clean>=1' if nc>=1 else 'unclean only' if na_>=1 else 'none')
    res[key]+=1
for k in sorted(res): print(f'{k:60s} {res[k]}')
# bodies with a well-fitting pure 4-night arc and a bright far night that are lost: what the loose one-night run made of them
if len(sys.argv) > 2:
    D = pickle.load(open(sys.argv[2] + '/carry.pkl', 'rb')); new = D['new']
    row2k = {int(r): i for i, r in enumerate(keep)} if False else None
    fa_b = {}
    for c in new:
        if c.get('src') != 4: continue
        d = np.asarray(c['det']); b = bod[d]
        if (b >= 0).all() and (b == b[0]).all(): fa_b[int(b[0])] = min(fa_b.get(int(b[0]), 9e9), c['fa'])
    lst = []
    for b in scored:
        if b in found or b not in arcs: continue
        ii=o[np.searchsorted(kbs,b):np.searchsorted(kbs,b,side='right')]
        if len(np.unique(night[de[ii]])) < 5: continue
        na,det,rr=max(arcs[b],key=lambda a:a[0])
        an=set(night[de[det]].tolist()); far=[i for i in ii if night[de[i]] not in an]
        nb7=len(set(night[de[i]] for i in far if snr[i]>=7 and clean[i]))
        if na == 4 and rr <= 0.9 and nb7 >= 1:
            t=et[de[det]]; lst.append((fa_b.get(b, -1), float(t.max()-t.min()), float(np.median(snr[det])), nb7, min(abs(et[de[i]]-t.mean()) for i in far if snr[i]>=7 and clean[i])))
    for r in sorted(lst): print('fa_loose %.3f arc %.0f d medsnr %.1f bright far nights %d nearest far %.0f d' % r)
