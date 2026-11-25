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
    if (b>=0).all() and (b==b[0]).all(): arcs[int(b[0])].append((len(np.unique(night[de[det]])),det))
o=np.argsort(kb,kind='stable'); kbs=kb[o]
res=collections.Counter()
for b in scored:
    ii=o[np.searchsorted(kbs,b):np.searchsorted(kbs,b,side='right')]
    nts=collections.defaultdict(list)
    for i in ii: nts[night[de[i]]].append(i)
    # nights with two clean detections in different exposures, >= 20 min apart
    dbl=[n for n,l in nts.items() if len([i for i in l if clean[i]])>=2 and (max(et[de[i]] for i in l if clean[i])-min(et[de[i]] for i in l if clean[i]))*24*60>=20]
    dblb=[n for n in dbl if sum(1 for i in nts[n] if clean[i] and snr[i]>=7)>=2]
    st='found' if b in found else 'lost'
    if len(nts)<5: res[st+' kept<5']+=1; continue
    na=max(arcs[b],key=lambda a:a[0])[0] if b in arcs else 0
    an=set(night[de[max(arcs[b],key=lambda a:a[0])[1]]].tolist()) if b in arcs else set()
    fd=[n for n in dbl if n not in an]; fdb=[n for n in dblb if n not in an]
    res[f'{st} arc{min(na,5)} | far double nights (clean, >=20 min): {min(len(fd),2)} | both S/N>=7: {min(len(fdb),1)}']+=1
    res[f'{st} | any double night: {min(len(dbl),3)}']+=1
for k in sorted(res): print(f'{k:80s} {res[k]}')
