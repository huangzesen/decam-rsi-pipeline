import numpy as np, json, pickle, collections, sys
Z=np.load('/tmp/w/cache_p4.npz'); keep=Z['keep']; de=Z['de']; snr=Z['snr_k'].astype(float); clean=Z['seed2']
E=json.load(open('/tmp/p4/exposures.json')); et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
T=np.load('/tmp/p4/truth.npz'); bod=-np.ones(int(T['ntot']),np.int64); bod[T['rows']]=T['body']; kb=bod[keep]
found=set(np.load(sys.argv[1]).tolist())
bn=collections.defaultdict(set)
for b,n in zip(T['body'],night[T['exp']]): bn[int(b)].add(int(n))
scored=[b for b in bn if len(bn[b])>=5]
grown=pickle.load(open('/tmp/w/base/grown.pkl','rb')); g3=pickle.load(open('/tmp/w/base/g3.pkl','rb'))
arcs=collections.defaultdict(list)
def add(det):
    b=kb[det]
    if (b>=0).all() and (b==b[0]).all(): arcs[int(b[0])].append((len(np.unique(night[de[det]])),det))
for v in grown.values():
    if v is not None and not v.get('src'): add(np.asarray(v['det']))
for c in g3: add(np.asarray(c['kdet']))
o=np.argsort(kb,kind='stable'); kbs=kb[o]
out=collections.defaultdict(lambda: collections.Counter())
for b in scored:
    if b in found or b not in arcs: continue
    ii=o[np.searchsorted(kbs,b):np.searchsorted(kbs,b,side='right')]
    na,det=max(arcs[b],key=lambda a:a[0])
    an=set(night[de[det]].tolist()); t0,t1=et[de[det]].min(),et[de[det]].max()
    for SN in (4.5,7.0):
        for R in (1100,2600,5000):
            far=set(night[de[i]] for i in ii if night[de[i]] not in an and snr[i]>=SN and clean[i] and t0-R<=et[de[i]]<=t1+R)
            out[(min(na,4),SN)][(R,min(len(far),3))]+=1
for k in sorted(out):
    print('arc nights',k[0],'far clean S/N>=',k[1],'| bodies', sum(v for (R,n),v in out[k].items() if R==1100))
    for R in (1100,2600,5000):
        print('    reach %5d: far nights 0/1/2/3+ = %s'%(R,[out[k][(R,n)] for n in range(4)]))
