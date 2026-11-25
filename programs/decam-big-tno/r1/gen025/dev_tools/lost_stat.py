import numpy as np, json, pickle, collections
Z=np.load('/tmp/w/cache_p4.npz'); keep=Z['keep']; de=Z['de']; snr=Z['snr_k'].astype(float); clean=Z['seed2']
E=json.load(open('/tmp/p4/exposures.json')); et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
T=np.load('/tmp/p4/truth.npz'); bod=-np.ones(int(T['ntot']),np.int64); bod[T['rows']]=T['body']; kb=bod[keep]
found=set(np.load(''+__import__('sys').argv[1]+'').tolist())
# scored bodies: >=5 nights among all planted detections
allb=T['body']; alln=night[T['exp']] if 'exp' in T.files else None
bn=collections.defaultdict(set)
for b,n in zip(allb,alln): bn[int(b)].add(int(n))
scored=[b for b in bn if len(bn[b])>=5]
print('scored',len(scored),'found',len(found&set(scored)))
grown=pickle.load(open('/tmp/w/base/grown.pkl','rb')); g3=pickle.load(open('/tmp/w/base/g3.pkl','rb'))
arcs=collections.defaultdict(list)   # body -> list of (nnights, det idx)
def add(det,kind):
    b=kb[det]; 
    if (b>=0).all() and (b==b[0]).all():
        arcs[int(b[0])].append((len(np.unique(night[de[det]])),det,kind))
for v in grown.values():
    if v is not None and not v.get('src'): add(np.asarray(v['det']),'g')
for c in g3: add(np.asarray(c['kdet']),'3')
o=np.argsort(kb,kind='stable'); kbs=kb[o]
res=collections.Counter(); tab=collections.defaultdict(list)
for b in scored:
    if b in found: continue
    ii=o[np.searchsorted(kbs,b):np.searchsorted(kbs,b,side='right')]
    nk=len(np.unique(night[de[ii]]))
    if nk<5: res['kept nights <5']+=1; continue
    if b not in arcs: res['no pure arc (>=3 nights)']+=1; continue
    na,det,kind=max(arcs[b],key=lambda a:a[0])
    an=set(night[de[det]].tolist())
    far=[i for i in ii if night[de[i]] not in an]
    def nn(thr,cl=False): return len(set(night[de[i]] for i in far if snr[i]>=thr and (clean[i] or not cl)))
    smed=np.median(snr[det])
    key=f'arc{min(na,5)}'
    tab[key].append((nn(4.5),nn(4.5,True),nn(6.0,True),nn(7.0,True),nn(8.0,True),smed))
print(res)
for k in sorted(tab):
    v=np.array(tab[k]); print(k,'lost bodies with such an arc:',len(v))
    for need in (1,2,3):
        print('   far nights >=%d: any %d, clean %d, clean S/N>=6 %d, >=7 %d, >=8 %d'%(need,*[(v[:,j]>=need).sum() for j in range(5)]), '| arc medS/N>=8 & clean far S/N>=7: %d'%((v[:,3]>=need)&(v[:,5]>=8)).sum())
