import numpy as np, json, pickle, collections, sys
Z=np.load('/tmp/w/cache_p4.npz'); keep=Z['keep']; de=Z['de']; snr=Z['snr_k'].astype(float); clean=Z['seed2']
E=json.load(open('/tmp/p4/exposures.json')); et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
T=np.load('/tmp/p4/truth.npz'); bod=-np.ones(int(T['ntot']),np.int64); bod[T['rows']]=T['body']; kb=bod[keep]
found=set(np.load(sys.argv[1]).tolist())
D=pickle.load(open(sys.argv[2]+'/carry.pkl','rb'))
dbg=[d for d in D['dbg'] if d.get('one',0)>0]
print('one-night dbg entries', len(dbg), 'all', len(D['dbg']))
# truth: per call; entries with one>0
tr=[t for t in (D['truth'] or []) if t.get('one',0)>0]
# seeds of the one-night call are indexed 0.. in order of dbg entries (rank 0, one per seed)
vot=collections.defaultdict(list)
for t in tr:
    for s_,d_,ok_,z_ in zip(t['seed'],t['det'],t['ok'],t['z']):
        vot[int(s_)].append((int(d_), bool(np.any(ok_)), float(z_[np.argmax(ok_)]) if np.any(ok_) else 9.0))
cnt=collections.Counter()
for k,d in enumerate(dbg):
    b=bod[d['det']]
    if not ((b>=0).all() and (b==b[0]).all()): continue
    b=int(b[0])
    if b in found: continue
    v=vot.get(k,[])
    far_b=[x for x in v if snr[x[0]]>=7 and clean[x[0]]]
    okv=[x for x in v if x[1]]
    st=('no true far det in boxes' if not v else 'true far det not a voter' if not okv else 'voter')
    cnt[st]+=1
    print(b, st, 'lam1 %.2f'%d.get('lam1',-1), 'm', d.get('m'), 'cnt', d.get('cnt'), 'acc', d.get('acc'), 'zs', d.get('zs'), 'smaj %.3f smin %.3f'%(d['smaj'],d['smin']), 'voters z', [round(x[2],2) for x in okv][:4], 'nfar', len(v))
print(cnt)
