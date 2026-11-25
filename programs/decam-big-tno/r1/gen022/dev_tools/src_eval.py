import json, sys, numpy as np, collections
cands=json.load(open(sys.argv[1])); T=np.load('/tmp/p4/truth.npz')
E=json.load(open('/tmp/p4/exposures.json')); night=np.floor(np.array([e['mjd_mid'] for e in E])-0.5).astype(int)
exp=np.load('/tmp/p4/det_exp.npy',mmap_mode='r')
bod=-np.ones(int(T['ntot']),np.int64); bod[T['rows']]=T['body']
bn=collections.defaultdict(set)
for b,n in zip(T['body'],night[T['exp']]): bn[int(b)].add(int(n))
res=collections.defaultdict(lambda: collections.Counter()); found=set()
rows=[]
for c in cands:
    d=np.array(c['det']); b=bod[d]; src=c.get('src',0)
    pl=b[b>=0]
    if len(pl)==0: kind='none'
    else:
        bb=np.bincount(pl).argmax(); m=b==bb
        nn=len(set(night[np.asarray(exp[d[m]])].tolist()))
        kind='true' if (m.sum()>=0.8*len(d) and nn>=5 and len(bn[int(bb)])>=5) else 'mixed'
        if kind=='true': found.add(int(bb))
    pb='p>=0.5' if c['p']>=0.5 else ('p>=0.1' if c['p']>=0.1 else 'p<0.1')
    res[src][(kind,pb)]+=1
print('found',len(found))
for src in sorted(res):
    print('src',src,' '.join(f'{k}:{[res[src][(k,pb)] for pb in ("p>=0.5","p>=0.1","p<0.1")]}' for k in ('true','mixed','none')))
