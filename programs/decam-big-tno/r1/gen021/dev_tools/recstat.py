import json, sys, numpy as np
cands=json.load(open(sys.argv[1])); T=np.load('/tmp/p4/truth.npz'); Z=np.load('/tmp/w/cache_p4b.npz')
E=json.load(open('/tmp/p4/exposures.json')); night=np.floor(np.array([e['mjd_mid'] for e in E])-0.5).astype(int)
ntot=int(T['ntot']); body=-np.ones(ntot,np.int64); body[T['rows']]=T['body']
rexp=np.zeros(ntot,np.int64); rexp[T['rows']]=T['exp']
ina=np.zeros(ntot,bool); ina[Z['aux_rows']]=True
found=set(); good=bad=0; via=[0,0]
for c in cands:
    d=np.array(c['det']); b=body[d]; bb=b[b>=0]; ok=False
    if len(bb):
        u,cnt=np.unique(bb,return_counts=True); j=int(u[cnt.argmax()])
        ok = cnt.max()>=0.8*len(d) and len(np.unique(night[rexp[d[b==j]]]))>=5
        if ok: found.add(j)
    if c.get('nax'):
        ax=d[ina[d]]; g=int((body[ax]>=0).sum()); good+=g; bad+=len(ax)-g
        if c['nights']-0 >= 5 and len(np.unique(night[np.load('/tmp/p4/det_exp.npy',mmap_mode='r')[d[~ina[d]]]]))<5: via[0 if ok else 1]+=1
print(sys.argv[1], 'found', len(found), '| second-tier members: planted', good, 'not planted', bad, '| five nights only through them: true', via[0], 'not', via[1])
