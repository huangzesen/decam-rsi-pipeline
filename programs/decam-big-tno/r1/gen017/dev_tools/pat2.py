import json, numpy as np, collections, sys
T=np.load('/tmp/p2/truth.npz'); E=json.load(open('/tmp/p2/exposures.json'))
et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
nn=T['nights']; body=T['body']; ex=T['exp']
pats=collections.Counter(); info={}
for b in np.flatnonzero(nn>=5):
    ns=np.unique(night[ex[body==b]])
    # seasons: split at gaps > 150 d
    grp=np.split(ns, np.flatnonzero(np.diff(ns)>150)+1)
    pat=tuple(sorted([len(g) for g in grp], reverse=True))
    def maxin(w): return max(int(((ns>=a)&(ns<=a+w)).sum()) for a in ns)
    info[b]=dict(n=len(ns), pat=pat, m32=maxin(32), m64=maxin(64), m130=maxin(130), ns=ns)
import pickle; pickle.dump(info, open('/tmp/w/info.pkl','wb'))
tot=collections.Counter(); 
for b,i in info.items():
    cls = 'ge5in130' if i['m130']>=5 else ('4in64' if i['m64']>=4 else ('4in130' if i['m130']>=4 else ('3in32' if i['m32']>=3 else ('3in64' if i['m64']>=3 else ('3in130' if i['m130']>=3 else 'max2')))))
    i['cls']=cls
    tot[(min(i['n'],9) if i['n']<8 else 8, cls)]+=1
for k in sorted(tot): print(k, tot[k])
