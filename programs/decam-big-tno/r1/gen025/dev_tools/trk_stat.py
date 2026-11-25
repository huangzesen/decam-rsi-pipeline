import numpy as np, json, time, sys
from scipy.spatial import cKDTree
Z=np.load('/tmp/w/cache_p4.npz'); keep=Z['keep']; th=Z['th']; de=Z['de'].astype(np.int64); snr=Z['snr_k']
print({k:(Z[k].shape,Z[k].dtype) for k in Z.files})
E=json.load(open('/tmp/p4/exposures.json')); et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
T=np.load('/tmp/p4/truth.npz'); print(T.files)
rows=np.flatnonzero(keep) if keep.dtype==bool else keep
bod=-np.ones(int(T['ntot']),np.int64); bod[T['rows']]=T['body']; kb=bod[rows]
AS=np.radians(1/3600)
nE=len(E); o=np.argsort(de,kind='stable'); st=np.searchsorted(de[o],np.arange(nE)); en=np.searchsorted(de[o],np.arange(nE),side='right')
t0=time.time(); PI=[];PJ=[]
trees={}
for n in np.unique(night):
    ex=np.flatnonzero(night==n)
    if len(ex)<2: continue
    for a in range(len(ex)):
        for b in range(a+1,len(ex)):
            ea,eb=ex[a],ex[b]
            dt=abs(et[ea]-et[eb])*24
            R=(0.8+4.5*dt)*AS
            ia=o[st[ea]:en[ea]]; ib=o[st[eb]:en[eb]]
            if len(ia)==0 or len(ib)==0: continue
            # quick bbox overlap
            if th[ia,0].max()<th[ib,0].min()-R or th[ib,0].max()<th[ia,0].min()-R or th[ia,1].max()<th[ib,1].min()-R or th[ib,1].max()<th[ia,1].min()-R: continue
            if R>60*AS: R=60*AS
            for e_ in (ea,eb):
                if e_ not in trees: trees[e_]=cKDTree(th[o[st[e_]:en[e_]]])
            pr=trees[ea].query_ball_tree(trees[eb],R)
            for k,l in enumerate(pr):
                for q in l:
                    PI.append(ia[k]);PJ.append(ib[q])
PI=np.array(PI);PJ=np.array(PJ)
print('pairs',len(PI),'time',time.time()-t0)
dt=np.abs(et[de[PI]]-et[de[PJ]])*24
sep=np.hypot(*(th[PI]-th[PJ]).T)/AS
print('dt hours pct',np.percentile(dt,[5,25,50,75,95]))
tp=(kb[PI]>=0)&(kb[PI]==kb[PJ])
print('true pairs',tp.sum(),'true-pair dt pct',np.percentile(dt[tp],[5,25,50,75,95]), 'sep pct', np.percentile(sep[tp],[5,50,95]))
print('all sep pct',np.percentile(sep,[5,25,50,75,95]))
for lo,hi in ((0,0.1),(0.1,0.3),(0.3,1),(1,3),(3,12)):
    m=(dt>=lo)&(dt<hi); print('dt',lo,hi,'pairs',m.sum(),'true',(tp&m).sum())
np.savez('/tmp/w/trk.npz',PI=PI,PJ=PJ)
# bodies: tracklet nights
nb=int(T['body'].max())+1
nights_all={}
import collections
bn=collections.defaultdict(set); bt=collections.defaultdict(set)
for r_,b_ in zip(np.flatnonzero(kb>=0),kb[kb>=0]): bn[b_].add(night[de[r_]])
for i in np.flatnonzero(tp): bt[kb[PI[i]]].add(night[de[PI[i]]])
sc=[b for b in bn if len(bn[b])>=5]
print('bodies with >=5 kept nights',len(sc))
print('tracklet nights hist',np.bincount([len(bt[b]) for b in sc]))
import pickle; pickle.dump(dict(bn=dict(bn),bt=dict(bt)),open('/tmp/w/trkbody.pkl','wb'))
