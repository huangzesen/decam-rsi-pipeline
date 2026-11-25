import numpy as np, json, math, time
from scipy.spatial import cKDTree
AS = math.pi/180/3600
T=np.load('/tmp/p5/truth.npz'); E=json.load(open('/tmp/p5/exposures.json')); nE=len(E)
et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
C=np.load('/tmp/w/cache_p5.npz'); keep=C['keep']; th=C['th']; de=C['de']; seed=C['seed']; seed2=C['seed2']; snr=C['snr_k']
ntot=int(T['ntot']); bod=-np.ones(ntot,int); bod[T['rows']]=T['body']; bk=bod[keep]
def same_night(th, de, et, ne, sel, r0=0.8, rate=4.6, kp=6, tmax=60.0):
    """mask over detections: has a partner (also in sel) in another exposure of the night within r0 + rate*dt_h arcsec"""
    t0=time.time(); nE=len(et)
    idx=np.flatnonzero(sel); idx=idx[np.argsort(de[idx],kind='stable')]; ds=de[idx]
    st=np.searchsorted(ds,np.arange(nE)); en=np.searchsorted(ds,np.arange(nE),side='right')
    conf=np.zeros(len(de),bool); trees={}
    def tree(e):
        if e not in trees: trees[e]=cKDTree(th[idx[st[e]:en[e]]]) if en[e]>st[e] else None
        return trees[e]
    eo=np.argsort(et,kind='stable'); npair=0
    for n_ in np.unique(ne):
        ex=eo[ne[eo]==n_]
        if len(ex)<2: continue
        for ia,a in enumerate(ex[:-1]):
            if time.time()-t0>tmax: break
            for b in ex[ia+1:ia+1+kp]:
                dth=(et[b]-et[a])*24
                if dth>14: break
                ta,tb=tree(a),tree(b)
                if ta is None or tb is None: continue
                M=ta.sparse_distance_matrix(tb,(r0+rate*dth)*AS,output_type='ndarray'); npair+=1
                if len(M):
                    conf[idx[st[a]+M['i']]]=True; conf[idx[st[b]+M['j']]]=True
    return conf, npair
t=time.time(); conf,npair=same_night(th,de,et,night,seed2.astype(bool)); print('conf',conf.sum(),'of clean',seed2.sum(),'pairs',npair,'t',time.time()-t)
print('seeds',seed.sum(),'new seeds (conf & clean & ~seed):',(conf&seed2&~seed).sum(), ' planted among old seeds',(bk[seed]>=0).sum(),' planted among new',(bk[conf&seed2&~seed]>=0).sum())
print('planted kept',(bk>=0).sum(),'planted kept conf',(bk[conf]>=0).sum(), 'planted clean',(bk[seed2.astype(bool)]>=0).sum())
# bodies: seed nights in best 64-d window old vs new
nn=T['nights']; NB=len(nn)
newseed=seed|(conf&seed2)
def best(b,sd,win):
    m=np.flatnonzero((bk==b)&sd)
    if len(m)==0: return 0
    nt=np.unique(night[de[m]]); best=0
    for q in nt: best=max(best,int(((nt>=q)&(nt<=q+win)).sum()))
    return best
gain={}
for b in np.flatnonzero(nn>=5):
    for win in (32,64,128):
        o=best(b,seed,win); n=best(b,newseed,win)
        if n>o: gain.setdefault((win,o,n),[]).append(b)
for k in sorted(gain): print(k,len(gain[k]))
np.save('/tmp/w/gain_b.npy',np.array(sorted(set(sum([v for k,v in gain.items() if k[1]<=3],[])))))
