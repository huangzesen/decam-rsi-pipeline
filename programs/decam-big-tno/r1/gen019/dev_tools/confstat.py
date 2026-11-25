import json, numpy as np, collections, time
from scipy.spatial import cKDTree
AS=np.pi/180/3600
Z=np.load('/tmp/w/cache_p4b.npz'); keep=Z['keep']; seed=Z['seed']; seed2=Z['seed2']; th=Z['th']; de=Z['de']; snr=Z['snr_k']
T=np.load('/tmp/p4/truth.npz'); E=json.load(open('/tmp/p4/exposures.json')); et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
ntot=int(T['ntot']); body=-np.ones(ntot,np.int64); body[T['rows']]=T['body']
bk=body[keep]; nk=night[de]; tk=et[de]
t0=time.time()
conf=np.zeros(len(keep),bool); mov=np.zeros(len(keep),bool)
o=np.argsort(nk,kind='stable'); ns=nk[o]; st=np.flatnonzero(np.r_[True,ns[1:]!=ns[:-1]]); en=np.r_[st[1:],len(o)]
RMAX=30.0
for a,b in zip(st,en):
    i=o[a:b]
    if len(np.unique(de[i]))<2: continue
    tr=cKDTree(th[i]/AS)
    pr=tr.query_pairs(RMAX,output_type='ndarray')
    if len(pr)==0: continue
    p,q=i[pr[:,0]],i[pr[:,1]]
    m=de[p]!=de[q]
    p,q=p[m],q[m]
    d=np.hypot(*(th[p]-th[q]).T)/AS; dt=np.abs(tk[p]-tk[q])*24
    ok=d<=0.8+4.5*dt
    conf[p[ok]]=True; conf[q[ok]]=True
    mv=ok&(d>1.0)
    mov[p[mv]]=True; mov[q[mv]]=True
print('time',time.time()-t0)
pl=bk>=0
print('kept',len(keep),'conf',conf.sum(),'| planted kept',pl.sum(),'planted conf',(conf&pl).sum(), 'moving-conf', mov.sum(), 'planted moving-conf', (mov&pl).sum())
print('seeds',seed.sum(),'planted seeds',(seed&pl).sum(),'| seed2',seed2.sum(),'planted',(seed2&pl).sum())
for lo,hi in [(4.5,5),(5,5.5),(5.5,6.5),(6.5,9),(9,1e9)]:
    m=(snr>=lo)&(snr<hi)&seed2
    print(f'S/N {lo}-{hi}: seed2 {m.sum():8d} planted {(m&pl).sum():5d} | conf {(m&conf).sum():7d} planted {(m&conf&pl).sum():5d} | moving-conf {(m&mov).sum():6d} planted {(m&mov&pl).sum():4d}')
np.save('/tmp/w/conf.npy',conf); np.save('/tmp/w/mov.npy',mov)
