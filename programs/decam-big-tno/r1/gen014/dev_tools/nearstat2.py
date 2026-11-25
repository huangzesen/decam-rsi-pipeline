import numpy as np, json, sys, time, math
sys.path.insert(0,'/tmp/m')
from finder import tangent
t0=time.time()
D='/tmp/p4'
ex=np.load(D+'/det_exp.npy',mmap_mode='r')
x,y,F=tangent(np.load(D+'/det_ra.npy'),np.load(D+'/det_dec.npy'))
cnt=np.load('/tmp/w/cnt_p4.npy')
AS=math.radians(1/3600)
snr=np.load(D+'/det_snr.npy')
Z=np.load('/tmp/w/cacheB.npz'); keep=Z['keep']; seed=Z['seed']
T=np.load(D+'/truth.npz'); isp=np.zeros(len(ex),bool); isp[T['rows']]=True
x0,y0=x.min(),y.min()
kr=keep[~isp[keep]]; sd=seed[~isp[keep]]; ks=kr[sd]
rng=np.random.default_rng(1); ks=ks[rng.random(len(ks))<0.3]
print('n seeds sampled',len(ks),flush=True)
for minsn,lab in ((0,'all static'),(15,'static dets with S/N>=15')):
    st=(cnt>=3)&(snr>=minsn)
    cs=1.5*AS
    cx=np.floor((x[st]-x0)/cs).astype(np.int64); cy=np.floor((y[st]-y0)/cs).astype(np.int64)
    keys=np.unique((cx<<24)|cy)
    def ring(ii):
        qx=np.floor((x[ii]-x0)/cs).astype(np.int64); qy=np.floor((y[ii]-y0)/cs).astype(np.int64)
        out=np.full(len(ii),99)
        for d in range(0,6):
            hit=np.zeros(len(ii),bool)
            for dx in range(-d,d+1):
                for dy in range(-d,d+1):
                    if max(abs(dx),abs(dy))!=d: continue
                    k=((qx+dx)<<24)|(qy+dy)
                    p=np.searchsorted(keys,k).clip(0,len(keys)-1)
                    hit|=keys[p]==k
            out[hit&(out==99)]=d
        return out
    rs=ring(ks); rp=ring(T['rows'])
    print(lab, 'cells',len(keys))
    s8=snr[ks]>=8; s6=snr[ks]<6
    for d in list(range(6))+[99]:
        print(f'   d={d}: seeds {np.mean(rs==d):.4f}  planted {np.mean(rp==d):.4f}   bright seeds {np.mean(rs[s8]==d):.4f}  faint seeds(<6) {np.mean(rs[s6]==d):.4f}',flush=True)
print(time.time()-t0)
