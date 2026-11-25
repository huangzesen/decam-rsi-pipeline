import numpy as np, json, sys, time
t0=time.time()
src='/app/data/sample'
Z=np.load('/tmp/w/cacheB.npz'); keep=Z['keep']; seed=Z['seed']; seed2=Z['seed2']
T=np.load('/tmp/p4/truth.npz'); rows=T['rows']
# map planted-set rows back: keep indexes rows of /tmp/p4; planted rows have x=1000,y=2000
x=np.load('/tmp/p4/det_x.npy',mmap_mode='r'); y=np.load('/tmp/p4/det_y.npy',mmap_mode='r'); ccd=np.load('/tmp/p4/det_ccd.npy',mmap_mode='r'); ex=np.load('/tmp/p4/det_exp.npy',mmap_mode='r')
isp=np.zeros(len(ex),bool); isp[rows]=True
k=keep[~isp[keep]]; sd=seed[~isp[keep]]; s2=seed2[~isp[keep]]
xk=np.asarray(x[k]); yk=np.asarray(y[k]); ck=np.asarray(ccd[k]).astype(np.int64); ek=np.asarray(ex[k]).astype(np.int64)
print('kept real',len(k),'seeds',sd.sum(),'ccd range',ck.min(),ck.max(),'x',xk.min(),xk.max(),'y',yk.min(),yk.max(), f'{time.time()-t0:.0f}s')
snr=np.asarray(np.load('/tmp/p4/det_snr.npy',mmap_mode='r')[k])
for cs in (2.0,4.0,8.0):
    key=(ck<<40)|(np.floor(xk/cs).astype(np.int64)<<20)|np.floor(yk/cs).astype(np.int64)
    # count distinct exposures per cell
    ke=np.unique(np.stack([key,ek],1),axis=0)
    u,c=np.unique(ke[:,0],return_counts=True)
    cnt=c[np.searchsorted(u,key)]
    n=len(key); lam=n/ (62*2048*4096/cs**2)
    print(f'cell {cs}px: mean occupancy {lam:.4f}; frac of kept in cells with >=2 exps {np.mean(cnt>=2):.4f} (chance ~{1-np.exp(-lam):.4f}), >=3 {np.mean(cnt>=3):.4f}, >=5 {np.mean(cnt>=5):.4f}, >=10 {np.mean(cnt>=10):.4f}')
    print(f'    seeds: >=2 {np.mean(cnt[sd]>=2):.4f} >=3 {np.mean(cnt[sd]>=3):.4f} >=5 {np.mean(cnt[sd]>=5):.4f} >=10 {np.mean(cnt[sd]>=10):.4f};  bright seeds snr>=8: >=3 {np.mean(cnt[sd&(snr>=8)]>=3):.4f} n={int((sd&(snr>=8)).sum())}')
# 1-D: columns (x) and rows per ccd: bad columns
for nm,v in (('x',xk),('y',yk)):
    key=(ck<<20)|np.floor(v).astype(np.int64)
    u,c=np.unique(key,return_counts=True)
    med=np.median(c); cnt=c[np.searchsorted(u,key)]
    print(nm,'per-pixel-line median count',med,'frac of kept in lines with >5x median',np.mean(cnt>5*med),' >10x',np.mean(cnt>10*med), 'seeds >5x', np.mean(cnt[sd]>5*med))
# edge distance distribution
ed=np.minimum(np.minimum(xk,2048-xk),np.minimum(yk,4096-yk))
print('edge dist pct of seeds',np.percentile(ed[sd],[1,5,10,25,50]))
h,_=np.histogram(ed[sd],bins=[0,24,50,100,200,400,1024]); print('seed counts by edge distance bins',h, 'area-expected shares', np.diff([0,24,50,100,200,400,1024]))
