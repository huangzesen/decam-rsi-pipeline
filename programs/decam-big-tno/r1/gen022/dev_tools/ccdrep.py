import numpy as np, json, time, sys
t0=time.time()
D='/tmp/p4/'
x=np.load(D+'det_x.npy',mmap_mode='r'); y=np.load(D+'det_y.npy',mmap_mode='r'); ccd=np.load(D+'det_ccd.npy',mmap_mode='r'); exp=np.load(D+'det_exp.npy',mmap_mode='r')
print(x.dtype,y.dtype,ccd.dtype, x[:5],y[:5],ccd[:5])
x=np.asarray(x); y=np.asarray(y); ccd=np.asarray(ccd).astype(np.int64); exp=np.asarray(exp).astype(np.int64)
print('x range',np.nanmin(x),np.nanmax(x),'y',np.nanmin(y),np.nanmax(y),'ccd',ccd.min(),ccd.max(), time.time()-t0)
Z=np.load('/tmp/w/cache_p4.npz'); keep=Z['keep']; snr=Z['snr_k']
T=np.load('/tmp/p4/truth.npz'); planted=np.zeros(len(x),bool); planted[T['rows']]=True
B=float(sys.argv[1]) if len(sys.argv)>1 else 3.0
res={}
for off in (0.0,0.5):
    cx=np.floor(x/B+off).astype(np.int64); cy=np.floor(y/B+off).astype(np.int64)
    key=(ccd*8192+cx)*8192+cy
    # distinct exposures per ccd cell
    ke=np.unique(key*1024+exp)   # (cell, exposure) distinct
    cell=ke//1024
    uc,cnt=np.unique(cell,return_counts=True)
    pos=np.searchsorted(uc,key)
    res[off]=cnt[pos]
    print('off',off,'done',time.time()-t0)
n=np.maximum(res[0.0],res[0.5])
np.save('/tmp/w/ccdrep.npy',n.astype(np.int32))
nk=n[keep]
print('all detections: distinct exposures in own CCD cell: pct',np.percentile(n,[50,90,99,99.9]))
for thr in (2,3,4,5,6,8,10,15,20,40):
    print('thr >=',thr,'all %.4f'%np.mean(n>=thr),'kept %.4f'%np.mean(nk>=thr),'kept snr>=6.5 %.4f'%np.mean(nk[snr>=6.5]>=thr),'planted(kept) %.4f'%np.mean(n[keep][planted[keep]]>=thr), 'nonkept %.4f'%np.mean(n[~np.isin(np.arange(len(n)),keep,assume_unique=False)]>=thr) if thr==2 else '')
