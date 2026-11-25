import numpy as np, json, time
t=time.time()
D='/app/data/sample/'
ra=np.load(D+'det_ra.npy',mmap_mode='r'); dec=np.load(D+'det_dec.npy',mmap_mode='r')
dec_a=np.asarray(dec); ra_a=np.asarray(ra)
m=(np.abs(dec_a-17.6)<0.6)&(np.abs(ra_a-28.9)<0.6)
idx=np.nonzero(m)[0]
print(len(idx),time.time()-t)
r=ra_a[idx]; d=dec_a[idx]
snr=np.load(D+'det_snr.npy',mmap_mode='r')[idx]; ex=np.load(D+'det_exp.npy',mmap_mode='r')[idx]
sh=np.load(D+'det_sharp.npy',mmap_mode='r')[idx]; fl=np.load(D+'det_flag.npy',mmap_mode='r')[idx]
E=json.load(open(D+'exposures.json'))
print('nexp in box',len(np.unique(ex)))
# cluster: grid 1" cells, friends via cKDTree
from scipy.spatial import cKDTree
x=(r-28.9)*np.cos(np.radians(17.6))*3600; y=(d-17.6)*3600
tr=cKDTree(np.c_[x,y])
# seeds: bright detections snr>30, define static source = mean of all dets within 0.5" among snr>15
b=np.nonzero((snr>20)&(fl==0))[0]
nb=tr.query_ball_point(np.c_[x[b],y[b]],1.5)
used=np.zeros(len(x),bool)
res=[]  # (snr, dx, dy, nexp)
for i,l in zip(b,nb):
    if used[i]: continue
    l=np.array(l)
    if len(l)<6: continue
    used[l]=True
    br=l[(snr[l]>15)]
    if len(br)<4: continue
    # isolated: check all members within 1.5" lie within 0.8" of the median => else skip (blends)
    mx=np.median(x[br]); my=np.median(y[br])
    for j in l:
        if j in br: 
            # leave-one-out
            o=br[br!=j]
            if len(o)<3: continue
            res.append((snr[j],x[j]-np.mean(x[o]),y[j]-np.mean(y[o]),len(br),sh[j]))
        else:
            res.append((snr[j],x[j]-mx,y[j]-my,len(br),sh[j]))
res=np.array(res); print(len(res),time.time()-t)
np.save('/tmp/w/astro_res.npy',res)
for lo,hi in [(4.5,5),(5,5.5),(5.5,6.5),(6.5,8),(8,10),(10,15),(15,25),(25,50),(50,100),(100,1000)]:
    s=res[(res[:,0]>=lo)&(res[:,0]<hi)]
    if len(s)<10: continue
    rr=np.hypot(s[:,1],s[:,2])
    mad=1.4826*np.median(np.abs(np.r_[s[:,1],s[:,2]]))
    model=np.sqrt(0.04**2+(1.1/np.median(s[:,0]))**2)
    print(f'snr {lo}-{hi}: n={len(s)} robust sig/axis={mad:.3f}" model={model:.3f}" r50={np.median(rr):.3f} r90={np.percentile(rr,90):.3f} r98={np.percentile(rr,98):.3f}  frac>0.6"={np.mean(rr>0.6):.3f} >1.0"={np.mean(rr>1.0):.3f}')
