import numpy as np, json, time
t=time.time()
D='/app/data/sample/'
ra_a=np.load(D+'det_ra.npy',mmap_mode='r'); dec_a=np.load(D+'det_dec.npy',mmap_mode='r')
snr_a=np.load(D+'det_snr.npy',mmap_mode='r')
dec_a=np.asarray(dec_a); ra_a=np.asarray(ra_a); snr_a=np.asarray(snr_a)
m=(np.abs(dec_a-17.6)<2.0)&(np.abs(ra_a-28.9)<2.0)&(snr_a>12)
idx=np.nonzero(m)[0]
print(len(idx),time.time()-t)
r=ra_a[idx]; d=dec_a[idx]; snr=snr_a[idx]
ex=np.load(D+'det_exp.npy',mmap_mode='r')[idx]
sh=np.load(D+'det_sharp.npy',mmap_mode='r')[idx]; fl=np.load(D+'det_flag.npy',mmap_mode='r')[idx]
ccd=np.load(D+'det_ccd.npy',mmap_mode='r')[idx]
E=json.load(open(D+'exposures.json'))
mjd=np.array([e['mjd_mid'] for e in E]); band=np.array([e['band'] for e in E])
x=(r-28.9)*np.cos(np.radians(17.6))*3600; y=(d-17.6)*3600
# group into sources by 0.6" grid friends-of-friends (simple: round to cells, use kd-tree on star-like)
from scipy.spatial import cKDTree
ok=(fl==0)&(sh>0.8)&(sh<1.2)
ii=np.nonzero(ok)[0]
tr=cKDTree(np.c_[x[ii],y[ii]])
pairs=tr.query_pairs(0.7,output_type='ndarray')
import scipy.sparse as sp, scipy.sparse.csgraph as cg
n=len(ii)
g=sp.coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n))
nc,lab=cg.connected_components(g,directed=False)
cnt=np.bincount(lab,minlength=nc)
good=cnt[lab]>=5
mx=np.bincount(lab,weights=x[ii],minlength=nc)/np.maximum(cnt,1); my=np.bincount(lab,weights=y[ii],minlength=nc)/np.maximum(cnt,1)
# leave-one-out residual
c=cnt[lab]
dx=(x[ii]-mx[lab])*c/np.maximum(c-1,1); dy=(y[ii]-my[lab])*c/np.maximum(c-1,1)
dx=dx[good]; dy=dy[good]; e=ex[ii][good]; cc=ccd[ii][good]; s=snr[ii][good]; c=c[good]
print('star dets',len(dx),'sources',int((cnt>=5).sum()),time.time()-t)
hi=s>40
print('floor per-axis (snr>40, stars): robust', 1.4826*np.median(np.abs(np.r_[dx[hi],dy[hi]])), 'incl. ref err factor sqrt(1+1/(n-1)) ~', np.sqrt(1+1/np.median(c[hi]-1)))
# per exposure median offsets
ue=np.unique(e)
rows=[]
for u in ue:
    k=(e==u)&hi
    if k.sum()<30: continue
    rows.append((u,np.median(dx[k]),np.median(dy[k]),k.sum(),1.4826*np.median(np.abs(dx[k]-np.median(dx[k])))))
rows=np.array(rows)
print('exposures with >=30 bright stars:',len(rows))
off=np.hypot(rows[:,1],rows[:,2])
print('per-exposure median offset: median %.3f  p90 %.3f  max %.3f'%(np.median(off),np.percentile(off,90),off.max()))
print('within-exposure robust scatter after removing exposure median: median %.3f p90 %.3f'%(np.median(rows[:,4]),np.percentile(rows[:,4],90)))
for rr in rows[np.argsort(-off)[:12]]:
    u=int(rr[0]); print(u,band[u],round(mjd[u],2),'off=(%.3f,%.3f) n=%d sc=%.3f'%(rr[1],rr[2],rr[3],rr[4]))
# per exposure-ccd
key=e.astype(np.int64)*100+cc
uk,inv,ck=np.unique(key[hi],return_inverse=True,return_counts=True)
mdx=np.bincount(inv,weights=dx[hi])/ck; mdy=np.bincount(inv,weights=dy[hi])/ck
gk=ck>=15
print('exp-ccd groups',gk.sum(),'mean offset rms per axis %.3f'%np.sqrt(np.mean(np.r_[mdx[gk],mdy[gk]]**2)), ' expected from noise ~0.1/sqrt(15+)')
resx=dx[hi]-mdx[inv]; resy=dy[hi]-mdy[inv]
g2=gk[inv]
print('floor after exp-ccd mean removal: %.3f'%(1.4826*np.median(np.abs(np.r_[resx[g2],resy[g2]]))))
# by year
yr=((mjd[e]-56800)//365.25).astype(int)
for Y in np.unique(yr):
    k=hi&(yr==Y); 
    if k.sum()>100: print('year',Y,'n',k.sum(),'med dx %.3f dy %.3f  sig %.3f'%(np.median(dx[k]),np.median(dy[k]),1.4826*np.median(np.abs(np.r_[dx[k]-np.median(dx[k]),dy[k]-np.median(dy[k])]))))
