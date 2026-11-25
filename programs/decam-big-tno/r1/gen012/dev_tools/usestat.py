# share of seeds (all / planted) that the useful-seed test keeps, per pass window set
import sys, os, json, math, numpy as np
sys.path.insert(0, '/tmp/m'); os.environ.setdefault('FINDER_USEF','1')
import finder as F, torch
G=F.G
D='/tmp/p4'; Z=np.load('/tmp/w/cacheB.npz')
keep, seed, th, de, snr_k, flux = Z['keep'], Z['seed'], Z['th'], Z['de'], Z['snr_k'], Z['flux']
E=json.load(open(D+'/exposures.json')); nE=len(E)
et=np.array([e['mjd_mid'] for e in E]); night_e=np.floor(et-0.5).astype(np.int64); nu,nidx=np.unique(night_e,return_inverse=True)
mz=np.array([e.get('magzero') if e.get('magzero') is not None else np.nan for e in E],float)
col=dict(g=0.6,r=0.0,i=-0.25,z=-0.4,Y=-0.45,VR=0.1); cb=np.array([col.get(e.get('band'),0.0) for e in E])
mag=mz[de]-2.5*np.log10(np.maximum(flux,1e-9))-cb[de]
sigf=np.full(nE,np.nan); ferr=flux/np.maximum(snr_k,1e-3)
for k in range(nE):
    i=np.flatnonzero((de==k)&(snr_k<10))
    if len(i)>=20: sigf[k]=np.median(ferr[i])
m5c=mz-2.5*np.log10(5*sigf)-cb
m5e=np.array([e.get('m5') if e.get('m5') is not None else np.nan for e in E],float)-cb
ok=np.isfinite(m5c)&np.isfinite(m5e); print('measured - catalogue m5: median',np.median((m5c-m5e)[ok]),'std',np.std((m5c-m5e)[ok]), 'nan sigf',(~np.isfinite(m5c)).sum())
print('r-equivalent depth pct 5/25/50/75/95',np.percentile(m5c[ok],[5,25,50,75,95]))
m5c=np.where(np.isfinite(m5c),m5c,m5e)
Fm=Z['F']; rp,dp=np.radians([e['ra0'] for e in E]),np.radians([e['dec0'] for e in E])
vp=np.stack([np.cos(dp)*np.cos(rp),np.cos(dp)*np.sin(rp),np.sin(dp)],1); den=np.maximum(vp@Fm[2],0.2); pt=np.stack([(vp@Fm[0])/den,(vp@Fm[1])/den],1)
ub=sorted(set(str(e.get('band')) for e in E))
o_=np.argsort(de,kind='stable'); ds_=de[o_]; st_=np.searchsorted(ds_,np.arange(nE)); en_=np.searchsorted(ds_,np.arange(nE),side='right'); has_=np.flatnonzero(en_>st_)
fc_=np.zeros((nE,2)); fr_=np.full(nE,-1.0); xs_=th[o_,0]; ys_=th[o_,1]; s0_=st_[has_]
fc_[has_,0]=0.5*(np.minimum.reduceat(xs_,s0_)+np.maximum.reduceat(xs_,s0_)); fc_[has_,1]=0.5*(np.minimum.reduceat(ys_,s0_)+np.maximum.reduceat(ys_,s0_))
d_=np.hypot(xs_-fc_[ds_,0],ys_-fc_[ds_,1]); fr_[has_]=np.maximum.reduceat(d_,s0_)+math.radians(0.02)
print('footprint radius pct 5/50/99/max',np.degrees(np.percentile(fr_[has_],[5,50,99])),np.degrees(fr_.max()),'none',nE-len(has_))
G.update(fc_np=fc_,fr_np=fr_)
G.update(dev='cpu',m5c_np=m5c,pt_np=pt,th_np=th,bandi_np=np.array([ub.index(str(e.get('band'))) for e in E]),et_np=et,de_np=de,snr_np=snr_k,mag_np=mag,night_e_np=nidx.astype(np.int64),night_day=nu,seed_np=seed)
T=np.load(D+'/truth.npz'); body=-np.ones(int(T['ntot']),np.int64); body[T['rows']]=T['body']; bk=body[keep]
torch.set_num_threads(1)
for h in (8.0,16.0,32.0,64.0):
    wins=F.make_windows_days(dict(h=h)); tot=[0,0]; pl=[0,0]; pl3=[0,0]
    for Wn in wins:
        nights=Wn['nights']; lut=-np.ones(int(nidx.max())+2,np.int64); lut[nights]=np.arange(len(nights))
        dn=lut[G['night_e_np'][de]]; idx=np.flatnonzero((dn>=0)&seed)
        um=F.useful_seeds(idx,lut[G['night_e_np']],len(nights))
        tot[0]+=len(idx); tot[1]+=int(um.sum())
        b=bk[idx]; ip=b>=0; pl[0]+=int(ip.sum()); pl[1]+=int((ip&um).sum())
        # planted bodies with seeds on >= 3 nights of this window: are all their seeds kept?
        bb=b[ip]; nn_=dn[idx][ip]; uu=um[ip]
        key=np.unique(bb*1000+nn_); cnt=np.bincount(key//1000,minlength=3000)
        g3=cnt[bb]>=3
        pl3[0]+=int(g3.sum()); pl3[1]+=int((g3&uu).sum())
    print(f'h={h}: windows {len(wins)}, seeds {tot[0]:,} -> {tot[1]:,} ({tot[1]/tot[0]:.3f}); planted seeds {pl[0]} -> {pl[1]} ({pl[1]/max(pl[0],1):.3f}); planted seeds of bodies with >=3 seed nights in the window {pl3[0]} -> {pl3[1]} ({pl3[1]/max(pl3[0],1):.4f})',flush=True)
