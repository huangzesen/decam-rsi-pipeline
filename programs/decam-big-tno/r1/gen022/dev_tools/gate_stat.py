import numpy as np, json
Z=np.load('/tmp/w/cache_p4.npz'); keep=Z['keep']; de=Z['de']; flux=Z['flux']; snr=Z['snr_k'].astype(float); mzc=Z['mzc']
E=json.load(open('/tmp/p4/exposures.json')); band=[e['band'] for e in E]
col=dict(g=0.65,r=0.0,i=-0.26,z=-0.44,Y=-0.48,VR=0.11); cb=np.array([col.get(b,0.0) for b in band])
lev_e=np.array([dict(g=1.0,r=0.0,i=-0.5,z=-0.9,Y=-1.0,VR=0.2).get(b,0.0) for b in band])
mz=np.array([e.get('magzero') if e.get('magzero') is not None else np.nan for e in E],float)
mz=np.where(np.isfinite(mzc),mzc,mz) if len(mzc)==len(E) else mz
mag=mz[de]-2.5*np.log10(np.maximum(flux,1e-9))-cb[de]; lev=lev_e[de]
et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
T=np.load('/tmp/p4/truth.npz'); bod=-np.ones(int(T['ntot']),np.int64); bod[T['rows']]=T['body']; kb=bod[keep]
FG,INT,COL=2.5,0.15,0.18
def gate(sn,m,lv,med,msig,alev):
    dm=m-med; snp=sn*10**(0.4*np.clip(dm,-8,8)); s2=INT**2+(0.92*msig)**2+(COL*(lv-alev))**2
    z=(sn-snp)/np.sqrt(1+s2*snp*snp); return ~(np.abs(z)>FG)&~(np.abs(dm)>1.6)
rng=np.random.default_rng(1)
o=np.argsort(kb,kind='stable'); kbs=kb[o]; st=np.searchsorted(kbs,np.arange(kbs.max()+1)); en=np.searchsorted(kbs,np.arange(kbs.max()+1),side='right')
bybin={}
rnd=rng.integers(0,len(de),200000)   # random kept detections = chance population
for b in range(len(st)):
    ii=o[st[b]:en[b]]
    if len(np.unique(night[de[ii]]))<5: continue
    # arc = 4 random members; test = the others
    for trial in range(2):
        a=rng.choice(ii,4,replace=False); rest=np.setdiff1d(ii,a)
        ma=mag[a]; fin=np.isfinite(ma)
        if fin.sum()<2: continue
        med=np.nanmedian(ma); sa=np.maximum(snr[a],4.5); msig=1.25*np.sqrt(((1.09/sa[fin])**2).sum())/fin.sum(); alev=lev[a][fin].mean(); smed=np.median(sa)
        okt=gate(np.maximum(snr[rest],4.5),mag[rest],lev[rest],med,msig,alev); old_t=~(np.abs(mag[rest]-med)>1.0)
        okc=gate(np.maximum(snr[rnd],4.5),mag[rnd],lev[rnd],med,msig,alev); old_c=~(np.abs(mag[rnd]-med)>1.0)
        k=min(int(smed//2)*2,20)
        bybin.setdefault(k,[]).append((okt.sum(),old_t.sum(),len(rest),okc.mean(),old_c.mean()))
print('arc median S/N bin | arcs | true members passing: new, old | chance detections passing: new, old | ratio')
for k in sorted(bybin):
    v=np.array(bybin[k]); print(f'{k:3d}-{k+2:<3d} {len(v):5d}   true {v[:,0].sum()/v[:,2].sum():.3f} {v[:,1].sum()/v[:,2].sum():.3f}   chance {v[:,3].mean():.3f} {v[:,4].mean():.3f}   {v[:,3].mean()/max(v[:,4].mean(),1e-9):.2f}')
