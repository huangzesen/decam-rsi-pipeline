import numpy as np, json, time, sys
sys.path.insert(0,'/app/history/gen026')
D='/app/data/sample/'
t=time.time()
ra=np.asarray(np.load(D+'det_ra.npy',mmap_mode='r')); dec=np.asarray(np.load(D+'det_dec.npy',mmap_mode='r'))
m=(np.abs(dec-17.6)<1.5)&(np.abs(ra-28.9)<1.5)
idx=np.nonzero(m)[0]
ra=ra[idx]; dec=dec[idx]
snr=np.load(D+'det_snr.npy',mmap_mode='r')[idx]; ex=np.load(D+'det_exp.npy',mmap_mode='r')[idx].astype(np.int64)
sharp=np.load(D+'det_sharp.npy',mmap_mode='r')[idx]; flag=np.load(D+'det_flag.npy',mmap_mode='r')[idx]
E=json.load(open(D+'exposures.json'))
et=np.array([e['mjd_mid'] for e in E]); night_e=np.floor(et-0.5).astype(np.int64)
nu,nidx=np.unique(night_e,return_inverse=True)
AS=np.radians(1/3600)
x=np.radians((ra-28.9)*np.cos(np.radians(17.6))); y=np.radians(dec-17.6)
nd=nidx[ex]
area=9.0*3600**2*np.cos(np.radians(17.6))**0  # sq arcsec approx (x already scaled)
print('dets',len(x),'density per sq arcsec',len(x)/(9*3600**2*1.0), 'nexp',len(np.unique(ex)),time.time()-t)
def static_count(x,y,nd,cell=1.2,w=None):
    N=len(x); cnt=np.zeros(N,np.int16)
    for sx in (0,.5):
        cx=np.floor(x/(cell*AS)+sx).astype(np.int64); cx-=cx.min()
        for sy in (0,.5):
            cy=np.floor(y/(cell*AS)+sy).astype(np.int64); cy-=cy.min()
            key=((cx*(int(cy.max())+1)+cy)<<12)|nd
            o=np.argsort(key,kind='stable'); ks=key[o]
            nk=np.ones(N,bool); nk[1:]=ks[1:]!=ks[:-1]
            if w is not None: nk=nk&w[o]   # count only nights having a 'w' detection... approx: first det of night in cell must be w
            c=ks>>12; ncl=np.ones(N,bool); ncl[1:]=c[1:]!=c[:-1]
            gid=np.cumsum(ncl)-1
            nn=np.bincount(gid,weights=nk).astype(np.int16)
            cnt[o]=np.maximum(cnt[o],nn[gid])
    return cnt
cnt=static_count(x,y,nd)
print('cnt>=3 share',np.mean(cnt>=3),'cnt==2',np.mean(cnt==2),'cnt==1',np.mean(cnt==1),time.time()-t)
sg=3.4/np.maximum(snr,4.5)
keep=(cnt<3)&((flag&1)==0)&(sharp>=np.minimum(0.35,1-2.5*sg))&(sharp<=np.maximum(2.5,1+2.5*sg))&(sharp!=0)&(snr>=4.5)
shape_seed=((flag&~np.uint8(2))==0)&(sharp>=np.minimum(0.5,1-2.0*sg))&(sharp<=np.maximum(1.8,1+2.0*sg))&(snr>=5.0)&keep
print('kept',keep.sum(),'seed-like cnt<2:',(shape_seed&(cnt<2)).sum(),' seed-like cnt==2:',(shape_seed&(cnt==2)).sum())
# chance veto: fake points at random positions on random nights, weighted like exposures: shift real seed positions by 30"
rng=np.random.default_rng(1)
sd=np.nonzero(shape_seed&(cnt<2))[0]
ang=rng.uniform(0,2*np.pi,len(sd)); sh=30*AS
fx=x[sd]+sh*np.cos(ang); fy=y[sd]+sh*np.sin(ang)
X=np.r_[x,fx]; Y=np.r_[y,fy]; ND=np.r_[nd,nd[sd]]
c2=static_count(X,Y,ND)[len(x):]
print('chance: fake seeds with cnt>=2: %.4f  cnt>=3: %.4f'%(np.mean(c2>=2),np.mean(c2>=3)))
# variant B: veto counts only S/N>=5 detections on other nights (the fake itself counts)
for thr in (5.0,5.5):
    selB=snr>=thr
    xb=np.r_[x[selB],fx]; yb=np.r_[y[selB],fy]; nb=np.r_[nd[selB],nd[sd]]
    cb=static_count(xb,yb,nb)[selB.sum():]
    # real: detections with snr>=thr have cnt_b
    cr=static_count(x[selB],y[selB],nd[selB])
    full=np.zeros(len(x),np.int16); full[selB]=cr
    ss=shape_seed&(snr>=thr)
    print(f'veto by S/N>={thr} only: chance veto {np.mean(cb>=2):.4f}; seed-like (snr>={thr}) with cntB<2: {(ss&(full<2)).sum()} vs with cnt<2: {(ss&(cnt<2)).sum()}')
# variant C: smaller cell
for cell in (0.8,0.6):
    cc=static_count(X,Y,ND,cell=cell)
    print(f'cell {cell}: chance veto {np.mean(cc[len(x):]>=2):.4f}; seed-like with cntC<2 (and cnt<3): {(shape_seed&(cc[:len(x)]<2)).sum()}')
# per-density scaling: non-static density
print('kept density per sq arcsec',keep.sum()/(9*3600**2),' all',len(x)/(9*3600**2))
