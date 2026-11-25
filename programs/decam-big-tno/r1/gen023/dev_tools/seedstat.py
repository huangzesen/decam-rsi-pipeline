import numpy as np, json, math, sys
sys.path.insert(0,'/app/submission')
import finder as Fd
T=np.load('/tmp/p5/truth.npz'); E=json.load(open('/tmp/p5/exposures.json')); nE=len(E)
et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int); nu,nidx=np.unique(night,return_inverse=True)
C=np.load('/tmp/w/cache_p5.npz'); keep=C['keep']; th=C['th']; de=C['de']; seed=C['seed'].astype(bool); seed2=C['seed2'].astype(bool); snr=C['snr_k']
ntot=int(T['ntot']); bod=-np.ones(ntot,int); bod[T['rows']]=T['body']; bk=bod[keep]
cf=Fd.conf_mask(th,de,nidx.astype(np.int64),seed2)&seed2
seedb=seed&(snr>=6.5); newb=seedb|cf; news=seed|cf
print('conf',cf.sum(),'planted share: conf %.3f seedb %.3f seed %.3f added-to-seedb %.3f'%((bk[cf]>=0).mean(),(bk[seedb]>=0).mean(),(bk[seed]>=0).mean(),(bk[cf&~seedb]>=0).mean()))
found=set(np.load('/tmp/w/found_f1.npy').tolist()); nn=T['nights']
def pat(b,sd):
    m=np.flatnonzero((bk==b)&sd)
    nt=np.unique(night[de[m]])
    if len(nt)==0: return []
    g=np.flatnonzero(np.diff(nt)>120); return [len(p) for p in np.split(nt,g+1)]
def p8ok(per):  # two neighbouring seasons each with >=2 seed nights
    return any(per[i]>=2 and per[i+1]>=2 for i in range(len(per)-1))
def w3(per): return any(v>=3 for v in per)
cnt=dict(p8_old=0,p8_new=0,w3_old=0,w3_new=0,n=0)
for b in np.flatnonzero(nn>=5):
    if b in found: continue
    cnt['n']+=1
    po=pat(b,seedb); pn=pat(b,newb)
    cnt['p8_old']+=p8ok(po); cnt['p8_new']+=p8ok(pn)
    so=pat(b,seed); sn=pat(b,news)
    cnt['w3_old']+=w3(so); cnt['w3_new']+=w3(sn)
print('missed bodies:',cnt)
