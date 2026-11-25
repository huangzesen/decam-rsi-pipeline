import numpy as np, json, pickle, sys
T=np.load('/tmp/p5/truth.npz'); E=json.load(open('/tmp/p5/exposures.json')); nE=len(E)
et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
C=np.load('/tmp/w/cache_p5.npz'); keep=C['keep']; de=C['de']; seed=C['seed'].astype(bool); seed2=C['seed2'].astype(bool); snr=C['snr_k']
ntot=int(T['ntot']); bod=-np.ones(ntot,int); bod[T['rows']]=T['body']; bk=bod[keep]
found=set(np.load(sys.argv[1]).tolist()); nn=T['nights']
rows,body,exp=T['rows'],T['body'],T['exp']
keptm=np.zeros(ntot,bool); keptm[keep]=True
seedm=np.zeros(ntot,bool); seedm[keep[seed]]=True
snr_all=np.load('/tmp/p5/det_snr.npy',mmap_mode='r')
# grown cache: linkages per body
gr=pickle.load(open('/tmp/w/f1/grown.pkl','rb'))
best={}
for v in gr.values():
    if v is None: continue
    b=bk[v['det']]; bb=b[b>=0]
    if len(bb)==0: continue
    u,cn=np.unique(bb,return_counts=True); j=int(u[cn.argmax()])
    nt=len(np.unique(night[de[v['det']][b==j]]))
    pure=cn.max()/len(b)
    if nt>best.get(j,(0,0))[0]: best[j]=(nt,round(float(pure),2))
out={}
for b in np.flatnonzero(nn>=5):
    if b in found: continue
    m=body==b; e=exp[m]; r=rows[m]
    nb=np.unique(night[e]); g=np.flatnonzero(np.diff(nb)>120); parts=np.split(nb,g+1)
    per=sorted([len(p) for p in parts],reverse=True)
    far2=any(v>=2 for v in per[1:])
    cls='a' if per[0]>=5 else ('b' if per[0]==4 and far2 else None)
    if cls is None: continue
    big=max(parts,key=len)
    inb=np.isin(night[e],big)
    nk=len(np.unique(night[e][inb&keptm[r]])); ns=len(np.unique(night[e][inb&seedm[r]]))
    span=big.max()-big.min()
    out.setdefault(cls,[]).append((b,per,nk,ns,int(span),round(float(np.median(snr_all[r])),1),best.get(int(b))))
for cls in out:
    print(cls,len(out[cls]))
    for o in out[cls][:60]: print('   body %d per %s kept-nights(best season) %d seed-nights %d span %d medSNR %.1f best grown linkage (nights,purity) %s'%o)
