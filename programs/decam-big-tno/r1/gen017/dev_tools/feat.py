"""feat.py NAME [NAME...]: per-candidate features of runs in /tmp/o/NAME (planted: class T/M/F by truth; unplanted 'u*': U)."""
import numpy as np, json, sys, pickle
E=json.load(open('/tmp/p3/exposures.json')); mjd=np.array([e['mjd_mid'] for e in E]); night=np.floor(mjd-0.5).astype(int)
T=np.load('/tmp/p3/truth.npz'); ntot=int(T['ntot']); body=-np.ones(ntot,np.int64); body[T['rows']]=T['body']
def load(name):
    unpl=name.startswith('u')
    scen='/app/data/sample' if unpl else '/tmp/p3'
    exp=np.load(scen+'/det_exp.npy',mmap_mode='r'); snr=np.load(scen+'/det_snr.npy',mmap_mode='r')
    C=json.load(open(f'/tmp/o/{name}/candidates.json')); out=[]
    for c in C:
        d=np.array(c['det']); e=np.asarray(exp[d]); s=np.asarray(snr[d]); nt=night[e]
        if unpl: cl='U'
        else:
            b=body[d]; npl=(b>=0).sum()
            cl='F' if npl==0 else 'M'
            if npl:
                u,cnt=np.unique(b[b>=0],return_counts=True); j=u[cnt.argmax()]
                if cnt.max()>=0.8*len(d) and len(np.unique(nt[b==j]))>=5: cl='T'
        un=np.unique(nt); g=np.flatnonzero(np.diff(un)>120); per=sorted([len(p) for p in np.split(un,g+1)],reverse=True)
        # best S/N per night
        bn=np.array([s[nt==n].max() for n in un])
        out.append(dict(c,cl=cl,per=per,bn=np.sort(bn)[::-1],snr=s,name=name))
    return out
if __name__=='__main__':
    R=[]
    for n in sys.argv[1:]: R+=load(n)
    pickle.dump(R,open('/tmp/w/feat.pkl','wb'))
    print(len(R))
