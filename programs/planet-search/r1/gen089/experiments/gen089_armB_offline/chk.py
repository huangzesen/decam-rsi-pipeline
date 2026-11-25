import sys, numpy as np, importlib.util as iu
sys.path.insert(0,'$RSI_SANDBOX/r1/gen089/tmp/armB/offline')
import ev, json
sp=iu.spec_from_file_location('fa','$RSI_SANDBOX/r1/gen089/tmp/armB/stackd/finder.py'); fa=iu.module_from_spec(sp); sp.loader.exec_module(fa)
z=np.load(ev.D+'/chi_feats.npz',allow_pickle=True)
vC,cC,vN,cN,wf=z['vC'].astype(float),z['cC'],z['vN'].astype(float),z['cN'],z['wf']; ni=[np.asarray(i,int) for i in z['nights']]
S=z['states']; pv=z['pv']; zp=z['zp']
# mock rankx: states keyed by row; null states = fake ids
nN=len(vN); nC=len(vC)
R=np.zeros((nN,6)); R[:,0]=-1-np.arange(nN)
Sc=S.copy()
tab={}
for j in range(nC): tab[tuple(Sc[j])]=(vC[j],cC[j])
for j in range(nN): tab[tuple(R[j])]=(vN[j],cN[j])
def ss(X):
    a=[tab[tuple(x)] for x in X]; return np.array([x[0] for x in a]),np.array([x[1] for x in a])
pre=json.load(open(ev.D+'/pre.json'))['candidates']
key0={tuple(np.round(c['state'],9)):c for c in pre}
cands=[dict(key0[tuple(np.round(s,9))], state=list(s), p=float(pv[k])) for k,s in enumerate(S)]
cvz={id(c):float(zp[j]) for j,c in enumerate(cands)}
out=fa.rankx(cands,R,np.full(nN,2),np.zeros(nN)+0.0,ss,wf,ni,cvz,print)
print('new default rankx (mock, rsc null zero so zp from cvz):',ev.ev([dict(c,p=max(c['p'],c.get('pn',0))) for c in out]))
