import json,sys,numpy as np,csv
sys.path.insert(0,'$RSI_SANDBOX/r1/gen001/workspace/submission')
from orbit import predict_dirs
SDIR='$DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev/'
def feats(f):
    return [min(f['nm'],15), f['nn'], f['miss'], np.log(max(f['msn'],3)), f['fpos'], f['fnights'], f['nm_pos'], f['nm_n']-f['nm_pos'], np.clip(f['stack'],-5,60), float(f['nn']==3)]
names=['nm','nn','miss','logmsn','fpos','fnights','nm_pos','nm_neg','stack','nn3']
X=[];Y=[]
# real: injected matches
for truth,dbg,scen in [('runs/mi18.npy.truth.json','runs/dbgm18.json','scen03'),('runs/mi149.npy.truth.json','runs/dbgm149.json','scen02')]+[tuple(a.split(':')) for a in sys.argv[1:] if a.count(':')==2]:
    tr=json.load(open(truth));F=json.load(open(dbg));rows=list(csv.DictReader(open(SDIR+scen+'/exposures.csv')))
    et=np.array([float(r['mjd_mid']) for r in rows]);eo=np.array([[float(r['obs_x']),float(r['obs_y']),float(r['obs_z'])] for r in rows])
    T=[predict_dirs(np.array(t['state']),t['epoch'],et,eo) for t in tr]; ep=tr[0]['epoch']
    n=0
    for f in F:
        if f.get('fpos') is None: continue
        dc=predict_dirs(np.array(f['state']),ep,et,eo)
        if any((np.degrees(np.arccos(np.clip((d*dc).sum(1),-1,1)))*3600<2).sum()>=3 for d in T):
            X.append(feats(f));Y.append(1);n+=1
    print(scen,'real',n)
for nf in [a for a in sys.argv[1:] if a.count(':')==0]:
    F=json.load(open(nf)); n=0
    for f in F:
        if f.get('fpos') is None: continue
        X.append(feats(f));Y.append(0);n+=1
    print(nf,'junk',n)
X=np.array(X,float);Y=np.array(Y,float)
mu=X.mean(0);sd=X.std(0)+1e-9
Z=(X-mu)/sd; Z=np.c_[np.ones(len(Z)),Z]
# weight classes equally-ish then fix intercept to a prior
w=np.zeros(Z.shape[1]); lam=1.0
cw=np.where(Y==1, 0.5/Y.mean(), 0.5/(1-Y.mean()))
for it in range(200):
    p=1/(1+np.exp(-Z@w)); g=Z.T@(cw*(p-Y))+lam*np.r_[0,w[1:]]
    H=(Z*(cw*p*(1-p))[:,None]).T@Z+lam*np.diag(np.r_[0,np.ones(len(w)-1)])
    w-=np.linalg.solve(H,g)
p=1/(1+np.exp(-Z@w))
# convert to raw-feature weights
wr=w[1:]/sd; b=w[0]-np.sum(w[1:]*mu/sd)
print('weights',dict(zip(names,np.round(wr,3))),'bias',round(b,3))
for thr in [0.5,0.7,0.9,0.97]:
    print(thr,'real kept',(p[Y==1]>thr).mean().round(3),'junk kept',(p[Y==0]>thr).sum(),'of',(Y==0).sum())
json.dump(dict(w=wr.tolist(),b=float(b),names=names),open('model.json','w'))
# current hand p on same data
P=[];YY=[]
