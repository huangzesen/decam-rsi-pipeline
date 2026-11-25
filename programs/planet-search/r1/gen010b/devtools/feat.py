import json,sys,numpy as np,csv
sys.path.insert(0,'$RSI_SANDBOX/r1/gen001/workspace/submission')
from orbit import predict_dirs
tr=json.load(open(sys.argv[1]));F=json.load(open(sys.argv[2]));rows=list(csv.DictReader(open(sys.argv[3]+'/exposures.csv')))
et=np.array([float(r['mjd_mid']) for r in rows]);eo=np.array([[float(r['obs_x']),float(r['obs_y']),float(r['obs_z'])] for r in rows])
T=[predict_dirs(np.array(t['state']),t['epoch'],et,eo) for t in tr]
ep=tr[0]['epoch']
real=[]
for f in F:
    dc=predict_dirs(np.array(f['state']),ep,et,eo)
    ok=any((np.degrees(np.arccos(np.clip((d*dc).sum(1),-1,1)))*3600<2).sum()>=3 for d in T)
    real.append(ok)
real=np.array(real)
keys=['p','nm','nn','miss','msn','rms','stack','n_on','nm_pos','nm_n','fpos','fmed','fnights']
print('REAL');
for f in sorted([f for f,r in zip(F,real) if r],key=lambda f:-f['stack'])[:30]: print({k:round(f[k],2) for k in keys})
print('JUNK top by stack')
for f in sorted([f for f,r in zip(F,real) if not r],key=lambda f:-f['stack'])[:25]: print({k:round(f[k],2) for k in keys})
