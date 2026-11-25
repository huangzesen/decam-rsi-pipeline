import json,sys,numpy as np
sys.path.insert(0,'$RSI_SANDBOX/r1/gen001/workspace/submission')
from orbit import predict_dirs
tr=json.load(open(sys.argv[1]));c=json.load(open(sys.argv[2]))['candidates']
import csv
rows=list(csv.DictReader(open(sys.argv[3]+'/exposures.csv')))
et=np.array([float(r['mjd_mid']) for r in rows]);eo=np.array([[float(r['obs_x']),float(r['obs_y']),float(r['obs_z'])] for r in rows])
found=0
for t in tr:
    dt=predict_dirs(np.array(t['state']),t['epoch'],et,eo)
    best=None
    for i,cc in enumerate(c):
        dc=predict_dirs(np.array(cc['state']),cc['epoch_mjd'],et,eo)
        sep=np.degrees(np.arccos(np.clip((dt*dc).sum(1),-1,1)))*3600
        if (sep<2).sum()>=3: best=(i,cc['p']);break
    print(round(t['R'],1),round(t.get('snr',0),1),t.get('ndet'),best); found+=best is not None
print('found',found,len(tr),'ncand',len(c),'p>0.5',sum(x['p']>0.5 for x in c))
if len(sys.argv)>4:
    F=json.load(open(sys.argv[4]))
    for t in tr:
        dt=predict_dirs(np.array(t['state']),t['epoch'],et,eo)
        for f in F:
            dc=predict_dirs(np.array(f['state']),t['epoch'],et,eo)
            sep=np.degrees(np.arccos(np.clip((dt*dc).sum(1),-1,1)))*3600
            if (sep<5).sum()>=2: print(round(t['R'],1),{k:(round(v,2) if isinstance(v,float) else v) for k,v in f.items() if k!='state'}, sep.round(1).tolist())
