import json, numpy as np, collections
dst='/tmp/p4'
T=np.load(dst+'/truth.npz'); E=json.load(open(dst+'/exposures.json')); mjd=np.array([e['mjd_mid'] for e in E]); night=np.floor(mjd-0.5).astype(int)
snr=np.load(dst+'/det_snr.npy',mmap_mode='r')
NB=len(T['nights']); pats=collections.Counter(); rows_all=T['rows']; bi=T['body']; ei=T['exp']
out=[]
for b in range(NB):
    if T['nights'][b]<5: continue
    m=bi==b; nb=np.unique(night[ei[m]])
    g=np.flatnonzero(np.diff(nb)>120); parts=np.split(nb,g+1); per=[len(p) for p in parts]
    if max(per)>2: continue
    s=np.asarray(snr[rows_all[m]]); 
    # seed-level nights (S/N>=6.5 / 5.5 / 5.0) per season
    def seedper(thr):
        nn=np.unique(night[ei[m]][s>=thr]); return [int(np.isin(p,nn).sum()) for p in parts]
    yrs=[int((p[0]-56658)/365.25+2014) for p in parts]
    # adjacent seasons both with 2 nights at thr
    def adj22(thr):
        sp=seedper(thr); return any(sp[i]>=2 and sp[i+1]>=2 and (parts[i+1][0]-parts[i][-1])<420 for i in range(len(sp)-1))
    out.append((b,per,yrs,round(float(np.median(s)),1),adj22(6.5),adj22(5.5),adj22(5.0)))
    pats[tuple(sorted(per,reverse=True))]+=1
print(len(out),'class-f bodies; patterns:',dict(pats))
print('with 2+2 seed nights in adjacent seasons at S/N>=6.5: %d, >=5.5: %d, >=5.0: %d'%(sum(o[4] for o in out),sum(o[5] for o in out),sum(o[6] for o in out)))
for o in out[:49]:
    if o[4] or o[5]: print(o)
