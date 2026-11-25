import json, numpy as np
T=np.load('/tmp/p2/truth.npz'); E=json.load(open('/tmp/p2/exposures.json')); et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
Z=np.load('/tmp/w/cache_p2.npz'); keep,seed=Z['keep'],Z['seed']
ntot=int(T['ntot']); kept=np.zeros(ntot,bool); kept[keep]=True; isseed=np.zeros(ntot,bool); isseed[keep[seed]]=True
snr=np.load('/tmp/p2/det_snr.npy',mmap_mode='r')
rows,body,ex=T['rows'],T['body'],T['exp']; nn=T['nights']
s=np.asarray(snr[rows])
print('planted dets',len(rows),'kept',kept[rows].mean(),'seed',isseed[rows].mean(),'snr<5',np.mean(s<5), 'kept&snr>=5 not seed', np.mean(kept[rows]&(s>=5)&~isseed[rows]))
def pat(nts, allnts, W, reach):
    # 3 seed nights within W days and a 4th (any) night within reach of the window
    nts=np.unique(nts); allnts=np.unique(allnts)
    for i in range(len(nts)-2):
        for j in range(i+2,len(nts)):
            if nts[j]-nts[i]>W: break
            # any other night within reach of the window
            oth=allnts[(allnts>=nts[i]-reach)&(allnts<=nts[j]+reach)]
            if len(oth)>=4: return 2
    for i in range(len(nts)-2):
        if nts[i+2]-nts[i]<=W: return 1
    return 0
res={}
sc=np.flatnonzero(nn>=5)
o=np.argsort(body,kind='stable'); bs=body[o]; st=np.searchsorted(bs,sc); en=np.searchsorted(bs,sc,side='right')
cnt={k:np.zeros(3,int) for k in ['seed64','any64','seed32','any32','seed16','any16']}
for b,a,c in zip(sc,st,en):
    i=o[a:c]; nt=night[ex[i]]; sd=isseed[rows[i]]; kp=kept[rows[i]]
    for W,k in ((64,'64'),(32,'32'),(16,'16')):
        cnt['seed'+k][pat(nt[sd],nt[kp],W,40)]+=1
        cnt['any'+k][pat(nt[kp],nt[kp],W,40)]+=1
print('scored',len(sc))
for k,v in cnt.items(): print(k,'none/3only/3+4th',v, 'frac 3+4th %.3f  >=3 %.3f'%(v[2]/len(sc),(v[1]+v[2])/len(sc)))
