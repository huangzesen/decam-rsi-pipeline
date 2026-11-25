import json, numpy as np, sys
dst='/tmp/p2'
T=np.load(dst+'/truth.npz'); E=json.load(open(dst+'/exposures.json')); et=np.array([e['mjd_mid'] for e in E]); night=np.floor(et-0.5).astype(int)
Z=np.load('/tmp/w/cache_p2.npz'); keep=Z['keep']
ntot=int(T['ntot']); kept=np.zeros(ntot,bool); kept[keep]=True
rows,body,ex=T['rows'],T['body'],T['exp']; nn=T['nights']
found=np.zeros(len(nn),bool); found[np.load('/tmp/found.npy')]=True
sc=np.flatnonzero(nn>=5)
o=np.argsort(body,kind='stable'); bs=body[o]; st=np.searchsorted(bs,sc); en=np.searchsorted(bs,sc,side='right')
def maxwin(nts,W):
    nts=np.unique(nts); return max(np.searchsorted(nts,nts+W,side='right')-np.arange(len(nts)))
tab={}
for b,a,c in zip(sc,st,en):
    i=o[a:c]; nt=np.unique(night[ex[i]][kept[rows[i]]])
    if len(nt)==0: continue
    n=len(nt); n64=maxwin(nt,64); n130=maxwin(nt,130)
    # second-best cluster: nights outside the best 130-day window
    k=np.argmax(np.searchsorted(nt,nt+130,side='right')-np.arange(len(nt))); inw=(nt>=nt[k])&(nt<=nt[k]+130); rest=nt[~inw]
    r2=maxwin(rest,130) if len(rest) else 0
    key=(min(n,8), min(n64,5), min(n130,5), min(r2,3))
    t=tab.setdefault(key,[0,0]); t[0]+=1; t[1]+=found[b]
print('nights(cap8) n64(cap5) n130(cap5) rest130(cap3): total found')
for k in sorted(tab): 
    if tab[k][0]>=8: print(k, tab[k], '%.2f'%(tab[k][1]/tab[k][0]))
# marginals
for name,idx in (('n64',1),('n130',2)):
    m={}
    for k,v in tab.items():
        if k[0]<=7:
            t=m.setdefault(k[idx],[0,0]); t[0]+=v[0]; t[1]+=v[1]
    print(name,'for nights 5-7:',{k:(v[0],v[1]) for k,v in sorted(m.items())})
