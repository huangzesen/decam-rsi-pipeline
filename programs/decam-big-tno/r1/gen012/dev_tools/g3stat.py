import pickle, numpy as np, json, sys, math
D='/tmp/p4'; Z=np.load('/tmp/w/cacheB.npz'); keep=Z['keep']; de=Z['de']; snr=Z['snr_k']; flux=Z['flux']
E=json.load(open(D+'/exposures.json')); et=np.array([e['mjd_mid'] for e in E])
mz=np.array([e.get('magzero') if e.get('magzero') is not None else np.nan for e in E],float)
col=dict(g=0.6,r=0.0,i=-0.25,z=-0.4,Y=-0.45,VR=0.1); cb=np.array([col.get(e.get('band'),0.0) for e in E])
mag=mz[de]-2.5*np.log10(np.maximum(flux,1e-9))-cb[de]
T=np.load(D+'/truth.npz'); body=-np.ones(int(T['ntot']),np.int64); body[T['rows']]=T['body']; bk=body[keep]; nn=T['nights']
g3=pickle.load(open(sys.argv[1],'rb'))
print('3-night linkages',len(g3), 'keys',sorted(g3[0].keys()))
rows=[]
for c in g3:
    d=np.asarray(c['kdet']); b=bk[d]
    pl = (b>=0).all() and (b==b[0]).all()
    sc = bool(pl and nn[b[0]]>=5)
    t=et[de[d]]; arc=t.max()-t.min(); sn=np.sort(snr[d])[-3:]
    rexp=math.sqrt(2*np.mean((1.1/np.maximum(snr[d],4.5))**2+0.04**2))
    m=mag[d]; sg=1.09/np.maximum(snr[d],4.5)
    msp=(np.nanmax(m)-np.nanmin(m))/math.sqrt(2*np.mean(sg**2)) if np.isfinite(m).sum()>=2 else 0
    q=min(arc,64.0)/24.0+float(np.log(np.maximum(sn,4.5)/4.5).sum())
    rows.append((sc,pl,arc,float(np.log(np.maximum(sn,4.5)/4.5).sum()),c['rms'],rexp,msp,q,len(d),c.get('h',0)))
R=np.array(rows,float); sc=R[:,0]>0
print('scored planted arcs',int(sc.sum()),'other planted',int((R[:,1]>0).sum()-sc.sum()),'chance',int((R[:,1]==0).sum()))
def show(nm,v):
    print(f'  {nm}: planted pct 10/50/90 {np.percentile(v[sc],[10,50,90]).round(3)}  chance {np.percentile(v[~sc],[10,50,90]).round(3)}')
show('arc',R[:,2]); show('snr term',R[:,3]); show('rms',R[:,4]); show('rexp',R[:,5]); show('rms/rexp',R[:,4]/R[:,5]); show('mag spread / sigma',R[:,6]); show('ndet',R[:,8])
def rank(q,lab):
    o=np.argsort(-q); cs=np.cumsum(sc[o]); n=len(q)
    print(f'  {lab}: planted among the first 5/10/20/40 % of the queue: '+' '.join(f'{cs[int(n*f)-1]/sc.sum():.3f}' for f in (0.05,0.1,0.2,0.4)))
q=R[:,7]; x=R[:,4]/R[:,5]
rank(q,'q (now)')
for k in (0.5,1.0,1.5,2.0,3.0):
    rank(q-k*np.minimum(x,3.0),f'q - {k} x min(rms/rexp, 3)')
for k in (1.0,2.0):
    rank(q-k*np.minimum(x,3.0)-0.5*np.minimum(R[:,6],4),f'q - {k} rms/rexp - 0.5 magspread')
rank(-x,'by rms/rexp alone')
rank(-R[:,4],'by rms alone')
np.save('/tmp/w/g3R.npy',R)
