import numpy as np,sys,csv,json,math
sys.path.insert(0,'$RSI_SANDBOX/r1/gen001/workspace/submission')
from orbit import radec_to_unit, predict_dirs, GM
from forced import Forced
from detect import gnomonic, poly2
S=sys.argv[1]; src=sys.argv[2]; out=sys.argv[3]; nobj=int(sys.argv[4]); smin=float(sys.argv[5]); smax=float(sys.argv[6])
SD='$DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev/'+S
res=np.load(src,allow_pickle=True).item()
rows=list(csv.DictReader(open(SD+'/exposures.csv')));meta=json.load(open(SD+'/meta.json'))
et=np.array([float(r['mjd_mid']) for r in rows]);eo=np.array([[float(r['obs_x']),float(r['obs_y']),float(r['obs_z'])] for r in rows])
t0=0.5*(et.min()+et.max())
rng=np.random.default_rng(2);truth=[]
for k in range(nobj):
    R=rng.uniform(30,80); snr=rng.uniform(smin,smax)
    ra=np.radians(meta['ra']+rng.uniform(-0.8,0.8));de=np.radians(meta['dec']+rng.uniform(-0.8,0.8))
    o=eo[np.argmin(abs(et-t0))]
    u=radec_to_unit(ra,de);ou=o@u;rho=-ou+np.sqrt(ou**2-o@o+R*R);r=o+rho*u
    z=np.array([0,-0.3978,0.9175]);v=np.cross(z,r);v/=np.linalg.norm(v);v*=np.sqrt(GM/R)*rng.uniform(0.85,1.1)
    st=np.r_[r,v]; d=predict_dirs(st,t0,et,eo)
    rr=np.arctan2(d[:,1],d[:,0]);dd=np.arcsin(d[:,2])
    ndet=0
    for e in res:
        s=snr*rng.uniform(0.8,1.2)
        # into the map
        hit=False
        for g,m in res[e].get('maps',[]):
            xi,eta=gnomonic(np.array([rr[e]]),np.array([dd[e]]),g['rc'],g['dc'])
            A=poly2(xi,eta);x=float(A@g['cx'])/2;y=float(A@g['cy'])/2
            if 2<x<m.shape[1]-3 and 2<y<m.shape[0]-3:
                sig=math.sqrt(2)*(res[e]['fwhm']/2.355)/2
                x0,y0=int(x),int(y)
                ys,xs=np.mgrid[y0-6:y0+7,x0-6:x0+7]
                ok=(ys>=0)&(ys<m.shape[0])&(xs>=0)&(xs<m.shape[1])
                ys,xs=ys[ok],xs[ok]
                m[ys,xs]=(m[ys,xs].astype(np.float32)+s*np.exp(-((xs-x)**2+(ys-y)**2)/(2*sig*sig))).astype(np.float16)
                hit=True
        meas=s+rng.normal()
        if hit and meas>=3:
            row=np.array([[np.degrees(rr[e])%360+rng.normal(0,0.1/3600)/np.cos(dd[e]),np.degrees(dd[e])+rng.normal(0,0.1/3600),meas,0.9,1000.0,99,0,0]])
            res[e]['det']=np.r_[res[e]['det'],row]; ndet+=1
    truth.append(dict(state=st.tolist(),epoch=t0,R=R,snr=snr,ndet=ndet))
np.save(out,res,allow_pickle=True);json.dump(truth,open(out+'.truth.json','w'))
