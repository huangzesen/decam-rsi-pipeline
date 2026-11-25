"""Give seasonal quads an independent fifth-night check before their queue cap."""
import time
import numpy as np
import torch
import fifth_night


def promote(F,found,deadline):
    G=F.G;dev=G['dev'];AS=F.AS
    done=fifth_night.verify(found,G,F.MU,AS,time.time()+.45*max(deadline-time.time(),0.))
    proposals=[];parents=[]
    for c in found.values():
        hits=c.get('x5',[])
        if not hits:continue
        d=np.asarray(c['det'],dtype=np.int64)
        med=float(np.nanmedian(G['mag_np'][d]))
        for h in hits:
            if abs(float(G['mag_np'][h])-med)<=.9:
                proposals.append(np.r_[d,int(h)]);parents.append(c)
    result={};tested=0
    for start in range(0,len(proposals),2048 if dev!='cpu' else 512):
        if time.time()>=deadline:break
        cs=parents[start:start+2048] if dev!='cpu' else parents[start:start+512]
        p=torch.as_tensor(np.array(proposals[start:start+len(cs)]),device=dev)
        x=torch.tensor([[c['st'][0],c['st'][1],c['g'],c['st'][2],c['st'][3],c.get('gd',0.)] for c in cs],dtype=torch.float64,device=dev)
        tref=torch.tensor([c['tref'] for c in cs],dtype=torch.float64,device=dev)
        ee=G['de'][p];dt=G['et'][ee]-tref[:,None]
        x,res=F.kfit(x,dt,G['ob'][ee],G['th'][p],torch.ones_like(dt),iters=8)
        rms=res.square().mean(1).sqrt()/AS
        radius=(1+x[:,:2].square().sum(1)).sqrt()
        bound=x[:,3:6].square().sum(1)*radius < 2*F.MU*x[:,2].clamp(min=0)**3
        ok=(rms<.32)&(res.max(1).values<.65*AS)&(x[:,2]>0)&bound&torch.isfinite(x).all(1)
        tested+=len(cs)
        for j in torch.nonzero(ok,as_tuple=True)[0].cpu().tolist():
            c=cs[j];key=tuple(c['det'].tolist());xx=x[j].cpu().numpy();rr=float(rms[j])
            if key not in result or rr<result[key]['rms']:
                result[key]=dict(c,det=np.sort(p[j].cpu().numpy()),nights=5,rms=rr,
                    st=xx[[0,1,3,4]].tolist(),g=float(xx[2]),gd=float(xx[5]),f5=1)
        # Preserve queue preference even when the exact confirmation did not
        # converge: the ordinary orbit growth can still resolve that quad.
    F.log('seasonal confirmation:',done,'quads checked,',len(proposals),'fifth-night hypotheses,',tested,'fitted,',len(result),'confirmed quads')
    return result
