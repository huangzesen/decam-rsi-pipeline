"""Predict disjoint nights and compare a curved orbit with chance peaks and a star."""
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import expit,logit


def confirmation(F, v, visits, hits):
    rep=v['take']
    if len(rep)<5 or 'design' not in v:return 0.
    e=v['e'][rep];sigma=np.hypot(v['errors'][rep],.070)
    xy=v['xy'][rep]/F.AS
    resid=v['resxy'][rep]/sigma[:,None]
    design=v['design'][rep]/sigma[:,None,None]
    if not np.isfinite(design).all() or not np.isfinite(resid).all() or not np.isfinite(xy).all():return 0.
    t=F.G['et_np'][e];order=np.argsort(t,kind='stable')
    # A star of any proper motion can associate the same sky positions if
    # the apparent trajectory is sufficiently straight over these visits.
    line=np.column_stack((np.ones(len(t)),(t-t.mean())/max(1.,np.ptp(t))))
    line/=sigma[:,None]
    y=(xy-xy.mean(0))/sigma[:,None]
    try:fit=np.linalg.lstsq(line,y,rcond=None)[0]
    except np.linalg.LinAlgError:return 0.
    star=.5*(np.sum((y-line@fit)**2)-np.sum(resid**2))-np.log(len(t))
    if star<=0.:return 0.
    index={int(a):k for k,a in enumerate(visits['e'])}
    rho=np.full(len(rep),.004)
    for j,a in enumerate(e):
        k=index.get(int(a))
        if k is not None and hits['covered'][k]:
            rho[j]=np.clip((hits['near'][k]+2.5)/(np.pi*40.**2),.00015,.05)
    g=v['x'][2]
    if not 1./9500.<g<1./10.:return 0.
    vc=np.sqrt(F.MU*g**3);speed=v['x'][3:]/vc
    # A broad velocity prior bounds an undetermined range rate. The prior
    # mean is zero in all three directions, including for retrograde orbits.
    A=np.zeros((3,6));A[:,2]=-1.5*speed;A[:,3:]=np.eye(3)
    prior=A.T@A+np.diag([1.e-9,1.e-9,.30,0.,0.,0.])
    gradient=A.T@speed
    # Fit at least four other nights. Alternating time ranks keep each
    # training sample spread over the arc without selecting on residuals.
    folds=max(2,int(np.ceil(len(rep)/max(1,len(rep)-4))))
    support=[]
    for j in range(folds):
        held=order[j::folds];core=np.ones(len(rep),bool);core[held]=False
        # Nearly coincident detections share foreground and detector structure.
        gap=((xy[held,None]-xy[None,core])**2).sum(axis=2).min(axis=1)
        held=held[gap>1.5**2]
        if not len(held):return 0.
        u=design[core].reshape(-1,6);b=resid[core].ravel()
        P=prior+u.T@u
        wh=design[held].reshape(-1,6)
        full=P+wh.T@wh
        try:
            dx=-np.linalg.solve(P,gradient+u.T@b)
            if abs(dx[2])>.42:return 0.
            r=resid[held].ravel()+wh@dx
            signs,ld=np.linalg.slogdet(np.stack((P,full)))
            if np.any(signs<=0):return 0.
            rhs=wh.T@r
            chi=max(0.,r@r-rhs@np.linalg.solve(full,rhs))
        except np.linalg.LinAlgError:
            return 0.
        # Full joint predictive covariance shares the uncertainty of the six
        # fitted parameters between all withheld sky positions.
        logbf=-np.log(2.*np.pi*sigma[held]**2*rho[held]).sum()-.5*(ld[1]-ld[0]+chi)
        support.append(logbf)
    return float(np.clip(min(star,min(support)),0.,70.))


def log_weight(value,nights):
    # Existing odds already use the fit and its number of nights. Reward
    # the predictive part conservatively to account for their dependence.
    return float(np.clip(.30*(value-2.-2.*max(0,nights-5)),-1.,3.))


def rank(F,outdir,deadline):
    import image_followup as I
    path=Path(outdir)/'candidates.json';reports=json.loads(path.read_text())
    if not reports:return
    G=F.G;cat=I.arrays(G['scen'])
    G['ob_np_follow']=G['ob'].cpu().numpy()
    states=dict(G.get('survey_orbits',{}));states.update(G.get('report_orbits',{}))
    sky=(G.get('survey_control') or {}).get('original_sky')
    ordered=sorted(reports,key=lambda c:-c['p'])[:300]
    for lo in range(0,len(ordered),36):
        if time.time()+.8>=deadline:break
        candidates=[]
        for c in ordered[lo:lo+36]:
            if 'holdout_weight' in c or len(c['det'])>80:continue
            trial=dict(c,_report=c);key=tuple(sorted(c['det']))
            if key in states:trial['_orbit']=states[key]
            if key in G.get('report_components',{}):trial['_components']=G['report_components'][key]
            candidates.append(trial)
        batch=I.prepare(F,candidates,cat)
        if not batch:continue
        I.model(F,batch)
        e=np.concatenate([v['e'] for v in batch]);q=np.zeros(len(e),I.QUERY)
        q['e']=e;q['x'],q['y']=np.concatenate([v['xy'] for v in batch]).T
        q['xx']=q['yy']=.20**2;q['flux']=cat['flux'][np.concatenate([v['d'] for v in batch])]
        if sky is not None:
            h=sky.inspect(q,G['mz_np']-G['cb_np'],deadline)
        else:
            h=np.zeros(len(q),I.HIT)
        start=0
        for v in batch:
            sl=slice(start,start+len(v['e']));start=sl.stop
            c=v['candidate']['_report'];odds=confirmation(F,v,q[sl],h[sl])
            shift=log_weight(odds,len(v['take']))
            c.update(holdout_odds=round(odds,3),holdout_weight=round(shift,3))
            c['p']=float(expit(logit(np.clip(c['p'],1.e-14,1.-1.e-14))+shift))
    reports.sort(key=lambda c:-c['p'])
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(reports));os.replace(tmp,path)
