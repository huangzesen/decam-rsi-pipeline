"""Sequential confirmation of an orbit constrained initially on two nights.

Adapted from gen018's compact covariance calculation. Each uncertain match is
charged against every searched exposure, refitted, and followed by independent
nights. Two-night hypotheses may require three successive associations. No
partial branch is emitted: five nights and a bound, coherent fit are required.
"""
import math,os,time
import numpy as np
import torch

def extend(F, x, memb, tref, deadline, depth=0, prior_chance=None, seed_n=4, chance_out=None):
    G=F.G; AS=F.AS; MU=F.MU; COV_C=F.COV_C
    log=F.log
    COVEXT_SIG=F.COVEXT_SIG; COVEXT_LR=F.COVEXT_LR; COVEXT_MAX=F.COVEXT_MAX
    COV3SPAN=2.; COVEXT_MAX3=1000.; COVSEP=1.; COVSEP_FLOOR=.3
    members=F.members; kres=F.kres; kjac=F.kjac; gd_prior=F.gd_prior
    arc_phot=F.arc_phot; kmodel=F.kmodel; cfit=F.cfit
    dev = G['dev']; et, ob, ne = G['et'], G['ob'], G['night_e']
    n, nE = memb.shape
    if time.time() >= deadline or 'box' not in G:
        return x, memb, 0
    # Limit this extra search to incomplete, reasonably fitting orbits. Complete
    # orbits already receive the ordinary all-epoch extension.
    ix, mm, w, dt, obs, thm = members(memb, tref)
    rr = kres(x, dt, obs, thm, w)
    rms = torch.sqrt((rr * rr).sum((1, 2)) / w.sum(1).clamp(min=1)) / AS
    ns = torch.sort(torch.where(w > 0, ne[ix], -torch.ones_like(ix)), dim=1).values
    nnight = (ns[:, 0] >= 0).long() + ((ns[:, 1:] >= 0) & (ns[:, 1:] != ns[:, :-1])).sum(1)
    span = torch.where(w > 0, dt, -1e9).max(1).values - torch.where(w > 0, dt, 1e9).min(1).values
    sel = torch.nonzero((nnight == seed_n + depth) & (rms < 0.45) & (span >= (COV3SPAN if seed_n == 3 else float(os.environ.get("FINDER_COVSPAN", "16")) if seed_n == 4 else 16.0)), as_tuple=True)[0]
    if not len(sel):
        return x, memb, 0
    xs=x[sel]; ms=memb[sel]; refs=tref[sel]
    ix, mm, w, dt, obs, thm = members(ms, refs)
    r0 = kres(xs, dt, obs, thm, w)
    J = kjac(xs, dt, obs, thm, w, r0, range(6))
    N = torch.einsum('nkci,nkcj->nij', J, J)
    N[:, 5, 5] += gd_prior(xs)
    scale = N.diagonal(dim1=1,dim2=2).sqrt().clamp(min=1e-30)
    normal = N / scale[:, :, None] / scale[:, None, :]
    C = torch.linalg.pinv(normal, hermitian=True, rtol=1e-10) / scale[:, :, None] / scale[:, None, :]
    if seed_n <= 3:
        short = (span[sel] < 16.0) | (seed_n == 2)
        if bool(short.any()):
            # A short arc, or two nights containing a close pair, can leave
            # an almost unconstrained range mode. A
            # pseudoinverse would give that mode zero variance, producing a
            # falsely precise extrapolation. Retain a large variance instead;
            # the ordinary prediction-width gate then limits the search.
            lam, vec = torch.linalg.eigh(normal[short])
            floor = lam.abs().amax(1, keepdim=True).clamp(min=1e-30) * 1e-12
            inv = (vec / torch.maximum(lam, floor)[:, None, :]) @ vec.transpose(1, 2)
            C[short] = inv / scale[short, :, None] / scale[short, None, :]
    sigma = torch.maximum(torch.full_like(rms[sel], COVEXT_SIG), rms[sel] / math.sqrt(2)) * AS
    C *= sigma[:,None,None] ** 2
    med, msig, lev, snmed = arc_phot(mm)
    # A histogram estimates the density of unrelated detections with compatible
    # photometry in each exposure. Use the measured occupied area, including on
    # sky cuts, rather than assuming a full camera footprint.
    if 'covext_hist' not in G:
        mag=np.asarray(G['mag_np'])[:G['nk']]; de=np.asarray(G['de_np'])
        bins=np.clip(np.nan_to_num(np.floor((mag-10)*4),nan=79),0,79).astype(np.int64)
        hist=np.bincount(de*80+bins,minlength=nE*80).reshape(nE,80)
        area=np.bincount(G['cov'][1] >> 42,minlength=nE) * (COV_C/AS)**2
        area=np.maximum(area,3600.)
        G['covext_hist']=torch.as_tensor(np.pad(np.cumsum(hist,axis=1),((0,0),(1,0))),device=dev)
        G['covext_area']=torch.as_tensor(area,device=dev)
    hist=G['covext_hist']; area=G['covext_area']
    m=len(sel); tt=et[None]-refs[:,None]
    pred=kmodel(xs,tt,ob[None].expand(m,nE,3))
    # Most orbit/exposure pairs cannot provide an independent observation:
    # either the field is elsewhere or this night already belongs to the arc.
    # Evaluate the expensive six orbit derivatives only for the remaining pairs.
    near=((pred-G['pt'][None])**2).sum(-1)<math.radians(1.55)**2
    member_nights=torch.where(w > 0, ne[ix], -torch.ones_like(ix))
    eligible=(ms<0)&near
    eligible &= ~(ne[None,:,None] == member_nights[:,None,:]).any(-1)
    ci,ei=torch.nonzero(eligible,as_tuple=True)
    if not len(ci):return x,memb,0
    JP=kjac(xs[ci],tt[ci,ei,None],ob[ei,None],pred[ci,ei,None],
            torch.ones((len(ci),1),dtype=torch.float64,device=dev),
            torch.zeros((len(ci),1,2),dtype=torch.float64,device=dev),range(6))[:,0]
    S=torch.einsum('nci,nij,ndj->ncd',JP,C[ci],JP)
    S=S+(0.22*AS)**2*torch.eye(2,dtype=torch.float64,device=dev)
    del JP
    R=G['box'].R
    SR=torch.einsum('ai,nij,bj->nab',R,S,R)
    width=4.0*SR.diagonal(dim1=-2,dim2=-1).clamp(min=0).sqrt()
    max_width = (COVEXT_MAX3 if seed_n == 3 else COVEXT_MAX) if seed_n < 5 else max(COVEXT_MAX, float(os.environ.get("FINDER_COV5MAX", "600")))
    narrow=(width.max(-1).values<max_width*AS)&torch.isfinite(width).all(-1)
    eligible[ci[~narrow],ei[~narrow]]=False
    ci,ei,S,width=ci[narrow],ei[narrow],S[narrow],width[narrow]
    if not len(ci):return x,memb,0
    dmwin=0.75+1.5*msig
    blo=torch.floor((med-dmwin-10)*4).long().clamp(0,79)
    bhi=(torch.floor((med+dmwin-10)*4).long()+1).clamp(1,80)
    rho=(hist[ei,bhi[ci]]-hist[ei,blo[ci]]).double()/area[ei]
    detS=torch.linalg.det(S).clamp(min=(0.1*AS)**4)
    # Expected unrelated hits across the entire family of searched positions.
    hazard=torch.zeros(m,dtype=torch.float64,device=dev)
    hazard.index_add_(0,ci,math.pi*16.0*torch.sqrt(detS)/AS**2*rho)
    # Likelihood ratio at the centre of each Gaussian prediction, against the
    # measured background density. A far, diffuse prediction cannot produce as
    # strong a match as a precise prediction merely by landing within one sigma.
    # Keep the exposure dimension so the trial correction includes every query.
    lrscale = torch.zeros((m,nE),dtype=torch.float64,device=dev)
    lrscale[ci,ei] = (2.0*math.pi*torch.sqrt(detS)/AS**2*rho).clamp(min=1e-15)
    rp=pred[ci,ei]@R.T
    ncol = max(24, int(math.ceil(2*max_width*AS/G['box'].S))+2)
    qi,hd=G['box'].query(ei,rp-width,rp+width,maxc=ncol)
    if not len(qi):return x,memb,0
    hc,he=ci[qi],ei[qi]
    delta=G['th'][hd]-pred[hc,he]
    invS=torch.linalg.inv(S[qi])
    d2=torch.einsum('ni,nij,nj->n',delta,invS,delta)
    phot=(G['mag'][hd]-med[hc]).abs()
    good=(d2<16)&(phot<dmwin[hc])&torch.isfinite(d2)
    hc,he,hd,d2=hc[good],he[good],hd[good],d2[good]
    if not len(hd):return x,memb,0
    # Keep a few hypotheses per orbit; a second independent new night is sought
    # after the exact fit, rather than concatenating unrelated ellipse hits.
    rank=d2+2.0*phot[good]**2
    if COVEXT_LR > 0:
        rank = rank + 2.0*torch.log(lrscale[hc,he].clamp(min=1e-15))
    orders = [torch.argsort(rank).cpu().numpy()]
    if COVEXT_LR > 0:
        # Preserve a geometrically close alternative when the most likely hit
        # instead comes from a particularly sparse exposure. Exact orbit fits
        # decide between them; the chance calculation still includes all hits.
        orders.append(torch.argsort(d2+2.0*phot[good]**2).cpu().numpy())
    hcc=hc.cpu().numpy(); selected=[]; selected_set=set()
    nhyp = int(os.environ.get("FINDER_COVHYP", 2 if seed_n >= 4 else 4))
    for order in orders:
        used={}
        for j in order:
            c=int(hcc[j])
            if used.get(c,0)>=nhyp:continue
            used[c]=used.get(c,0)+1
            if int(j) not in selected_set:
                selected.append(int(j));selected_set.add(int(j))
    jj=torch.as_tensor(selected,dtype=torch.long,device=dev)
    pc=hc[jj]; pe=he[jj]; pd=hd[jj]
    newm=ms[pc].clone();newm[torch.arange(len(pc),device=dev),pe]=pd
    newx,newm,res=cfit(xs[pc].clone(),newm,refs[pc],8,clip=None)
    # Test the orbit on every other exposure with a small, fixed aperture. A
    # second new night is evidence independent of the uncertain extrapolation.
    pp=kmodel(newx,et[None]-refs[pc,None],ob[None].expand(len(pc),nE,3))
    qa,qb=torch.nonzero(newm<0,as_tuple=True)
    hits=G['index'].query(qb,pp[qa,qb],1.0*AS)
    hh=hits.clamp(min=0)
    kg=(hits>=0)&((G['mag'][hh]-med[pc[qa]]).abs()<dmwin[pc[qa]])
    newm[qa[kg],qb[kg]]=hits[kg]
    sep2=((G['th'][hh]-pp[qa,qb])**2).sum(-1)/AS**2
    newx,newm,res=cfit(newx,newm,refs[pc],6,clip=0.7)
    hitn=(newm>=0).sum(1).clamp(min=1)
    fitr=torch.sqrt((res*res).sum(1)/hitn)/AS
    # The first uncertain hit changes the predicted orbit. Its precise scan
    # queried every exposure, including epochs excluded from the original
    # covariance envelope. Count its new nights and its trial area on that
    # actual scan. The first hit's night already has its own chance term.
    independent = (ms[pc] < 0) & ~((ne[None,:,None] == member_nights[pc,None,:]).any(-1))
    precise_support = independent & (((pp-G['pt'][None])**2).sum(-1) < math.radians(1.55)**2)
    follow_support = precise_support & (ne[None] != ne[pe,None])
    follow_rho = (hist[:,bhi[pc]].T-hist[:,blo[pc]].T).double()/area[None]
    follow_hazard = math.pi*(follow_rho*follow_support).sum(1)
    newnight=torch.where((newm>=0)&precise_support,ne[None],-torch.ones_like(newm))
    ns=torch.sort(newnight,dim=1).values
    nnew=(ns[:,0]>=0).long()+((ns[:,1:]>=0)&(ns[:,1:]!=ns[:,:-1])).sum(1)
    retained=((ms[pc]<0)|(newm==ms[pc])).all(1)
    bound=(newx[:,3:6]**2).sum(1)<1.98*MU*newx[:,2].clamp(min=0)**3
    # One uncertain hit is accepted only where such a hit is rare anywhere along
    # the searched track. Multiple new nights must survive the same exact orbit.
    # Use the observed Mahalanobis separation, not the outer search aperture,
    # when computing the probability of an equally close accidental match.
    match_chance=hazard[pc]*d2[jj].clamp(min=1.0)/16.0
    if COVEXT_LR > 0:
        # Integrate the background over the regions in ALL searched exposures
        # whose Gaussian/background likelihood ratio exceeds this hit's. For
        # exposure e the region is an ellipse with squared radius
        # 2(log L0_e - log L_hit), truncated at the actual four-sigma search.
        # Its expected count is (scale_e / 2) times that squared radius.
        # Use the observed separation, measured BEFORE adding this detection
        # to the fit. Even a centred hit in a diffuse prediction has competing
        # area in more precise exposures; no artificial sigma floor is needed.
        log_lhit = -0.5*d2[jj]-torch.log(lrscale[pc,pe])
        scales = lrscale[pc]
        radius2 = (-2.0*torch.log(scales.clamp(min=1e-15))-2.0*log_lhit[:,None]).clamp(0.0,16.0)
        match_chance = (0.5*scales*radius2).sum(1)
    if os.environ.get('FINDER_COVMOTION','1') != '0':
        from motion_evidence import repeat_probability
        match_chance *= repeat_probability(F,newm,res,pe,blo[pc],bhi[pc])
    extra=(nnew-1).clamp(min=0).double()
    follow_area=torch.ones(len(pc),dtype=torch.float64,device=dev)
    if COVSEP > 0:
        # `follow_hazard` is the number of unrelated detections expected within the whole 1" aperture over all exposures.
        # A body's further nights lie within its astrometric error of the track, a chance detection anywhere in the
        # aperture (density ~ r dr): the expected number at least as near as the observed ones is the hazard times the
        # squared separation (arcsec; the farthest of the new nights, never below COVSEP_FLOOR).  The test stays a
        # tail probability under the chance hypothesis, so the limit means what it meant.
        try:
            still=kg&(newm[qa,qb]==hits)&follow_support[qa,qb]
            worst=torch.zeros(len(pc),dtype=torch.float64,device=dev)
            worst.scatter_reduce_(0,qa[still],sep2[still].double(),reduce='amax',include_self=True)
            follow_area=worst.clamp(min=COVSEP_FLOOR**2,max=1.0)
        except Exception as ex:
            log("separation hazard failed:",type(ex).__name__,str(ex)[:200])
            follow_area=torch.ones(len(pc),dtype=torch.float64,device=dev)
    joint_chance=match_chance*(follow_hazard.clamp(min=1e-12)*follow_area)**extra/torch.exp(torch.lgamma(extra+1))
    if prior_chance is not None:
        joint_chance = joint_chance * prior_chance[sel[pc]]
    chance_limit = (0.003 if seed_n >= 5 else 0.03) if seed_n >= 4 else float(os.environ.get("FINDER_COV3CHANCE", "0.0003"))
    acc=retained&bound&(fitr<0.35)&(nnew>=max(1,5-seed_n-depth))&(joint_chance<chance_limit)
    if depth < max(1,4-seed_n) and time.time() < deadline:
        pending = torch.nonzero(~acc & retained & bound & (fitr < 0.35) & (nnew >= 1) & (nnew < 5-seed_n-depth) & (joint_chance < 1.0),as_tuple=True)[0]
        for advance in range(1,max(2,5-seed_n-depth)):
            take=pending[nnew[pending]==advance]
            if not len(take):continue
            recursive_chance=torch.full((len(take),),float('inf'),dtype=torch.float64,device=dev)
            rx,rm,ra=extend(F, newx[take].clone(),newm[take].clone(),refs[pc[take]],deadline,
                depth=depth+advance,prior_chance=2.0*joint_chance[take],seed_n=seed_n,chance_out=recursive_chance)
            changed=(rm!=newm[take]).any(1)
            if bool(changed.any()):
                dest=take[changed];newx[dest]=rx[changed];newm[dest]=rm[changed];acc[dest]=True
                joint_chance[dest]=recursive_chance[changed]
                rn=torch.sort(torch.where(newm[dest]>=0,ne[None],-torch.ones_like(newm[dest])),dim=1).values
                total_n=(rn[:,0]>=0).long()+((rn[:,1:]>=0)&(rn[:,1:]!=rn[:,:-1])).sum(1)
                nnew[dest]=total_n-seed_n-depth
                G['cov_cascade']=G.get('cov_cascade',0)+int(changed.sum())
    accepted=0; chosen=[]; destinations=[]; done=set()
    order=torch.argsort(nnew.double()-fitr,descending=True).cpu().numpy()
    acc_np=acc.cpu().numpy();pc_np=pc.cpu().numpy();sel_np=sel.cpu().numpy()
    for k in order:
        dest=int(sel_np[pc_np[k]])
        if acc_np[k] and dest not in done:
            chosen.append(int(k));destinations.append(dest);done.add(dest)
    if chosen:
        src=torch.as_tensor(chosen,device=dev);dst=torch.as_tensor(destinations,device=dev)
        x[dst]=newx[src];memb[dst]=newm[src];accepted=len(chosen)
        if chance_out is not None:
            chance_out[dst] = joint_chance[src]
        trace=G.setdefault('covext_trace',[])
        trace_rows=G.get('covext_trace_rows',0)
        for row in newm[src].cpu().numpy():
            ids=row[row>=0]
            if len(trace)>=20000 or trace_rows+len(ids)>500000:break
            trace.append(G['keep_np'][ids].tolist());trace_rows+=len(ids)
        G['covext_trace_rows']=trace_rows
    G['covext_stat']=G.get('covext_stat',0)+accepted
    if os.environ.get('FINDER_DEBUG2'):
        log('covariance extension',len(sel),'orbits',len(hd),'hits',accepted,'accepted', 'hazard',hazard.detach().cpu().tolist(), 'match_chance', match_chance.detach().cpu().tolist(), 'd2', d2[jj].detach().cpu().tolist())
    return x,memb,accepted
