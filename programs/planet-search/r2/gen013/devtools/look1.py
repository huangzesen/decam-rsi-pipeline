import sys, pickle, json, numpy as np
for s in ["scen01","scen02","scen03","scen04"]:
    D = pickle.load(open(f"run/{sys.argv[1]}_{s}/cands.pkl","rb"))
    night, ixy = D["night"], D["inj_xy"]; E=len(night); cs=D["cands"]
    cs = sorted([c for c in cs if c.get("score") is not None and np.isfinite(c["score"]) and c["score"]>-1e8], key=lambda c:-c["score"])
    tracks=[]; out=[]
    for c in cs:
        xy=c["xy"]
        if tracks:
            dd=np.abs(np.array(tracks)-xy[None]).max(axis=2)
            if (np.sum(dd<2.0,axis=1)>=max(2,E//2)).any(): continue
        tracks.append(xy); out.append(c)
        if len(out)>=1500: break
    lab=[]
    for c in out:
        xy=c["xy"]; d=np.hypot(xy[:,None,0]-ixy[:,:,0], xy[:,None,1]-ixy[:,:,1])
        nn=[len(set(night[d[:,b]<2.0])) for b in range(d.shape[1])]
        near=[(d[:,b]<3).sum() for b in range(d.shape[1])]
        lab.append(1 if max(nn)>=3 else (-1 if max(near)>=1 else 0))
    lab=np.array(lab); sc=np.array([c["score"] for c in out])
    print(s, "E",E,"cands",len(cs),"dedup",len(out),"inj hits",(lab==1).sum(),"near-inj",(lab==-1).sum())
    neg=sc[lab==0]
    print("  neg score at rank 1,2,3,5,10,25,50,100,200,400,800:", [round(neg[k-1],1) for k in (1,2,3,5,10,25,50,100,200,400,800) if k<=len(neg)])
    print("  inj scores:", np.round(sc[lab==1],1).tolist())
    for c,l in list(zip(out,lab))[:int(sys.argv[2])]:
        print("   ", "INJ" if l==1 else "nr " if l==-1 else "   ", f"{c['score']:.1f} Z={c['Z']:.1f} pos={c['pos']:.1f} nn={c['nn']} fit={c['fit']} two={c.get('two')} r={np.linalg.norm(c['P']):.1f}", np.round(c["sv"],1).tolist())
