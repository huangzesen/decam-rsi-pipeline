import pickle, numpy as np, torch, sys, csv
scen=sys.argv[1]; e_i=int(sys.argv[2])
dets, maps = pickle.load(open(f"$RSI_SANDBOX/r2/gen021/tmp/cache/{scen}/{scen}.pkl","rb"))
used=sorted(maps); print(used)
rows=list(csv.DictReader(open(f"$DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev/{scen}/exposures.csv")))
night=[int(round(float(rows[i]["mjd_mid"])-0.2)) for i in used]; bands=[rows[i]["band"] for i in used]
# crop: common central window in lattice coords
y0=max(maps[i][0] for i in used)+5000; x0=max(maps[i][1] for i in used)+5000; S=3000
crop={k:torch.from_numpy(np.ascontiguousarray(maps[i][2][y0-maps[i][0]:y0-maps[i][0]+S, x0-maps[i][1]:x0-maps[i][1]+S])) for k,i in enumerate(used)}
E=len(used); e=e_i; f32=torch.float32
# crude depth weights from detection catalogue: median flux/snr
sig=[]
for i in used:
    d=np.asarray(dets[i]); m=(d[:,2]>3.5)&(d[:,2]<50); sig.append(float(np.median(d[m,3]/d[m,2])))
print("sigF (raw counts, no zp):", np.round(sig,2), bands, night)
oth=[k for k in range(E) if night[k]!=night[e]]
sm=torch.zeros((S,S)); cnt=torch.zeros((S,S)); mxv=torch.full((S,S),-128.)
for k in oth:
    v=crop[k]!=-128; sm+=torch.where(v,crop[k],torch.zeros_like(crop[k])).float(); cnt+=v.float(); mxv=torch.maximum(mxv,crop[k].float())
stat=(sm>28*torch.sqrt(cnt))|(mxv>=35)|(cnt==0)
st=torch.nn.functional.max_pool2d(stat.float()[None,None],3,1,1)[0,0]>0
bd=torch.nn.functional.max_pool2d(((mxv>=125)&stat).float()[None,None],11,1,5)[0,0]>0
st|=bd
vld=crop[e]!=-128
M=st&vld; n=float(M.sum())
print("valid",float(vld.float().mean()),"masked of valid",n/float(vld.sum()),"undilated",float((stat&vld).sum())/float(vld.sum()))
print("of masked: bright-dilated",float((M&bd).sum())/n,"cnt==0",float((M&(cnt==0)).sum())/n, "raw>=127", float((M&(crop[e]>=127)).sum())/n)
# unweighted mean static S/N prediction per exposure
pred=sm/cnt.clamp(min=1)
for thr in (10,20,30,50,80,125):
    print(" mean other-exposure S/N <=",thr/10,":",round(float((M&~bd&(cnt>0)&(pred<=thr)).sum())/n,3), " max other <=",thr/10,":",round(float((M&~bd&(cnt>0)&(mxv<=thr)).sum())/n,3))
for c in range(1,int(cnt.max())+1):
    print(" cnt",c,round(float((M&(cnt==c)).sum())/n,3))
print("decomposition of masked: stat", float((M&stat).sum())/n, "3x3 only", float((M&~stat&(torch.nn.functional.max_pool2d(stat.float()[None,None],3,1,1)[0,0]>0)).sum())/n, "bright-dilation only", float((M&~(torch.nn.functional.max_pool2d(stat.float()[None,None],3,1,1)[0,0]>0)).sum())/n)
# recovered map, both tiers merged crude: weights 1/sig (same units only within band; use S/N-space weights w=1/sig*zp unknown -> use all with equal depth weights as approximation)
w=np.array([1.0/s for s in sig])
def tmpl(ks):
    num=torch.zeros((S,S)); den=torch.zeros((S,S))
    for k in ks:
        v=crop[k]!=-128; num+=torch.where(v,crop[k],torch.zeros_like(crop[k])).float(); den+=v.float()
    return num/den.clamp(min=1), 1.0/den.clamp(min=1), den>0
num,var,cov=tmpl(oth)   # equal weights: pred S/N (0.1 units), variance
raw=crop[e].float()
mr=M&cov&(num>=20)&(num<=60)&(raw<127)
al=float((num[mr]*raw[mr]).sum()/100)/float(((num[mr]*0.1)**2-var[mr]).sum()); print("alpha",al)
num=num*al; var=var*al*al
def peaks(v, sel, thr):
    mp=torch.nn.functional.max_pool2d(v[None,None],3,1,1)[0,0]
    return float(((v>=thr)&(v==mp)&sel).sum())
clean=vld&~st
vv=torch.where(vld,raw,torch.full_like(raw,-128.))
print("clean area",float(clean.sum()),"peaks>=4.0 per Mpix",peaks(vv,clean,40)/float(clean.sum())*1e6,">=4.5",peaks(vv,clean,45)/float(clean.sum())*1e6,">=5.5",peaks(vv,clean,55)/float(clean.sum())*1e6)
for name,extra in (("all",torch.ones_like(M)),("not bright-dil",~bd),("bright-dil only",bd)):
  for mmax in (1.0,2.0,3.0):
    for sysf in (0.3,0.6):
        ok=M&cov&extra&(var<=2.0)&(num<=10*mmax)&(raw<127)
        sg=torch.sqrt(1+var+(0.1*sysf*num)**2)
        rec=torch.where(ok,(raw-num)/sg,torch.full_like(raw,-128.))
        a=float(ok.sum())
        if a<1000: continue
        print(f" {name} mmax {mmax} sys {sysf}: share of masked {a/n:.3f} | peaks/Mpix >=4.0: {peaks(rec,ok,40)/a*1e6:.0f}  >=4.5: {peaks(rec,ok,45)/a*1e6:.0f} >=5.5: {peaks(rec,ok,55)/a*1e6:.0f} | raw peaks there >=4.5: {peaks(vv,ok,45)/a*1e6:.0f}")
