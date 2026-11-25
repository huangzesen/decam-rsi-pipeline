import json, sys, numpy as np, math
cands=json.load(open(sys.argv[1])); T=np.load('/tmp/p4/truth.npz')
E=json.load(open('/tmp/p4/exposures.json')); night=np.floor(np.array([e['mjd_mid'] for e in E])-0.5).astype(int)
ntot=int(T['ntot']); body=-np.ones(ntot,np.int64); body[T['rows']]=T['body']
rexp=np.zeros(ntot,np.int64); rexp[T['rows']]=T['exp']
rows=[]; seen=set()
for c in cands:
    d=np.array(c['det']); b=body[d]; bb=b[b>=0]; cls='chance' if len(bb)==0 else 'mixed'
    if len(bb):
        u,cnt=np.unique(bb,return_counts=True); j=int(u[cnt.argmax()])
        if cnt.max()>=0.8*len(d) and len(np.unique(night[rexp[d[b==j]]]))>=5:
            cls='true' if j not in seen else 'dup'; seen.add(j)
    rows.append((cls,c))
def tab2(name,f,sel=lambda c:True):
    print(name)
    for cl in ('true','chance','mixed'):
        v=np.array([f(c) for k,c in rows if k==cl and sel(c)])
        if len(v): print(f'   {cl:7s} n={len(v):4d}  0:{(v==0).mean():.2f} 1:{(v==1).mean():.2f} 2:{(v==2).mean():.2f} 3+:{(v>=3).mean():.2f}')
tab2('h3 nights with repeats', lambda c:c['h3'])
tab2('h3m nights with motion', lambda c:c['h3m'])
tab2('h3, nights<=6', lambda c:c['h3'], lambda c:c['nights']<=6)
tab2('h3m, nights<=6', lambda c:c['h3m'], lambda c:c['nights']<=6)
tab2('h3, nights<=6 & medSN<7', lambda c:c['h3'], lambda c:c['nights']<=6 and c['p']<0.6)
print('chance with h3>0:', [(round(c['p'],2), c['nights'], c['h3'], c['h3m'], len(c['det']), c['rms'], c['msc']) for k,c in rows if k=='chance' and c['h3']>0])
print()
def ev(wh,wv):
    out=[]
    for k,c in rows:
        s=c['s']+wh*min(c['h3'],3)+wv*min(c['h3m'],2)
        p0=1/(1+math.exp(-0.7*c['s'])); fac=c['p']/max(p0,1e-12)
        out.append((k,fac/(1+math.exp(-0.7*s)),c))
    return out
for wh,wv in [(0,0),(0.6,1.0),(1.0,1.0),(1.0,1.5),(1.2,1.5),(1.5,2.0),(0.0,2.0),(1.0,0.0)]:
    o=ev(wh,wv)
    pt=np.sort([p for k,p,c in o if k=='true'])[::-1]; pc=np.sort([p for k,p,c in o if k=='chance'])[::-1]
    pn=np.sort([p for k,p,c in o if k in('chance','mixed')])[::-1]
    print(f'wh {wh} wv {wv}: true above n-th non-true (1,3,5,10,20): '+' '.join(str(int((pt>pn[n-1]).sum())) for n in (1,3,5,10,20))
      +' | above n-th pure chance (1,3,5,10): '+' '.join(str(int((pt>pc[n-1]).sum())) for n in (1,3,5,10))
      +f' | true p>=0.9/0.7/0.5/0.3: {(pt>=0.9).sum()}/{(pt>=0.7).sum()}/{(pt>=0.5).sum()}/{(pt>=0.3).sum()} chance p>=0.5/0.3/0.1: {(pc>=0.5).sum()}/{(pc>=0.3).sum()}/{(pc>=0.1).sum()} mixed>=0.5: {sum(p>=0.5 for k,p,c in o if k=="mixed")}')
