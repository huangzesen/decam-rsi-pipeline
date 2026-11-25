import json,sys
import numpy as np
for r in sys.argv[1:]:
    S=json.load(open(r+'.stack.json'))['candidates']; F=json.load(open(r))['candidates']
    p=np.array([c['p'] for c in S]); k3=np.array(['zc' in c for c in S])
    pf=np.sort([c['p'] for c in F])[::-1]
    print(f"{r.split('/')[-1]:28s} stack n{len(p)} p>.5 {int((p>.5).sum())} p>.8 {int((p>.8).sum())} zc-rows {int(k3.sum())} (p>.5 {int((p[k3]>.5).sum())}) | final p@10 {pf[10]:.3f} @25 {pf[25]:.3f} @50 {pf[50]:.3f}")
