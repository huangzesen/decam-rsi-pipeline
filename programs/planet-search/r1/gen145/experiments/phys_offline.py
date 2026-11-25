import sys, json, csv, math, os
import numpy as np
W="$RSI_SANDBOX/r1/gen038/workspace/submission"; V="$RSI_SANDBOX/r1/gen038/tmp/varB"
sys.path.insert(0, W+"/linker"); sys.path.insert(0, W); sys.path.insert(0,"$RSI_SANDBOX/r1/gen038/tmp/exp")
import finder as F, score_inject as SI
INJ="$RSI_SANDBOX/r1/gen038/tmp/INJ"
rows=list(csv.DictReader(open(INJ+"/exposures.csv")))
nt={}
for r in rows:
    t_=float(r["mjd_mid"]); n_=math.floor(t_-0.5); nt.setdefault(n_,(t_,[float(r["obs_x"]),float(r["obs_y"]),float(r["obs_z"])]))
night_t=np.array([v[0] for v in nt.values()]); night_o=np.array([v[1] for v in nt.values()])
F.PHYS=int(sys.argv[1])
m=F.merge(F.load(V+"/base.json.linker.json"),F.load(V+"/base.json.stack.json"),night_t,night_o)
fin=F.rank_balance(m)[:200]
truth=json.load(open(INJ+"/truth.json"))
cands,res,hr,dup=SI.score(truth,fin,rows)
print("top25 first-time hits:", len({res[k][0] for k in range(25) if res[k] is not None}))
print("hit ranks:", sorted(hr.values()) if isinstance(hr,dict) else hr)
