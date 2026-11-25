import json, csv, math, numpy as np, collections
S="$RSI_SANDBOX/r1/gen104/tmp/inj7c"
F=json.load(open("forced.json")); T=json.load(open(S+"/truth.json"))
rows=list(csv.DictReader(open(S+"/exposures.csv"))); t=np.array([float(r["mjd_mid"]) for r in rows]); night=np.floor(t+0.3).astype(int)
z={}
for r in T["injections"]: pass
for rec in F:
    j=rec["j"]
    if rec["cls"]=="A": continue
    ex=sorted({r["exp"] for r in T["injections"] if r["body"]==j})
    zz=dict(zip(ex,rec["exp_z"])) if len(ex)==len(rec["exp_z"]) else None
    if zz is None: print(j,"len mismatch"); continue
    by=collections.defaultdict(list)
    for e,v in zz.items(): by[night[e]].append(v)
    ok=0
    for n,vs in by.items():
        if len(vs)>=2: ok+= (min(vs)>=3.5 and sum(vs)/math.sqrt(len(vs))>=5.5) or max(vs)>=4.6
        else: ok+= vs[0]>=4.6
    print(j,rec["cls"],"linker-detectable nights",ok,"of",len(by),"night_z",rec["night_z"],"all",rec["all_z"],"best3",rec["best3"])
