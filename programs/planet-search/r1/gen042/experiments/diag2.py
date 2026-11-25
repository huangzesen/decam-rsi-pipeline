import sys, os, json, csv, math, importlib.util, io, contextlib
import numpy as np
SUB=os.path.abspath("submission"); T="$RSI_SANDBOX/r1/gen042/tmp"; RUN=sys.argv[1]; i=int(sys.argv[2])
sys.path.insert(0, os.path.join(SUB,"linker")); sys.path.insert(0,"exp")
import score_inject as SI
spec=importlib.util.spec_from_file_location("wrap", os.path.join(SUB,"finder.py")); w=importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
scen=f"{T}/INJ{i}"; rows=list(csv.DictReader(open(scen+"/exposures.csv")))
nt={}
for r in rows:
    t_=float(r["mjd_mid"]); n_=math.floor(t_-0.5); nt.setdefault(n_,(t_,[float(r["obs_x"]),float(r["obs_y"]),float(r["obs_z"])]))
night_t=np.array([v[0] for v in nt.values()]); night_o=np.array([v[1] for v in nt.values()])
with contextlib.redirect_stdout(io.StringIO()):
    m=w.merge(w.load(f"{RUN}/inj{i}.json.linker.json"), w.load(f"{RUN}/inj{i}.json.stack.json"), night_t, night_o)
    fin=w.rank_balance(m)[:200]
truth=json.load(open(scen+"/truth.json")); cls=[b.get("cls",b.get("class","?")) for b in truth["bodies"]]
cands,res,hr,dup=SI.score(truth,fin,rows)
print(sorted(fin[0].keys()))
for k,c in enumerate(fin[:60]):
    s=np.array(c["state"]); r=np.linalg.norm(s[:3]); v=np.linalg.norm(s[3:]); GM=2.959122082855911e-4
    a=1/(2/r-v*v/GM); h=np.cross(w.R_ECL@s[:3],w.R_ECL@s[3:]); inc=math.degrees(math.acos(h[2]/np.linalg.norm(h)))
    tag=(cls[res[k][0]]+str(res[k][0])) if res[k] is not None else "--"
    ex={kk:(round(vv,2) if isinstance(vv,float) else vv) for kk,vv in c.items() if kk not in("state","epoch_mjd")}
    print(k,tag,f"r{r:.1f} a{a:.1f} i{inc:.0f}",ex)
