# Per-round history of the bodies gen071 gained, by the grader's own matching rule. Prints and stores numbers only; no body names or field ids.
import glob, json, os, re, sys, math
import numpy as np
RSI = "$ALE_HOME/planet-search/rsi"
sys.path.insert(0, os.path.join(RSI, "harness"))
import grade as G
sid, truth_path, scen_path, out_path = sys.argv[1:5]
T = json.load(open(truth_path)); P = json.load(open(scen_path))
E = P["all_exposures"]; mjds = [e["mjd_mid"] for e in E]; obs = [np.array([e["obs_x"], e["obs_y"], e["obs_z"]]) for e in E]
night = [int(np.floor(m - 0.5)) for m in mjds]
gens = os.path.join(RSI, "runs/r1/gens")
def num(g): return int(re.match(r"gen(\d+)", g).group(1))
g70 = json.load(open(f"{gens}/gen070/grade/grade.json")); g71 = json.load(open(f"{gens}/gen071/grade/grade.json"))
old = {h["pdes"] for h in g70["hits"] if h["sid"] == sid}; new = [h["pdes"] for h in g71["hits"] if h["sid"] == sid and h["pdes"] not in old]
B = [b for b in T["bodies"] if b["pdes"] in new]
def matches(c, b):
    pos = G.positions(c, mjds, obs)
    if pos is None or np.linalg.norm(c["state"][:3]) < 25: return False
    ra, de = pos; ok = set()
    for k, r in enumerate(b["recs"]):
        if r["held"] or r.get("ra") is None: continue
        if G.sep(ra[k], de[k], r["ra"], r["dec"]) <= max(2.0, r.get("s_ra") or 0, r.get("s_dec") or 0): ok.add(night[k])
    return len(ok) >= 3
rows = []
for gd in sorted(glob.glob(f"{gens}/gen*/grade/grade.json"), key=lambda p: (num(p.split("/")[-3]), p)):
    g = gd.split("/")[-3]
    if not re.fullmatch(r"gen\d{3}", g) or num(g) < 20: continue
    d = json.load(open(gd)); allp = []
    for s in d["settings"]["scenarios"]:
        f = f"{gens}/{g}/ls6_runs/{s}/candidates.json"
        if not os.path.exists(f): allp = None; break
        c = json.load(open(f)); c = c if isinstance(c, list) else c["candidates"]
        allp += sorted((x["p"] for x in c), reverse=True)[:200]
    f = f"{gens}/{g}/ls6_runs/{sid}/candidates.json"
    if allp is None or not os.path.exists(f): continue
    allp.sort(reverse=True); cut = allp[d["K"] - 1] if len(allp) >= d["K"] else 0.0
    c = json.load(open(f)); c = c if isinstance(c, list) else c["candidates"]; c = sorted(c, key=lambda x: -x["p"])[:200]
    row = dict(n=num(g), cut=round(cut, 4), reward=d["reward"], bodies=[])
    for b in B:
        rk = None
        for i, x in enumerate(c):
            if matches({**x, "epoch": x.get("epoch_mjd", x.get("epoch"))}, b): rk = i; break
        row["bodies"].append(dict(rank=(rk + 1) if rk is not None else None, p=round(c[rk]["p"], 4) if rk is not None else None, hit=any(h["sid"] == sid and h["pdes"] == b["pdes"] for h in d["hits"])))
    rows.append(row); print(row["n"], row["cut"], [(x["rank"], x["p"], x["hit"]) for x in row["bodies"]], flush=True)
json.dump(rows, open(out_path, "w"))
