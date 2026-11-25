"""Offline: merge kept intermediates, label each final candidate (injected body class or junk), print features.
python feat.py INJDIR OUTJSON_PREFIX [SUBDIR] [N]"""
import sys, json, csv, math, os
import numpy as np
INJ, PRE = sys.argv[1], sys.argv[2]
W = sys.argv[3] if len(sys.argv) > 3 else "$RSI_SANDBOX/r1/gen039/workspace/submission"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 40
sys.path.insert(0, W + "/linker"); sys.path.insert(0, W); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finder as F, score_inject as SI, orbit
rows = list(csv.DictReader(open(INJ + "/exposures.csv")))
nt = {}
for r in rows:
    t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
    nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
night_t = np.array([v[0] for v in nt.values()]); night_o = np.array([v[1] for v in nt.values()])
m = F.merge(F.load(PRE + ".linker.json"), F.load(PRE + ".stack.json"), night_t, night_o)
fin = F.rank_balance(m)[:200]
truth = json.load(open(INJ + "/truth.json"))
cands, res, hr, dup = SI.score(truth, fin, rows)
seen = set()
print("rank   p     src  cls   a     e     i     r    mag")
for k, c in enumerate(cands[:N]):
    s = np.array(c["state"], float)
    r, v = F.R_ECL @ s[:3], F.R_ECL @ s[3:]
    rn = np.linalg.norm(r); en = 0.5 * v @ v - orbit.GM / rn; a = -orbit.GM / (2 * en)
    h = np.cross(r, v); inc = math.degrees(math.acos(h[2] / np.linalg.norm(h)))
    e = math.sqrt(max(0, 1 - (h @ h) / (orbit.GM * a)))
    lab = "junk"
    if res[k] is not None:
        j = res[k][0]; lab = truth["bodies"][j]["cls"] + ("dup" if j in seen else ""); seen.add(j)
    print(f"{k:4d} {c['p']:.3f} {c.get('src','?'):>3s} {lab:>5s} {a:6.1f} {e:5.2f} {inc:5.1f} {rn:5.1f} {c.get('mag', float('nan')):5.1f}")
top = {res[k][0] for k in range(25) if res[k] is not None}
cl = {}
for j in top:
    cl[truth["bodies"][j]["cls"]] = cl.get(truth["bodies"][j]["cls"], 0) + 1
allh = {}
for j in hr:
    allh[truth["bodies"][j]["cls"]] = allh.get(truth["bodies"][j]["cls"], 0) + 1
print("top25 first-time hits:", len(top), cl, "| all hits:", allh, "dup", dup)
