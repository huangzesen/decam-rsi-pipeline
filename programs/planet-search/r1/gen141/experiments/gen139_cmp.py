import sys, json, csv, math, numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, "$RSI_SANDBOX/r1/gen139/workspace/submission/linker")
import score_inject
S, tags = sys.argv[1], sys.argv[2:]
truth = json.load(open(f"scen_inj{S}/truth.json")); rows = list(csv.DictReader(open(f"scen_inj{S}/exposures.csv")))
# angle off ecliptic prograde direction: approximate via velocity vs cross(ecl pole, r)
OBL = math.radians(23.4392911); pole = np.array([0, -math.sin(OBL), math.cos(OBL)])
ang = []
for b in truth["bodies"]:
    x = np.array(b["state"][:3]); v = np.array(b["state"][3:])
    vc = np.cross(pole, x); vc /= np.linalg.norm(vc)
    vt = v - x * (v @ x) / (x @ x)
    ang.append((math.degrees(math.acos(np.clip(vt @ vc / np.linalg.norm(vt), -1, 1))), np.linalg.norm(vt) / math.sqrt(2.959122082855911e-4 / np.linalg.norm(x))))
res = {}
for tag in tags:
    for kind in ("out", "fdump"):
        c = json.load(open(f"runs/{kind}_{tag}.json"))["candidates"]
        _, r, hr, _ = score_inject.score(truth, c, rows, top=25)
        res[(tag, kind)] = hr
print("body  ang  v/vc  snr  " + "  ".join(f"{t}:{k}" for t in tags for k in ("out", "fd")))
for i, b in enumerate(truth["bodies"]):
    print(f"{i:3d} {ang[i][0]:5.1f} {ang[i][1]:5.2f} {b['snr']:5.2f}  " + "  ".join(f"{str(res[(t,k)].get(i,'-')):>8s}" for t in tags for k in ("out", "fdump")))
