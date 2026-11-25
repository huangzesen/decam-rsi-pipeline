"""k3apply.py OUTX.json NEW.json K3PRO K3Z K3N K3F K3L K3RANK : offline copy of the TNO_K3PRO rule."""
import sys, json
f, fo = sys.argv[1:3]
K3PRO, K3Z, K3N, K3F, K3L, K3RANK = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8])
cands = sorted(json.load(open(f))["candidates"], key=lambda c: -c["p"])
elig = [(c["zc"], j) for j, c in enumerate(cands) if j >= 25 and "npo" in c and c["zc"] >= K3Z and c["npo"] >= K3N
        and c["npo"] >= K3F * c["ncv"] and c["lo"] >= K3L]
elig = sorted(elig, reverse=True)[:K3PRO]
pro = [j for _, j in elig if j > K3RANK]
ps = [c["p"] for c in cands]
rest = [c for j, c in enumerate(cands) if j not in set(pro)]
order = rest[:K3RANK] + [dict(cands[j], k3pro=1) for j in pro] + rest[K3RANK:]
json.dump({"candidates": [dict(c, p=p) for c, p in zip(order, ps)]}, open(fo, "w"))
print("eligible", [(round(z, 2), j) for z, j in elig], "promoted", pro)
