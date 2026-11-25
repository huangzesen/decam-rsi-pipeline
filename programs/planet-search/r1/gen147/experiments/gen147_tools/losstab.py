import sys, json, csv, os
sys.path.insert(0, "$RSI_SANDBOX/r1/gen147/tmp/tools")
import score_inject
S, tag = sys.argv[1], sys.argv[2]
T = "$RSI_SANDBOX/r1/gen147/tmp"
truth = json.load(open(f"{T}/scen_inj{S}/truth.json")); rows = list(csv.DictReader(open(f"{T}/scen_inj{S}/exposures.csv")))
st = [("fdump", f"{T}/runs/fdump_{tag}.json")]
for nm in ("refined", "tocands", "dedup", "rankx"):
    st.append((nm, f"{T}/runs/sd_{tag}/{nm}.json"))
st += [("stack", f"{T}/runs/out_{tag}.json.stack.json"), ("out", f"{T}/runs/out_{tag}.json")]
res = {}
for nm, f in st:
    try:
        c = json.load(open(f))["candidates"]
        if nm == "refined":
            for x in c: x["p"] = x["sc"]
        _, _, hr, _ = score_inject.score(truth, c, rows, top=25)
        res[nm] = hr; print(nm, len(c), "found", len(hr))
    except Exception as e:
        print(nm, "ERR", e); res[nm] = {}
print("body snr " + " ".join(f"{n:>8s}" for n, _ in st))
for i, b in enumerate(truth["bodies"]):
    if any(i in res[n] for n, _ in st):
        print(f"{i:3d} {b.get('snr',0):5.2f} " + " ".join(f"{str(res[n].get(i,('-',))[0]):>8s}" for n, _ in st))
