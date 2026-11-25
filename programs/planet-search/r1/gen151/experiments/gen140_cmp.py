import sys, json, csv
sys.path.insert(0, "$RSI_SANDBOX/r1/gen140/tmp/tools")
import score_inject
S, tags = sys.argv[1], sys.argv[2:]
T = "$RSI_SANDBOX/r1/gen140/tmp"
truth = json.load(open(f"{T}/scen_inj{S}/truth.json")); rows = list(csv.DictReader(open(f"{T}/scen_inj{S}/exposures.csv")))
res = {}
for tag in tags:
    for kind in ("out", "fdump"):
        try:
            c = json.load(open(f"{T}/runs/{kind}_{tag}.json"))["candidates"]
            _, r, hr, _ = score_inject.score(truth, c, rows, top=25)
            res[(tag, kind)] = hr
            print(tag, kind, "n_cand", len(c), "found", len(hr), "top25", sum(1 for v in hr.values() if v[0] < 25))
        except Exception as e:
            print(tag, kind, "ERR", e); res[(tag, kind)] = {}
print("body snr " + " ".join(f"{t}:{k[:2]}" for t in tags for k in ("out", "fdump")))
for i, b in enumerate(truth["bodies"]):
    print(f"{i:3d} {b.get('snr',0):5.2f} " + " ".join(f"{str(res[(t,k)].get(i,("-",))[0]):>9s}" for t in tags for k in ("out", "fdump")))
