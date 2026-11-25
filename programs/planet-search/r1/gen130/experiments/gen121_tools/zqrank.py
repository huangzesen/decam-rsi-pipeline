import sys, json, csv
import score_inject as si
truth = json.load(open(sys.argv[1])); rows = list(csv.DictReader(open(sys.argv[3])))
cands = json.load(open(sys.argv[2]))["candidates"]
cs, res, hr, dup = si.score(truth, cands, rows)
R = [(c, None if h is None else truth["bodies"][h[0]]["cls"] + str(h[0])) for c, h in zip(cs, res) if "zq" in c]
print(len(R), "kind-3 rows with zq")
for key in ("zc0", "zc", "zq"):
    s = sorted(R, key=lambda x: -x[0][key])
    print(key, [(k, lab) for k, (c, lab) in enumerate(s) if lab][:14])
