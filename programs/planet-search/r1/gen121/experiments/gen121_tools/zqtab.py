import sys, json, csv
sys.argv_ = sys.argv
import score_inject as si
truth = json.load(open(sys.argv[1])); cands = json.load(open(sys.argv[2]))["candidates"]; rows = list(csv.DictReader(open(sys.argv[3])))
cs, res, hr, dup = si.score(truth, cands, rows)
for k, (c, h) in enumerate(zip(cs, res)):
    if "zq" in c or (h is not None and k < 60):
        lab = "-" if h is None else truth["bodies"][h[0]]["cls"] + str(h[0])
        print(k, lab, round(c["p"], 3), "zc0", round(c.get("zc0", float("nan")), 2), "zc", round(c.get("zc", float("nan")), 2), "zq", round(c.get("zq", float("nan")), 2))
