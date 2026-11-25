import sys, json, csv, numpy as np
sys.path.insert(0, sys.argv[4]); import score_inject as S
truth = json.load(open(sys.argv[1])); C = json.load(open(sys.argv[2]))["candidates"]; rows = list(csv.DictReader(open(sys.argv[3])))
cands, res, hr, dup = S.score(truth, C, rows, top=25)
cls = [b.get("cls", b.get("class", "?")) for b in truth["bodies"]]
for k, (c, h) in enumerate(zip(cands[:40], res[:40])):
    lab = "-" if h is None else cls[h[0]] + str(h[0])
    print(k, lab, round(c.get("rbg", np.nan), 2), round(c.get("rpen", np.nan), 2), c.get("kind"))
pen = np.array([c.get("rpen", 0) for c in cands]); hit = np.array([h is not None for h in res])
print("hits penalised", int((pen[hit] > 0).sum()), "of", int(hit.sum()), "; junk penalised", int((pen[~hit] > 0).sum()), "of", int((~hit).sum()))
