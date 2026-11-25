"""cmp.py SUBMISSION_DIR DUMP.pkl INJ.json OUT.json : which injected bodies are in OUT.json (2" on >= 3 nights), ranks."""
import sys, pickle, json, numpy as np
sys.path.insert(0, sys.argv[1]); import finder
D = pickle.load(open(sys.argv[2], "rb")); bodies = json.load(open(sys.argv[3])); out = json.load(open(sys.argv[4]))["candidates"]
R, tref, t, O, night, ixy = D["R"], D["tref"], D["t"], D["O"], D["night"], D["inj_xy"]
out = sorted(out, key=lambda c: -c["p"])
found = {}; chance = []
for rk, c in enumerate(out):
    st = np.array(c["state"]); xy = finder.predict_xy(R @ st[:3], R @ st[3:], c["epoch_mjd"] - tref, t, O)
    d = np.hypot(xy[:, None, 0] - ixy[:, :, 0], xy[:, None, 1] - ixy[:, :, 1])        # [E, B]
    hit = None
    for b in range(d.shape[1]):
        if len(set(night[d[:, b] < 2.0])) >= 3:
            hit = b; break
    if hit is None:
        chance.append((rk, round(c["score"], 1), round(float(np.linalg.norm(st[:3])), 1)))
    elif hit not in found:
        found[hit] = (rk, round(c["score"], 1), round(bodies[hit]["mag"], 2))
print(len(out), "candidates;", len(found), "injected found; in top 25:", sum(1 for v in found.values() if v[0] < 25))
print(" found (body: rank, score, mag):", dict(sorted(found.items(), key=lambda kv: kv[1][0])))
print(" non-injected (rank, score, r):", chance[:25])
# rank among non-injected only (what a real scenario would look like): how many non-injected candidates outrank each body
nr = np.array([c[0] for c in chance])
print(" bodies with < 25 non-injected candidates above them:", sum(1 for v in found.values() if (nr < v[0]).sum() < 25),
      "; with < 10:", sum(1 for v in found.values() if (nr < v[0]).sum() < 10))
