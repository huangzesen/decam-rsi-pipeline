"""ana2.py DUMPDIR INJ.json : per score variant, greedy track de-dup, top 200, injected found / in top 25 / chance level."""
import sys, pickle, json, numpy as np
D = pickle.load(open(sys.argv[1] + "/cands.pkl", "rb")); bodies = json.load(open(sys.argv[2]))
night, ixy = D["night"], D["inj_xy"]; E = len(night)
cs = D["cands"]
nv = 3 if cs and cs[0].get("alt") is not None else 1
for v in range(nv):
    sc = np.array([c["alt"][v] if nv == 3 else c["score"] for c in cs])
    o = np.argsort(-sc)
    tracks = []; out = []
    for q in o:
        c = cs[q]
        if not np.isfinite(sc[q]) or sc[q] < -1e8: continue
        xy = c["xy0"] if (v == 0 and c.get("xy0") is not None) else c["xy"]
        if tracks:
            dd = np.abs(np.array(tracks) - xy[None]).max(axis=2)
            if (np.sum(dd < 2.0, axis=1) >= max(2, E // 2)).any(): continue
        tracks.append(xy); out.append((sc[q], xy))
        if len(out) >= 200: break
    found = {}; chance = []
    for rk, (s, xy) in enumerate(out):
        d = np.hypot(xy[:, None, 0] - ixy[:, :, 0], xy[:, None, 1] - ixy[:, :, 1])
        hit = None
        for b in range(d.shape[1]):
            if len(set(night[d[:, b] < 2.0])) >= 3: hit = b; break
        if hit is None: chance.append((rk, round(float(s), 1)))
        elif hit not in found: found[hit] = (rk, round(float(s), 1), round(bodies[hit]["mag"], 2))
    nr = np.array([c[0] for c in chance])
    print(f"variant {v}: found {len(found)}, top25 {sum(1 for x in found.values() if x[0] < 25)}, <25 non-inj above: {sum(1 for x in found.values() if (nr < x[0]).sum() < 25)}, <10: {sum(1 for x in found.values() if (nr < x[0]).sum() < 10)}")
    print("   chance scores:", [c[1] for c in chance[:12]], "25th:", chance[24][1] if len(chance) > 24 else None)
    print("   found:", dict(sorted(found.items(), key=lambda kv: kv[1][0])))
