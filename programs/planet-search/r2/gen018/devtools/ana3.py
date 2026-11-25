"""ana3.py DUMPDIR INJ.json N : the best N de-duplicated candidates of a FINDER_DUMPC dump, each with its class (two = two-night candidate, fit = fitted orbit), its S/N per exposure, the nearest injected body and the offset from it per exposure (gen009)."""
import sys, pickle, json, numpy as np
D = pickle.load(open(sys.argv[1] + "/cands.pkl", "rb")); bodies = json.load(open(sys.argv[2]))
night, ixy = D["night"], D["inj_xy"]; E = len(night); cs = D["cands"]
cs = sorted([c for c in cs if c.get("score") is not None and np.isfinite(c["score"])], key=lambda c: -c["score"])
tracks = []; n = 0
for c in cs:
    xy = c["xy"]
    if tracks:
        dd = np.abs(np.array(tracks) - xy[None]).max(axis=2)
        if (np.sum(dd < 2.0, axis=1) >= max(2, E // 2)).any(): continue
    tracks.append(xy); n += 1
    d = np.hypot(xy[:, None, 0] - ixy[:, :, 0], xy[:, None, 1] - ixy[:, :, 1])   # [E, B]
    b = int(np.argmin(np.sort(d, axis=0)[1]))        # body closest in its 2nd-best exposure
    hit = len(set(night[d[:, b] < 2.0])) >= 3
    print(n, f"{c['score']:.1f}", "two" if c.get("two") else "   ", "fit" if c["fit"] else "   ", np.round(c["sv"], 1).tolist(), "body", b, "HIT" if hit else "", "off", np.round(d[:, b], 1).tolist(), f"r={np.linalg.norm(c['P']):.1f} rtrue={bodies[b]['r']:.1f}")
    if n >= int(sys.argv[3]): break
