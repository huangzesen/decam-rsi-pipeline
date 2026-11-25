"""ev.py SCEN INJ.json RUNDIR... : which injected bodies are in OUT.json (2 arcsec on >= 3 nights), by rank; S/N from the log."""
import sys, json, csv, re, os, numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r2/gen024/workspace/submission")
import finder as F
scen, injf = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
mjd = np.array([float(r["mjd_mid"]) for r in rows]); O = np.array([[float(r[c]) for c in ("obs_x", "obs_y", "obs_z")] for r in rows])
night = np.round(mjd - 0.2).astype(int)
def sky(st, ep):
    P0 = np.array(st[:3]); V0 = np.array(st[3:])
    lt = np.full(len(mjd), np.linalg.norm(P0 - O[0]) / F.C_AU)
    for _ in range(3):
        P, _v = F.propagate(P0, V0, mjd - lt - ep)
        D = P - O; lt = np.linalg.norm(D, axis=1) / F.C_AU
    return D / np.linalg.norm(D, axis=1, keepdims=True)
inj = json.load(open(injf)); U = np.array([sky(b["state"], b["epoch_mjd"]) for b in inj])
for run in sys.argv[3:]:
    out = sorted(json.load(open(run + "/OUT.json"))["candidates"], key=lambda c: -c["p"])
    sn = {}
    for l in open(run + "/log.txt"):
        m = re.search(r"\]   inj (\d+): \[(.*)\]", l)
        if m:
            sn[int(m.group(1))] = [float(re.sub(r"[^0-9.]", "", v)) if re.search(r"\d", v) else np.nan for v in m.group(2).split(",")]
    rank = {}
    for rk, c in enumerate(out):
        u = sky(c["state"], c["epoch_mjd"])
        sep = np.degrees(np.arccos(np.clip((U * u[None]).sum(2), -1, 1))) * 3600
        for b in np.nonzero((sep < 2.0).sum(1) >= 3)[0]:
            if len(set(night[sep[b] < 2.0])) >= 3 and b not in rank: rank[b] = rk
    res = []
    for b, s in sn.items():
        s = np.array(s); ok = np.isfinite(s)
        if len(s) != len(mjd) or len(set(night[ok])) < 3: continue
        stk = lambda m: float(np.nansum(s[m]) / np.sqrt(max(m.sum(), 1)))
        n1 = sum(1 for n in set(night[ok]) if (s[ok & (night == n)] >= 5).any()); n2 = sum(1 for n in set(night[ok]) if stk(ok & (night == n)) >= 5); Z = stk(ok)
        w = 1 if n1 >= 3 else 2 if n2 >= 3 else 4 if Z >= 5 else 0
        res.append((Z, b, w, rank.get(b)))
    res.sort(reverse=True)
    for w in (1, 2, 4):
        rr = [r for r in res if r[2] == w]
        print(run.split("/")[-1], f"weight {w}: bodies {len(rr)}, in list {sum(r[3] is not None for r in rr)}, top 25 {sum(r[3] is not None and r[3] < 25 for r in rr)}, top 40 {sum(r[3] is not None and r[3] < 40 for r in rr)}")
    print("  in list (Z, body, w, rank):", [(round(z, 1), b, w, rk) for z, b, w, rk in res if rk is not None])
    print("  missed w>0 (Z, body, w):", [(round(z, 1), b, w) for z, b, w, rk in res if rk is None and w > 0][:40])
    print("  n cands", len(out), "p>=0.5:", sum(c["p"] >= 0.5 for c in out))
