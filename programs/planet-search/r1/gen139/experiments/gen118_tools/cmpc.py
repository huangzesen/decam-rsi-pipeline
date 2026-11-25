import sys, json, csv, numpy as np
sys.path.insert(0, '$RSI_SANDBOX/r1/gen118/workspace/submission/linker')
import orbit
A, B, scen = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(scen + '/exposures.csv')))
t = np.array([float(r['mjd_mid']) for r in rows]); obs = np.array([[float(r[k]) for k in ('obs_x','obs_y','obs_z')] for r in rows])
nt = np.floor(t + 0.3); _, first = np.unique(nt, return_index=True); t, obs = t[first], obs[first]
def U(cs): return np.array([orbit.predict_dirs(c['state'], c['epoch_mjd'], t, obs) for c in cs])
ca = sorted(json.load(open(A))['candidates'], key=lambda c: -c['p']); cb = sorted(json.load(open(B))['candidates'], key=lambda c: -c['p'])
ua, ub = U(ca), U(cb)
for i in range(int(sys.argv[4]) if len(sys.argv) > 4 else 10):
    d = np.degrees(np.arccos(np.clip(np.einsum('nk,mnk->mn', ua[i], ub), -1, 1))) * 3600
    ok = (d < 2.5).sum(1) >= 3
    j = np.where(ok)[0]
    print(i, round(ca[i]['p'], 3), '->', [(int(k), round(cb[k]['p'], 3)) for k in j[:3]])
