# which score path sets stackd p for hits vs junk (top 60 of stackd's own output)
import sys, json, csv, numpy as np
sys.path.insert(0, '$RSI_SANDBOX/r1/gen118/workspace/submission/linker'); import orbit
for name, scen in [('base7','inj7d'),('bg3_7','inj7d'),('baseD','injD'),('bg3_D','injD')]:
    rows = list(csv.DictReader(open(f'scen/{scen}/exposures.csv')))
    t = np.array([float(r['mjd_mid']) for r in rows]); obs = np.array([[float(r[k]) for k in ('obs_x','obs_y','obs_z')] for r in rows])
    nt = np.floor(t + 0.3)
    tr = json.load(open(f'scen/{scen}/truth.json'))['bodies']
    UT = np.array([orbit.predict_dirs(b['state'], b['epoch_mjd'], t, obs) for b in tr])
    cs = sorted(json.load(open(f'runs/b_{name}.json.stack.json'))['candidates'], key=lambda c: -c['p'])[:60]
    stat = {}
    for i, c in enumerate(cs):
        u = orbit.predict_dirs(c['state'], c['epoch_mjd'], t, obs)
        d = np.degrees(np.arccos(np.clip(np.einsum('bnk,nk->bn', UT, u), -1, 1))) * 3600
        hit = [len(set(nt[d[b] < 2])) >= 3 for b in range(len(tr))]
        lab = tr[int(np.argmax(hit))].get('cls', '?') if any(hit) else 'junk'
        src = 'pn' if abs(c.get('pn', -1) - c['p']) < 1e-9 else ('zc' if 'zc' in c else 'z')
        stat.setdefault((lab if lab == 'junk' else 'hit', src), []).append(i)
    print(name, {f'{k[0]}/{k[1]}': (len(v), v[:8]) for k, v in sorted(stat.items())})
