"""one_eval.py RUNDIR : carried linkages of carry.pkl by origin; for src 4 (one far night) purity against the planted truth and fa."""
import sys, pickle, json, numpy as np
D = pickle.load(open(sys.argv[1] + '/carry.pkl', 'rb')); new = D['new']
T = np.load('/tmp/p4/truth.npz'); ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
E = json.load(open('/tmp/p4/exposures.json')); et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
dexp = np.load('/tmp/p4/det_exp.npy', mmap_mode='r')
rows = []
for c in new:
    d = np.asarray(c['det']); b = body[d]
    pure = (b >= 0).all() and (b == b[0]).all()
    npl = int((b >= 0).sum())
    rows.append((c.get('src'), pure, npl, len(d), c.get('fa'), c.get('sn'), c.get('nights'), c.get('rms')))
for s in sorted(set(r[0] for r in rows)):
    rr = [r for r in rows if r[0] == s]
    print('src', s, 'n', len(rr), 'pure', sum(r[1] for r in rr), 'no planted det', sum(r[2] == 0 for r in rr), 'mixed', sum((not r[1]) and r[2] > 0 for r in rr))
r4 = sorted([r for r in rows if r[0] == 4], key=lambda r: r[4])
for thr in (0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0):
    q = [r for r in r4 if r[4] < thr]
    print(f'src4 fa<{thr}: n {len(q)} pure {sum(r[1] for r in q)} none {sum(r[2] == 0 for r in q)} mixed {sum((not r[1]) and r[2] > 0 for r in q)}')
for r in r4[:60]: print('   pure' if r[1] else ('   none' if r[2] == 0 else '   mixed'), 'planted %d/%d fa %.4f sn %s nights %s rms %.2f' % (r[2], r[3], r[4], r[5], r[6], r[7]))
