"""ana_c.py CAND.json DST : carried candidates (src>0): true/false by seed nights, fa, p, seed brightness."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
snr = np.load(sys.argv[2] + '/det_snr.npy', mmap_mode='r')
rows = []
seen = set()
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]; ok = False; j = -1
    if len(bb):
        u, cnt = np.unique(bb, return_counts=True); j = int(u[cnt.argmax()])
        if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5:
            ok = True
    first = ok and j not in seen
    if ok: seen.add(j)
    rows.append(dict(ok=ok, first=first, p=c['p'], src=c.get('src', 0), sn=c.get('sn', 0), fa=c.get('fa', 0.0), nights=c.get('nights'), seasons=c.get('seasons'),
                     npl=int((b >= 0).sum()), nd=len(d), ssn=c.get('ssn'), s=c.get('s'), med=float(np.median(snr[d])), pr=c.get('pr')))
print('candidates', len(rows), 'true (first)', sum(r['first'] for r in rows))
for src in (0, 1, 2):
    for sn in (3, 4):
        R = [r for r in rows if r['src'] == src and (r['sn'] >= 4) == (sn == 4)]
        if not R: continue
        t = [r for r in R if r['first']]; f = [r for r in R if not r['ok'] and r['npl'] <= 1]; mix = [r for r in R if not r['ok'] and r['npl'] > 1]
        print(f'src {src} seed nights {sn}: true {len(t)} chance {len(f)} mixed {len(mix)}')
        for name, S in (('true', t), ('chance', f)):
            if S:
                p = np.array([r['p'] for r in S]); fa = np.array([r['fa'] for r in S])
                print(f'   {name}: p>=0.5/0.3/0.1/0.03/0.01: {[(p >= v).sum() for v in (0.5, 0.3, 0.1, 0.03, 0.01)]}  fa 10/50/90%: {np.percentile(fa, [10, 50, 90])}  med snr 10/50/90: {np.percentile([r["med"] for r in S], [10, 50, 90]).round(1)}')
                if S[0].get('ssn') is not None:
                    print('      ssn 10/50/90:', np.percentile([r['ssn'] or 0 for r in S], [10, 50, 90]).round(1))
if len(sys.argv) > 3:
    for r in sorted(rows, key=lambda r: -r['p']):
        if r['src'] > 0 and r['sn'] < 4: print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
