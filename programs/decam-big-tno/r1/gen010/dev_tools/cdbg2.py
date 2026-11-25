import sys, json, pickle, numpy as np, collections
out = sys.argv[1]
T = np.load('/tmp/p2/truth.npz'); E = json.load(open('/tmp/p2/exposures.json'))
et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']
rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
info = pickle.load(open('/tmp/w/info.pkl', 'rb'))
F = pickle.load(open(out + '/funnel3.pkl', 'rb')); found = F['found']
C = pickle.load(open(out + '/carry.pkl', 'rb'))
dbg = C['dbg']
print('dbg entries', len(dbg))
per = {}
for d in dbg:
    b = body[d['det']]
    if (b >= 0).all() and len(set(b.tolist())) == 1:
        j = int(b[0])
        if j not in info: continue
        sn = len(np.unique(night[rexp[d['det']]]))
        ns = info[j]['ns']; mem = np.unique(night[rexp[d['det']]])
        far = ns[(~np.isin(ns, mem)) & (ns >= mem.min() - 800) & (ns <= mem.max() + 800)]
        e = dict(sn=sn, far=len(far), cnt=d['cnt'], m=d.get('m', -1), fa=d.get('fa', -1), acc=d.get('acc', False), mu=d['mu'], ncell=d.get('ncell', -1), nb=d['nb'], K2=d['K2'], m1=d['m1'], m2=d['m2'], arc=mem.max()-mem.min(), found=j in found, n=info[j]['n'])
        per.setdefault(j, []).append(e)
tab = collections.Counter()
rows = []
for j, es in per.items():
    for e in es:
        if e['sn'] == 3:
            tab[(min(e['far'], 4), 'cnt>=far' if e['cnt'] >= e['far'] else 'cnt<far', 'acc' if e['acc'] else 'rej', 'found' if e['found'] else 'lost')] += 1
            rows.append(e)
for k in sorted(tab): print(k, tab[k])
print('bodies with a pure 3-night seed:', len([j for j, es in per.items() if any(e['sn'] == 3 for e in es)]))
import random; random.seed(1)
for e in random.sample([r for r in rows if r['far'] >= 3 and not r['found']], 14): print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()})
