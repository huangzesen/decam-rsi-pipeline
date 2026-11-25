import sys, json, numpy as np
sys.path.insert(0, '/tmp/w')
from cmp import found
fa, na, T, night = found(sys.argv[1], sys.argv[3]); fb, nb, _, _ = found(sys.argv[2], sys.argv[3])
ntot = int(T['ntot']); planted = np.zeros(ntot, bool); planted[T['rows']] = True
print('candidates of the sample itself (no planted detection among the members), matched between the two runs by members:')
print('  nights seasons arc | rms A -> B | p A -> B | r')
for c in sorted(na, key=lambda c: -c['p']):
    d = np.array(c['det'])
    if planted[d].any() or c['p'] < 0.05: continue
    best = None
    for c2 in nb:
        o = len(np.intersect1d(d, np.array(c2['det'])))
        if o >= 0.6 * max(len(d), len(c2['det'])) and (best is None or o > best[0]): best = (o, c2)
    if best is None:
        print('  %2d %d %6.0f | %.3f -> (not in B) | %.3f |' % (c.get('nights'), c.get('seasons'), c.get('arc', 0), c.get('rms'), c['p']), c.get('r')); continue
    c2 = best[1]
    print('  %2d %d %6.0f | %.3f -> %.3f | %.3f -> %.3f | %s' % (c.get('nights'), c.get('seasons'), c.get('arc', 0), c.get('rms'), c2.get('rms'), c['p'], c2['p'], c.get('r')))
# planted true: rms distribution
for lab, f in (('A', fa), ('B', fb)):
    r = np.array([c['rms'] for c in f.values()]); s = np.array([c.get('seasons', 1) for c in f.values()])
    print(lab, 'true planted linkages: n %d, rms median %.3f (one season %.3f, >= 2 seasons %.3f), p >= 0.9: %d, p >= 0.5: %d' % (len(r), np.median(r), np.median(r[s == 1]), np.median(r[s >= 2]), sum(c['p'] >= 0.9 for c in f.values()), sum(c['p'] >= 0.5 for c in f.values())))
