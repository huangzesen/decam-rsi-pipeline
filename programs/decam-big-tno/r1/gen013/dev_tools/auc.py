import json, sys, numpy as np
def load(f, dst=None):
    cands = json.load(open(f)); out = []
    if dst:
        T = np.load(dst + '/truth.npz'); E = json.load(open(dst + '/exposures.json')); night = np.floor(np.array([e['mjd_mid'] for e in E]) - 0.5).astype(int)
        body = -np.ones(int(T['ntot']), np.int64); body[T['rows']] = T['body']; rexp = np.zeros(int(T['ntot']), np.int64); rexp[T['rows']] = T['exp']
    for c in cands:
        cls = 0
        if dst:
            d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
            if len(bb):
                u, cnt = np.unique(bb, return_counts=True); j = u[cnt.argmax()]
                if cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5: cls = 1
                elif len(bb) >= 3: cls = -1          # mixed: leave out
        out.append((cls, c))
    return out
A = load(sys.argv[1], sys.argv[2]); U = load(sys.argv[3]) if len(sys.argv) > 3 else []
pos = [c for k, c in A if k == 1 and c['nights'] <= 7]; neg = [c for k, c in A if k == 0 and c['nights'] <= 7] + [c for k, c in U if c['p'] < 0.9 and c['nights'] <= 7]
print('pos', len(pos), 'neg', len(neg))
def auc(f):
    a = np.array([f(c) for c in pos]); b = np.array([f(c) for c in neg])
    return ((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()), np.mean(a > np.sort(b)[-max(len(b) // 10, 1)]), np.mean(a > np.sort(b)[-max(len(b) // 4, 1)])
lg = lambda p: np.log(max(p, 1e-9) / max(1 - p, 1e-9))
print('current p: auc %.4f, true above top-10%% false %.3f, above top-25%% false %.3f' % auc(lambda c: c['p']))
for a_, b_ in ((0, 0), (0.3, 1.0), (0.3, 1.5), (0.3, 2.0), (0.5, 1.5), (0.5, 2.5), (0.0, 1.5), (0.7, 2.0), (1.0, 2.6)):
    print(a_, b_, 'auc %.4f, >top10%% %.3f, >top25%% %.3f' % auc(lambda c: lg(c['p']) + 0.7 * (a_ * min(c['ndh'], 5) - b_ * c['ndm'])))
print('--- variants with excess misses')
for a_, b_, c_ in ((0.5, 1.2, 0.0), (0.5, 1.2, 0.15), (0.5, 1.5, 0.15), (0.6, 1.5, 0.2), (0.4, 1.0, 0.15)):
    print(a_, b_, c_, 'auc %.4f, >top10%% %.3f, >top25%% %.3f' % auc(lambda c: lg(c['p']) + 0.7 * (a_ * min(c['ndh'], 5) - b_ * max(0.0, c['ndm'] - c_ * c['ndh']))))
# all nights (not only <= 7): how many true fall below the k-th best false
allpos = [c for k, c in A if k == 1]; allneg = [c for k, c in A if k == 0] + [c for k, c in U if c['p'] < 0.9]
for name, f in (('current', lambda c: lg(c['p'])), ('new', lambda c: lg(c['p']) + 0.7 * (0.5 * min(c['ndh'], 5) - 1.2 * max(0.0, c['ndm'] - 0.15 * c['ndh'])))):
    a = np.sort([f(c) for c in allpos])[::-1]; b = np.sort([f(c) for c in allneg])[::-1]
    print(name, 'true above the 5th/20th/50th false:', [(int((a > b[k]).sum())) for k in (4, 19, 49)], 'of', len(a))
