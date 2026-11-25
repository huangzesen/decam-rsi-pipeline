"""slots.py CAND.json DST [NSUB] [NTRIAL] [MULT]: the task's ranking on a patch with a realistic number of bodies (gen021).
A planted run holds ~800 scored bodies and one patch's chance crowd, so "found" hardly depends on p.  Here a random NSUB
of the scored planted bodies are "the known ones" (default 25), the candidates of all other planted bodies are dropped,
every candidate matching no planted body stays (the patch's chance crowd and its own real bodies), and the slots are
2 x NSUB.  Prints the share of the NSUB found inside the slots (mean over NTRIAL draws) and the share found at any rank."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); T = np.load(sys.argv[2] + '/truth.npz')
NSUB = int(sys.argv[3]) if len(sys.argv) > 3 else 25; NT = int(sys.argv[4]) if len(sys.argv) > 4 else 400
MULT = int(sys.argv[5]) if len(sys.argv) > 5 else 1      # the candidates matching nothing counted MULT times (a denser patch)
E = json.load(open(sys.argv[2] + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
ntot = int(T['ntot']); body = -np.ones(ntot, np.int64); body[T['rows']] = T['body']
rexp = np.zeros(ntot, np.int64); rexp[T['rows']] = T['exp']
scored = np.flatnonzero(T['nights'] >= 5)
cands = sorted(cands[:2000], key=lambda c: -c['p'])
own = []; ok = []                      # per candidate: planted body it is mostly made of (-1: none), and whether it finds it
for c in cands:
    d = np.array(c['det']); b = body[d]; bb = b[b >= 0]
    if len(bb) == 0:
        own.append(-1); ok.append(False); continue
    u, cnt = np.unique(bb, return_counts=True); j = int(u[cnt.argmax()])
    own.append(j)
    ok.append(bool(cnt.max() >= 0.8 * len(d) and len(np.unique(night[rexp[d[b == j]]])) >= 5))
own = np.array(own); ok = np.array(ok)
rng = np.random.default_rng(5)
fs, fa = [], []
for _ in range(NT):
    sub = set(rng.choice(scored, NSUB, replace=False).tolist())
    keep = np.array([(o == -1) or (o in sub) for o in own])
    idx = np.flatnonzero(keep)
    # slots taken in order of p; a candidate matching nothing takes MULT slots (MULT copies of the crowd)
    cost = np.where(own[idx] == -1, MULT, 1); top = idx[np.cumsum(cost) <= 2 * NSUB]
    fs.append(len(set(own[top][ok[top]].tolist()))); fa.append(len(set(own[idx][ok[idx]].tolist())))
print(f"{NSUB} known bodies, {2 * NSUB} slots, {int((own == -1).sum())} candidates matching nothing: found inside the slots {np.mean(fs) / NSUB:.3f}, at any rank {np.mean(fa) / NSUB:.3f}")
