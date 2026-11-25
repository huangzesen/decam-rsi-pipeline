# offline re-rank variants from dump/chi_feats.npz (truth used only for scoring)
import sys, json, numpy as np
sys.path.insert(0, '$RSI_SANDBOX/r1/gen089/tmp/armB/stackd')
sys.path.insert(0, '$RSI_SANDBOX/r1/gen089/tmp/armB/offline')
import ev
import importlib.util as iu
sp=iu.spec_from_file_location('fa','$RSI_SANDBOX/r1/gen089/tmp/armB/stackd/finder.py'); fa=iu.module_from_spec(sp); sp.loader.exec_module(fa); chi_stats=fa.chi_stats
D = '$RSI_SANDBOX/r1/gen089/tmp/armB/dump'
z = np.load(D + '/chi_feats.npz', allow_pickle=True)
vC, cC, vN, cN, wf = z['vC'].astype(float), z['cC'], z['vN'].astype(float), z['cN'], z['wf']
ni = [np.asarray(i, int) for i in z['nights']]
zx, zp, pv, S, p0 = z['zx'], z['zp'], z['pv'], z['states'], z['p']
pre = json.load(open(D + '/pre.json'))['candidates']
# pre.json was written with n3boost (p=max(p,pn)), candidates in same order? map by state
key0 = {tuple(np.round(c['state'], 9)): c for c in pre}
cands = [key0.get(tuple(np.round(s, 9)), dict(epoch_mjd=pre[0]['epoch_mjd'], state=list(s), p=0)) for s in S]
nz = lambda a, n: (a - np.median(n)) / (1.4826 * np.median(np.abs(n - np.median(n))) + 1e-3)
def out(key):
    o = np.argsort(-key, kind='stable')
    cs = []
    for k, j in enumerate(o):
        d = dict(cands[j]); d['p'] = float(pv[k])
        if 'pn' in d: d['p'] = max(d['p'], d['pn'])
        cs.append(d)
    return ev.ev(cs)
bC, XC = chi_stats(vC, cC, wf, ni); bN, XN = chi_stats(vN, cN, wf, ni)
zb = nz(bC, bN); pen = np.maximum(0, nz(XC, XN)); zX = nz(XC, XN)
def wins(v, c, k=2.0):
    # winsorised stack: robust flux by median ratio, clip each exposure at F w + k
    vv = np.where(c, v, 0.0); r = np.where(c & (wf[None] > 0), vv / wf[None], np.nan)
    F = np.nanmedian(r, 1); F = np.where(np.isfinite(F), F, 0)
    vc = np.minimum(vv, F[:, None] * wf[None] + k)
    return (vc * wf[None]).sum(1) / np.sqrt(np.maximum((c * wf[None] ** 2).sum(1), 1e-3))
def mednight(v, c):
    vv = np.where(c, v, 0.0); num = vv * wf[None]; den = c * wf[None] ** 2
    F = np.stack([np.where(den[:, i].sum(1) > 0, num[:, i].sum(1) / np.maximum(den[:, i].sum(1), 1e-6), np.nan) for i in ni], 1)
    return np.nanmedian(F, 1) * np.sqrt(den.sum(1))
zw = nz(wins(vC, cC), wins(vN, cN)); zm = nz(mednight(vC, cC), mednight(vN, cN))
print('baseline(lono w.75)', out(0.75 * zx + 0.25 * zp))
for w in (0.5, 0.75, 1.0):
    for nm, st in [('zb-pen', zb - pen), ('zb', zb), ('zx-pen', zx - pen), ('zb-.5pen', zb - .5 * pen),
                   ('wins2', zw), ('wins2-pen', zw - pen), ('mednight', zm), ('mednight-pen', zm - pen), ('zx-.5pen', zx - .5 * pen)]:
        print(nm, w, out(w * st + (1 - w) * zp))

import csv
sys.path.insert(0,'$RSI_SANDBOX/r1/gen089/workspace/submission/experiments')
import score_inject as si
cs,res,hr,dup=si.score(ev.truth,[dict(c,p=-k) for k,c in enumerate([cands[j] for j in np.argsort(-(0.75*zx+0.25*zp),kind='stable')])],ev.rows)
o=np.argsort(-(0.75*zx+0.25*zp),kind='stable')
hit={o[r]:ev.truth['bodies'][jb]['cls'] for jb,(r,p) in hr.items()}
print('rank cls zp zx zb X pen')
for r in range(40):
    j=o[r]; print(r, hit.get(j,'-'), *[round(float(a[j]),2) for a in (zp,zx,zb,XC,pen)])
for r in range(40,200):
    j=o[r]
    if j in hit: print(r, hit[j], *[round(float(a[j]),2) for a in (zp,zx,zb,XC,pen)])
