"""Magnitude scatter of the confident candidates of an unplanted run under the listed and the calibrated zero points,
with the fixed colours of gen001-gen010 and with the fitted redness.  realmag.py candidates.json SCEN [pmin]"""
import sys, json, numpy as np
sys.path.insert(0, '/app/submission')
import finder
from finder import zpcal, colfit, COL0, COLC, COLS
cands = json.load(open(sys.argv[1])); D = sys.argv[2]; pmin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
E = json.load(open(D + '/exposures.json'))
mz = np.array([e['magzero'] if e['magzero'] is not None else np.nan for e in E], float)
band = [e['band'] for e in E]
mzc, dz, nst = zpcal(D, E)
ex = np.load(D + '/det_exp.npy', mmap_mode='r'); fl = np.load(D + '/det_flux.npy', mmap_mode='r'); sn = np.load(D + '/det_snr.npy', mmap_mode='r')
c0 = np.array([COL0.get(b, 0.0) for b in band]); cc = np.array([COLC.get(b, 0.0) for b in band]); cs = np.array([COLS.get(b, 0.0) for b in band])
sd = lambda v: float(np.std(v[np.isfinite(v)] - np.median(v[np.isfinite(v)]))) if np.isfinite(v).sum() >= 3 else np.nan
print('p nights r | members without listed zero point, with |dz| > 0.2 | scatter: listed zp + fixed colours, calibrated zp + fixed colours, calibrated + fitted redness (dq)')
for c in cands:
    if c['p'] < pmin: continue
    d = np.array(c['det']); e = np.asarray(ex[d]); m_raw = -2.5 * np.log10(np.asarray(fl[d]).astype(float)); s = np.asarray(sn[d]).astype(float)
    m0 = mz[e] + m_raw - c0[e]; m1 = mzc[e] + m_raw - c0[e]; m2 = mzc[e] + m_raw - cc[e]
    dq = colfit(m2, cs[e], s)
    print(f"{c['p']:.3f} {c['nights']:2d} {c['r']:6.2f} | {int(np.isnan(mz[e]).sum())} {int((np.abs(dz[e]) > 0.2).sum())} of {len(d)} | {sd(m0):.3f} {sd(m1):.3f} {sd(m2 - dq * cs[e]):.3f} (dq {dq:+.2f})  bands {''.join(sorted(set(band[k] for k in e)))}")
