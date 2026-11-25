"""ucand.py CAND.json SCEN [PMIN]: candidates of an unplanted run with their night pattern per season."""
import sys, json, numpy as np
cands = json.load(open(sys.argv[1])); scen = sys.argv[2]; pmin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
E = json.load(open(scen + '/exposures.json')); mjd = np.array([e['mjd_mid'] for e in E]); night = np.floor(mjd - 0.5).astype(int)
dexp = np.load(scen + '/det_exp.npy', mmap_mode='r'); snr = np.load(scen + '/det_snr.npy', mmap_mode='r')
ps = np.array([c['p'] for c in cands])
print(len(cands), 'candidates; p >= 0.9 / 0.5 / 0.2 / 0.05:', [(ps >= t).sum() for t in (0.9, 0.5, 0.2, 0.05)])
for c in cands:
    if c['p'] < pmin: break
    d = np.array(c['det']); un = np.unique(night[dexp[d]]); g = np.flatnonzero(np.diff(un) > 120)
    per = [len(p) for p in np.split(un, g + 1)]
    print(f"p {c['p']:.3f} nights {c['nights']:2d} per season {per} r {c['r']:6.2f} rms {c['rms']:.3f} msc {c['msc']:.2f} hv {c['hv']:.1f} nd {c['ndh']}/{c['ndm']}/{c['ndo']} src {c['src']} S/N med {float(np.median(snr[d])):.1f} first row {int(d.min())}")
