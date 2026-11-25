import json, sys, numpy as np
out = sys.argv[1]; scen = sys.argv[2] if len(sys.argv) > 2 else '/app/data/sample'; cache = sys.argv[3] if len(sys.argv) > 3 else '/tmp/w/cache_s.npz'
C = json.load(open(out + '/candidates.json'))
E = json.load(open(scen + '/exposures.json')); et = np.array([e['mjd_mid'] for e in E]); night = np.floor(et - 0.5).astype(int)
Z = np.load(cache); keep, seed = Z['keep'], Z['seed']
snr = np.load(scen + '/det_snr.npy', mmap_mode='r'); flag = np.load(scen + '/det_flag.npy', mmap_mode='r'); sharp = np.load(scen + '/det_sharp.npy', mmap_mode='r')
exp = np.load(scen + '/det_exp.npy', mmap_mode='r')
isseed = np.zeros(len(exp), bool); isseed[keep[seed]] = True
print('n cands', len(C))
ps = np.array([c['p'] for c in C]); print('p hist', np.histogram(ps, bins=[0, .05, .1, .2, .3, .5, .7, .9, .99, 1.0001])[0])
tot = dict(n=0, seed=0, lowsnr=0, flag=0, sharp=0, other=0)
for k, c in enumerate(C):
    d = np.array(c['det']); s = np.asarray(snr[d]); f = np.asarray(flag[d]); sh = np.asarray(sharp[d]); sd = isseed[d]
    sure = c['p'] >= 0.9 and c['nights'] >= 6
    if k < 400:
        print(f"{k:3d} p={c['p']:.3f} n={c['nights']:2d} s={c['seasons']} nd={len(d):2d} rms={c['rms']:.2f} r={c['r']:6.1f} arc={c['arc']:7.1f} msc={c['msc']:.2f} hv={c['hv']:.1f} src={c['src']} fa={c['fa']:.1e} snr_med={np.median(s):.1f} seedfrac={sd.mean():.2f} seednights={len(np.unique(night[exp[d[sd]]]))} flag!=0:{(f != 0).sum()} lowsnr:{(s < 5).sum()}")
    if sure:
        tot['n'] += len(d); tot['seed'] += sd.sum(); ns = ~sd
        tot['lowsnr'] += (ns & (s < 5)).sum(); tot['flag'] += (ns & (s >= 5) & (f != 0)).sum()
        tot['sharp'] += (ns & (s >= 5) & (f == 0) & ((sh < 0.5) | (sh > 1.8))).sum()
        tot['other'] += (ns & (s >= 5) & (f == 0) & (sh >= 0.5) & (sh <= 1.8)).sum()
print('sure-real detections:', tot)
