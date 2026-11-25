import sys, time, json, math, numpy as np
sys.path.insert(0, '/app/submission')
import finder as F
D = '/tmp/p4/'
E = json.load(open(D + 'exposures.json')); nE = len(E)
et = np.array([e['mjd_mid'] for e in E]); night_e = np.floor(et - 0.5).astype(np.int64)
_, nidx = np.unique(night_e, return_inverse=True)
L = lambda k: np.load(D + f'det_{k}.npy', mmap_mode='r')
exp = np.asarray(L('exp')).astype(np.int64)
x, y, Fr = F.tangent(np.asarray(L('ra')), np.asarray(L('dec')))
cnt = F.static_count(x, y, nidx[exp].astype(np.int64), 'cpu')
snr, sharp, flag = np.asarray(L('snr')), np.asarray(L('sharp')), np.asarray(L('flag'))
sg = F.SHW / np.maximum(snr, 4.5)
shape = ((flag & 1) == 0) & (sharp >= np.minimum(0.35, 1 - 2.5 * sg)) & (sharp <= np.maximum(2.5, 1 + 2.5 * sg)) & (sharp != 0) & (snr >= 4.5)
keep = (cnt < 3) & shape
m5_ = np.array([e_.get("m5") if e_.get("m5") is not None else np.nan for e_ in E], float); m5m_ = float(np.nanmedian(m5_))
sb_ = 20.0 * 10 ** (0.4 * np.clip(np.nan_to_num(m5_ - m5m_, nan=0.0), 0.0, 3.5))
br_ = (snr >= sb_[exp]) & ((flag & 1) == 0)
cw_ = math.radians(F.CELLDEG)
cx_ = np.floor((x - x.min()) / cw_).astype(np.int64); cy_ = np.floor((y - y.min()) / cw_).astype(np.int64)
ncx_ = int(cx_.max()) + 1; ncy_ = int(cy_.max()) + 1
ck_ = (exp * ncx_ + cx_) * ncy_ + cy_
M = nE * ncx_ * ncy_
na_ = np.bincount(ck_[br_], minlength=M); nk_ = np.bincount(ck_[br_ & keep], minlength=M); n1_ = np.bincount(ck_[br_ & keep & (cnt < 2)], minlength=M)
badc = (na_ >= 15) & (nk_ > F.CELLCUT * na_)
print('exposure-cells with >= 15 bright sources:', int((na_ >= 15).sum()), ' failed:', int(badc.sum()))
f1 = n1_[badc] / na_[badc]
print('single-night share of the bright sources in the failed cells: pct 10/25/50/75/90', np.percentile(f1, [10, 25, 50, 75, 90]).round(3))
for thr in (0.05, 0.1, 0.15, 0.3):
    two = badc & (n1_ <= thr * na_)
    inb = two[ck_]
    print(f'  failed cells with single-night share <= {thr}: {int(two.sum())} ({two.sum() / badc.sum():.1%}); kept detections there {int((keep & inb).sum()):,}, of which single-night {int((keep & inb & (cnt < 2)).sum()):,}')
inbad = badc[ck_]
T = np.load(D + 'truth.npz'); rows = T['rows']; N0 = len(exp)
pl = np.zeros(len(exp), bool); pl[rows] = True
print('planted detections: %d; static-cut (cnt>=3) %.3f; cnt==2 %.3f; shape-cut %.3f; in failed cells %.3f' % (pl.sum(), np.mean(cnt[rows] >= 3), np.mean(cnt[rows] == 2), np.mean(~shape[rows]), np.mean(inbad[rows])))
for thr in (0.05, 0.1, 0.15, 0.3):
    two = badc & (n1_ <= thr * na_)
    print(f'  planted detections in failed cells with single-night share <= {thr}: {np.mean(two[ck_[rows]]):.3f} (single-night ones: {np.mean(two[ck_[rows]] & (cnt[rows] < 2)):.3f})')
# per body: nights lost to the cell cut
night = np.floor(et - 0.5).astype(int); bi = T['body']; ei = T['exp']
lostb = inbad[rows]
nb5 = 0; hurt = 0; hurt5 = 0; clsf = 0; clsf_hurt = 0
for b in np.unique(bi):
    m = bi == b
    n_all = len(np.unique(night[ei[m]]))
    if n_all < 5: continue
    nb5 += 1
    n_keep = len(np.unique(night[ei[m & ~lostb]]))
    if n_keep < n_all: hurt += 1
    if n_keep < 5: hurt5 += 1
    nb = np.unique(night[ei[m]]); g = np.flatnonzero(np.diff(nb) > 120); per = [len(p) for p in np.split(nb, g + 1)]
    if max(per) <= 2:
        clsf += 1; clsf_hurt += n_keep < n_all
print('scored bodies %d: lose a night to the cell cut %d, fall under 5 nights %d; class f %d, of which lose a night %d' % (nb5, hurt, hurt5, clsf, clsf_hurt))
np.savez('/tmp/w/cellstat.npz', badc=badc, n1=n1_, na=na_, nk=nk_)
