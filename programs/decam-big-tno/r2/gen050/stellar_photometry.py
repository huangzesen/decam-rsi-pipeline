"""Tie the photometry of overlapping visits to a common stellar reference."""
from pathlib import Path

import numpy as np

from reference_field import AS, project


def median_by(group, value, size):
    count = np.bincount(group, minlength=size)
    med = np.full(size, np.nan)
    if len(group):
        ordered = value[np.lexsort((value, group))]
        occupied = count > 0
        end = count.cumsum()[occupied]
        n = count[occupied]
        med[occupied] = .5 * (ordered[end-1-n//2] + ordered[end-(n+1)//2])
    return med, count


class StellarPhotometry:
    def __init__(self, directory, exposures, frame):
        path = Path(directory)
        load = lambda k: np.load(path / ('det_' + k + '.npy'), mmap_mode='r')
        ex, sn, sharp, flag, flux, ra, dec = [load(k) for k in
            ('exp', 'snr', 'sharp', 'flag', 'flux', 'ra', 'dec')]
        edges = np.searchsorted(ex, np.arange(len(exposures) + 1))
        cells, instrumental, visits = [], [], []
        for e, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            s, h = sn[lo:hi], sharp[lo:hi]
            index = np.flatnonzero((s > 19.) & (s < 420.) & (h > .69) & (h < 1.45) &
                                   (flag[lo:hi] == 0))
            if not len(index):
                continue
            rows = lo + index
            xy = project(ra[rows], dec[rows], frame) / (1.45 * AS)
            ij = np.floor(xy).astype('i8')
            key = (ij[:, 0] << 25) + ij[:, 1]
            if len(rows) > 5600:
                # Position-based thinning selects the same stars on different
                # visits. A row-based sample would lose most of the overlaps.
                with np.errstate(over='ignore'):
                    u = key.astype('u8') * np.uint64(0xbf58476d1ce4e5b9)
                    u = (u ^ (u >> 31)) * np.uint64(0x94d049bb133111eb)
                keep = np.argpartition(u, 5599)[:5600]
                key, rows = key[keep], rows[keep]
            cells.append(key)
            instrumental.append(-2.5 * np.log10(np.maximum(flux[rows], 1.e-9)))
            visits.append(np.full(len(rows), e, 'i4'))
        self.bands = np.array([a['band'] for a in exposures])
        self.ex = np.concatenate(visits) if visits else np.empty(0, 'i4')
        self.mi = np.concatenate(instrumental) if visits else np.empty(0)
        cell = np.concatenate(cells) if visits else np.empty(0, 'i8')
        self.keys, self.star = np.unique(cell, return_inverse=True)

    def calibrate(self, zp):
        zp = np.array(zp, dtype=float, copy=True)
        ex, star, mi, bands = self.ex, self.star, self.mi, self.bands
        nstar, ne = len(self.keys), len(bands)
        if nstar < 30:
            return zp, {}
        names, band = np.unique(bands, return_inverse=True)
        group = star * len(names) + band[ex]
        _, same = np.unique(group, return_inverse=True)
        ns = same.max() + 1
        initial = zp.copy()
        for _ in range(3):
            mag = mi + zp[ex]
            good = np.isfinite(mag)
            ref, num = median_by(same[good], mag[good], ns)
            use = (num[same] >= 2) & np.isfinite(mi)
            estimate, count = median_by(ex[use], (ref[same] - mi)[use], ne)
            new = (count >= 18) & ~np.isfinite(zp)
            zp[new] = estimate[new]
            # Shrink small-overlap corrections toward the catalogue scale.
            ok = (count >= 12) & np.isfinite(estimate + zp)
            zp[ok] += .85 * np.clip(estimate[ok] - zp[ok], -.65, .65) * count[ok]/(count[ok]+25.)
        # A broad filter with disconnected or missing zeropoints can be
        # placed on an r-equivalent scale using the stars with r photometry.
        # Empirically fit its colour term from g-r and use a reference colour
        # in the middle of the disk population. IR-only coverage is left on
        # its measured scale.
        mag = mi + zp[ex]
        lookup = {b: j for j, b in enumerate(names)}
        gr = np.full(nstar, np.nan)
        def magnitude(b):
            use = (band[ex] == lookup[b]) & np.isfinite(mag)
            return median_by(star[use], mag[use], nstar)[0]
        if 'r' in lookup:
            red = magnitude('r')
            if 'g' in lookup:
                gr = magnitude('g') - red
            for b in names:
                select = self.bands[ex] == b
                if b == 'r' or not np.any(select):
                    continue
                measured = magnitude(b)
                colors = measured - red
                valid = np.isfinite(gr + colors) & (gr > .20) & (gr < 1.20)
                # Use stars with measured colours to predict missing visits.
                # The fitted locus is only used across g,r,i,z,Y. An ordinary
                # science visit with its own zeropoint keeps its band system.
                if b == 'g':
                    continue
                slope = 0.
                intercept = 0.
                if valid.sum() >= 45:
                    x = gr[valid] - .65
                    y = colors[valid]
                    w = np.ones(len(x))
                    for _ in range(4):
                        mat = np.array([[w.sum(), w@x], [w@x, w@(x*x)+.2]])
                        intercept, slope = np.linalg.solve(mat, [w@y,w@(x*y)])
                        w = np.minimum(1., .075/np.maximum(.001, abs(y-intercept-slope*x)))
                elif b != 'VR':
                    continue
                use = select & np.isfinite(red[star]) & np.isfinite(mi)
                if b != 'VR':
                    use &= np.isfinite(gr[star])
                predicted = red[star] + slope * (np.clip(np.nan_to_num(gr[star],nan=.65),.1,1.5)-.65)
                if b != 'VR':
                    predicted += intercept
                deltas, count = median_by(ex[use], (predicted-mi)[use], ne)
                covered = (bands == b) & (count >= 25)
                missing = covered & ~np.isfinite(zp)
                zp[missing] = deltas[missing]
                if b == 'VR':
                    # The common synthetic passband is equal to r at the
                    # reference colour. Each overlapping visit determines
                    # its own offset (clouds and pipeline conventions vary).
                    ok = covered & np.isfinite(zp)
                    # The finder will subtract the nominal VR colour later.
                    shift = .11 + deltas[ok] - zp[ok]
                    zp[ok] += np.clip(shift, -1.25, 1.25) * count[ok]/(count[ok]+20.)
        finite = np.isfinite(initial + zp)
        moved = finite & (abs(initial-zp) > .10)
        return zp, dict(stars=nstar, positions=len(mi), shifted=int(moved.sum()),
                        filled=int((np.isfinite(zp)&~np.isfinite(initial)).sum()))


def start(directory, exposures):
    from concurrent.futures import ThreadPoolExecutor
    ra, dec = np.deg2rad([[e['ra0'], e['dec0']] for e in exposures]).T
    unit = np.column_stack((np.cos(ra)*np.cos(dec),np.sin(ra)*np.cos(dec),np.sin(dec)))
    center = np.nanmean(unit,axis=0)
    center /= np.linalg.norm(center)
    east = np.cross([0.,0.,1.],center)
    east /= max(1.e-12,np.linalg.norm(east))
    frame = np.stack((east,np.cross(center,east),center))
    pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='stellar-colours')
    future = pool.submit(StellarPhotometry,directory,exposures,frame)
    pool.shutdown(wait=False)
    return future
