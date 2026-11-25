"""Local intensity of discovery peaks conditional on flux and usable chip area."""
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr


def data(directory):
    root = Path(directory)
    return {k: np.load(root / ('det_' + k + '.npy'), mmap_mode='r')
            for k in ('x', 'y', 'ccd', 'exp', 'flux', 'ferr', 'snr')}


def locations(cat, rows):
    # Four broad subregions per amplifier along the long side of the CCD.
    chip = cat['ccd'][rows].astype('i4')
    x = np.clip(cat['x'][rows].astype('i4') // 256, 0, 7)
    y = np.clip(cat['y'][rows].astype('i4') // 256, 0, 15)
    return (chip * 8 + x) * 16 + y


def ratio(cat, edges, e, peaks, wanted):
    """Log change in signal/clutter intensity relative to this exposure.

    The detection efficiency and the observed count use the same magnitude
    kernel. This matters on a noisy amplifier and at a spatial subset edge.
    """
    start, end = edges[e:e+2]
    if end <= start or len(peaks) < 24 or not len(wanted):
        return np.zeros(len(wanted), 'f4')
    sl = slice(start, end)
    loc = locations(cat, sl)
    total = int(loc.max() // 128 + 1) * 128
    area_counts = np.bincount(loc, minlength=total)
    populated = area_counts[area_counts > 0]
    typical = max(2., np.median(populated))
    area = np.minimum(1., area_counts / max(1.5, .32 * typical))
    err = cat['ferr'][sl]
    sn = cat['snr'][sl]
    use = (sn < 15.) & np.isfinite(err) & (err > 0.)
    if use.sum() < 12:
        return np.zeros(len(wanted), 'f4')
    pivot = np.median(err[use])
    noise = np.clip(np.log(np.maximum(1.e-9, err[use]) / pivot), -2., 2.)
    counts = np.bincount(loc[use], minlength=total)
    logs = np.bincount(loc[use], weights=noise, minlength=total)
    amp = np.arange(total) // 64
    na = np.bincount(amp, weights=counts)
    amp_log = np.bincount(amp, weights=logs) / (na + 8.)
    local_log = (logs + 8. * amp_log[amp]) / (counts + 8.)
    # Magnitudes relative to the local survey limit share the same bins in
    # shallow and deep visits. Bins beyond the brightest one are pooled.
    step = .50
    mags = np.arange(-8., 2.01, step)
    def magnitude(rows):
        f = np.maximum(cat['flux'][rows], 1.e-9)
        return np.nan_to_num(-2.5 * np.log10(f / (pivot * 5.)), nan=2.)
    def bin_index(m):
        return np.clip(np.rint((m - mags[0]) / step).astype('i4'), 0, len(mags)-1)
    mb = bin_index(magnitude(peaks))
    pix = loc[peaks - start]
    shape = (total//128, 8, 16, len(mags))
    hist = np.bincount(pix * len(mags) + mb, minlength=total*len(mags)).reshape(shape)
    hist = gaussian_filter(hist.astype('f4'), (0., 0., 0., .82), mode='nearest')
    sn_pred = 5. * np.exp(-local_log[:, None]) * 10.**(-.4 * mags)
    efficiency = ndtr((sn_pred - 4.5) / np.hypot(1.15, .13 * sn_pred))
    # Background peaks close to threshold have biased fluxes. Broadening
    # the selection function avoids dividing by a tiny model probability.
    exposure = (area[:, None] * (.12 + .88 * efficiency)).reshape(shape).astype('f4')
    h_all = hist.sum((0, 1, 2))
    a_all = exposure.sum((0, 1, 2))
    mean = (h_all + 1.) / np.maximum(1., a_all)
    h_chip = hist.sum((1, 2), keepdims=True)
    a_chip = exposure.sum((1, 2), keepdims=True)
    chip_density = (h_chip + 12.) / (a_chip + 12. / mean)
    h_local = gaussian_filter(hist, (0., .92, .92, 0.), mode='constant') * 10.
    a_local = gaussian_filter(exposure, (0., .92, .92, 0.), mode='constant') * 10.
    local_density = (h_local + 4.) / (a_local + 4. / chip_density)
    at = loc[wanted-start]
    mass = bin_index(magnitude(wanted))
    # Remove the event being evaluated from the central spatial kernel in
    # a sparsity-safe way. Its contribution is at most one original event.
    indices = np.searchsorted(peaks, wanted).clip(0,len(peaks)-1)
    counted = (peaks[indices] == wanted)
    centre = h_local.reshape(total,-1)[at,mass]
    correction = np.log1p(-np.minimum(.80,.91*counted/(centre+4.)))
    value = -np.log(local_density.reshape(total, -1)[at, mass] / mean[mass]) - correction
    return np.clip(value, -4., 2.8).astype('f4')


def measure(directory, peaks, rows):
    cat = data(directory)
    e = cat['exp']
    peaks = np.asarray(peaks, 'i8')
    order = np.argsort(rows, kind='stable')
    rows = np.asarray(rows, 'i8')[order]
    if not len(rows) or not len(peaks):
        return np.zeros(len(rows), 'f4')
    if np.any(peaks[1:] < peaks[:-1]):
        peaks = np.sort(peaks)
    edges = np.searchsorted(e, np.arange(int(e[-1]) + 2))
    a = np.searchsorted(peaks, edges)
    b = np.searchsorted(rows, edges)
    result = np.zeros(len(rows), 'f4')
    for i in np.flatnonzero(np.diff(b)):
        sl = slice(b[i], b[i+1])
        result[order[sl]] = ratio(cat, edges, i, peaks[a[i]:a[i+1]], rows[sl])
    return result


def start(directory, rows, selected):
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='camera-background')
    future = pool.submit(measure, directory, rows[selected], rows)
    pool.shutdown(wait=False)
    return future
