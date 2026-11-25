"""Detector-frame recurrence: detections that are not sky-static but keep coming back at the same CCD pixel.

The static cut works on the sky: a source seen at one sky position on three nights is a star or a galaxy.  A defect of
the camera (hot pixel, trap, glowing edge, bad-column end) is the opposite case: the pointings are dithered, so it lands
on a different sky position in every exposure, passes the static cut as a "one-night" source, and competes with the
bodies in every pool and queue of the search.  It is fixed in the frame the search never looked at: (CCD, x, y).

`recurrent(...)` flags the rows, among those the static cut left (`cand`), whose 2-pixel detector cell (either of two
half-offset grids) holds candidates from at least `min_nights` different nights.  A moving body would have to fall on the
same 0.5" of the same CCD on four nights to be flagged; a sky-static source under repeated identical pointings is not
among the candidates at all.  Counting nights, not exposures, leaves undithered within-night sequences alone.
"""
import numpy as np


def _cell_nights(ccd, x, y, night, cell, off):
    kx = np.floor(x / cell + off).astype(np.int64)
    ky = np.floor(y / cell + off).astype(np.int64)
    key = (ccd.astype(np.int64) * 8192 + kx) * 8192 + ky
    nn = int(night.max()) + 1 if len(night) else 1
    full = key * nn + night.astype(np.int64)
    order = np.argsort(full, kind="stable")       # one sort; everything after it is linear
    fs = full[order]
    c = fs // nn
    newcell = np.concatenate(([True], c[1:] != c[:-1]))
    newpair = np.concatenate(([True], fs[1:] != fs[:-1]))      # first row of every (cell, night)
    cid = np.cumsum(newcell) - 1
    cnt = np.bincount(cid, weights=newpair).astype(np.int64)   # different nights per cell
    out = np.empty(len(key), np.int64)
    out[order] = cnt[cid]
    return out


def recurrent(load, cand, night_of_row, cell=2.0, min_nights=4):
    """load(name) -> memory-mapped det_<name>.npy; cand: bool mask over all rows; night_of_row: int array over all rows.
    Returns a bool mask over all rows (True: detector-recurrent candidate)."""
    idx = np.flatnonzero(cand)
    out = np.zeros(len(cand), bool)
    if len(idx) == 0:
        return out
    ccd = np.asarray(load("ccd"))[idx]
    x = np.asarray(load("x"))[idx].astype(np.float64)
    y = np.asarray(load("y"))[idx].astype(np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > -1e4) & (x < 1e5) & (y > -1e4) & (y < 1e5)
    if not ok.all():
        idx, ccd, x, y = idx[ok], ccd[ok], x[ok], y[ok]
    x = x - min(0.0, float(x.min())) + cell
    y = y - min(0.0, float(y.min())) + cell
    ccd = ccd - int(ccd.min())
    night = np.asarray(night_of_row)[idx]
    n0 = _cell_nights(ccd, x, y, night, cell, 0.0)
    n1 = _cell_nights(ccd, x, y, night, cell, 0.5)
    out[idx[(n0 >= min_nights) | (n1 >= min_nights)]] = True
    return out
