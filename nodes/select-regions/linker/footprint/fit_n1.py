"""Predict each DECam exposure's single-detection count n1 (measurements whose NSC object has ndet <= 1) from its
exposure-table row, fitted on the random sample (fetch_exposures.py): least squares of log10(n1 + 10) on log nmeas,
log coverage, log exptime, |b|, fwhm and depth95, back-transformed with Duan's smearing factor so that the mean is
unbiased. Prints 10-fold cross-validated sums, then adds `n1_pred` (NaN for other instruments) to EXPOSURES.npz.

Usage: python -I footprint/fit_n1.py EXPOSURES.npz SAMPLE.csv
"""
import csv, sys
import numpy as np


def features(X, ix):
    nm, cov, tex = X["nmeas"][ix], X["cover"][ix], X["exptime"][ix]
    b, fw, dep = np.abs(X["b"][ix]), X["fwhm"][ix], X["depth95"][ix]
    fw = np.where(np.isfinite(fw), fw, 1.2); dep = np.where(np.isfinite(dep), dep, 23.0)
    return np.c_[np.ones(len(ix)), np.log10(np.maximum(nm, 1)), np.log10(np.maximum(cov, 1)), np.log10(np.maximum(tex, 1)),
                 np.minimum(b, 60) / 30, fw, dep - 23]


def fit(F, y):
    c, *_ = np.linalg.lstsq(F, y, rcond=None)
    r = y - F @ c
    return c, np.mean(10 ** r)


def predict(F, c, smear):
    return np.maximum((10 ** (F @ c)) * smear - 10, 0)


def main():
    path, sample = sys.argv[1], sys.argv[2]
    X = dict(np.load(path))
    pos = {n: i for i, n in enumerate(X["name"])}
    S = list(csv.DictReader(open(sample)))
    ix = np.array([pos[r["exposure"]] for r in S]); n1 = np.array([float(r["n1"]) for r in S])
    F, y = features(X, ix), np.log10(n1 + 10)
    c, sm = fit(F, y)
    r = y - F @ c
    print(f"{len(S)} sampled exposures; coefficients {np.round(c, 3).tolist()}; residual sd {r.std():.2f} dex; smear {sm:.2f}")
    rng = np.random.default_rng(0); fold = rng.permutation(len(S)) % 10
    pred = np.zeros(len(S))
    for k in range(10):
        tr, te = fold != k, fold == k
        ck, sk = fit(F[tr], y[tr]); pred[te] = predict(F[te], ck, sk)
    print(f"10-fold CV: sum predicted / sum true = {pred.sum() / n1.sum():.3f}; "
          f"sum of squares (pair-weighted) predicted / true = {(pred ** 2).sum() / (n1 ** 2).sum():.3f}")
    for lo, hi in [(1, 50), (50, 200), (200, 1e9)]:
        m = (X["cover"][ix] >= lo) & (X["cover"][ix] < hi)
        print(f"  cover {lo:.0f}-{hi:.0g}: {m.sum()} exposures, CV predicted/true sum {pred[m].sum() / n1[m].sum():.3f}")
    allc = np.flatnonzero(X["inst"] == "c4d")
    out = np.full(len(X["name"]), np.nan); out[allc] = predict(features(X, allc), c, sm)
    X["n1_pred"] = out
    np.savez_compressed(path, **X)
    print(f"n1_pred for {len(allc)} DECam exposures: total {np.nansum(out):.3g}, median {np.nanmedian(out[allc]):.0f}")


if __name__ == "__main__":
    main()
