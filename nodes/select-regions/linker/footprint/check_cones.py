"""Cone-measured single-detection counts (fetch_cones.py) against exact per-exposure counts (sample_n1.csv,
validate_*.csv from the meas table): for every exactly counted exposure that a cone measured, the ratio
cone estimate / exact, and the same for the regression's prediction.

Usage: python -I footprint/check_cones.py EXPOSURES.npz CONES.jsonl EXACT.csv [...]
"""
import csv, json, sys
import numpy as np

R0 = 1.1
INSIDE = 0.85                         # deg


def main():
    X = np.load(sys.argv[1])
    pos = {n: i for i, n in enumerate(X["name"])}
    est = {}
    ra, dec = np.radians(X["ra"]), np.radians(X["dec"])
    for line in open(sys.argv[2]):
        q = json.loads(line)
        r0, d0 = np.radians(q["ra"]), np.radians(q["dec"])
        for k, v in q["counts"].items():
            k = int(k)
            sep = np.degrees(np.arccos(min(1, np.sin(d0) * np.sin(dec[k]) + np.cos(d0) * np.cos(dec[k]) * np.cos(ra[k] - r0))))
            if sep <= INSIDE:                    # the cone lies inside the exposure's disk
                est.setdefault(k, []).append(v / (np.pi * q["radius"] ** 2) * np.pi * R0 ** 2)
    ex = {}
    for f in sys.argv[3:]:
        for r in csv.DictReader(open(f)):
            ex[pos[r["exposure"]]] = float(r["n1"])
    both = [i for i in ex if i in est]
    c = np.array([np.mean(est[i]) for i in both]); t = np.array([ex[i] for i in both]); p = X["n1_pred"][both]
    print(f"{len(ex)} exact counts, {len(both)} also in a cone")
    if both:
        r = np.log10((c + 10) / (t + 10)); rp = np.log10((p + 10) / (t + 10))
        print(f"cone / exact: sum {c.sum() / t.sum():.3f}, median {np.median(10 ** r):.2f}, scatter {r.std():.2f} dex")
        print(f"regression / exact (same exposures): sum {p.sum() / t.sum():.3f}, median {np.median(10 ** rp):.2f}, scatter {rp.std():.2f} dex")


if __name__ == "__main__":
    main()
