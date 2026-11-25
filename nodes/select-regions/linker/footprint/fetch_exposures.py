"""Fetch the NSC DR2 exposure table (every instrument) in RA slices through the Data Lab query manager (anonymous),
and the number of single-detection measurements (NSC objects with ndet <= 1) for a random sample of DECam exposures.
Resumable: one CSV per slice / sample file, written to .part then renamed.

Usage: python -I footprint/fetch_exposures.py OUTDIR [--slice 10] [--sample 300] [--seed 1]
"""
import argparse, csv, io, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nsc_fetch import query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--slice", type=float, default=10.0)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "slices"), exist_ok=True)
    for r0 in np.arange(0, 360, a.slice):
        f = os.path.join(a.out, "slices", f"{r0:05.1f}.csv")
        if os.path.exists(f):
            continue
        t0 = time.time()
        txt = query(f"SELECT exposure, instrument, mjd, ra, dec, filter, exptime, depth95, fwhm, nmeas FROM nsc_dr2.exposure "
                    f"WHERE ra >= {r0} AND ra < {r0 + a.slice}")
        open(f + ".part", "w").write(txt); os.replace(f + ".part", f)
        print(f"ra {r0:5.1f}: {txt.count(chr(10)) - 1} exposures {time.time() - t0:.0f}s", flush=True)
    rows = [r for f in sorted(os.listdir(os.path.join(a.out, "slices"))) if f.endswith(".csv")
            for r in csv.DictReader(open(os.path.join(a.out, "slices", f)))]
    print(len(rows), "exposures in all", flush=True)
    dec = [r for r in rows if r["instrument"] == "c4d"]
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(dec), size=min(a.sample, len(dec)), replace=False)
    sf = os.path.join(a.out, "sample_n1.csv")
    done = {r["exposure"] for r in csv.DictReader(open(sf))} if os.path.exists(sf) else set()
    new = not os.path.exists(sf)
    with open(sf, "a") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["exposure", "n1", "nmeas"]); fh.flush()
        for k, i in enumerate(pick):
            e = dec[i]["exposure"]
            if e in done:
                continue
            t0 = time.time()
            txt = query(f"SELECT COUNT(*) AS n FROM nsc_dr2.meas m JOIN nsc_dr2.object o ON m.objectid = o.id "
                        f"WHERE m.exposure = '{e}' AND o.ndet <= 1")
            n = int(txt.strip().splitlines()[-1])
            w.writerow([e, n, dec[i]["nmeas"]]); fh.flush()
            print(f"sample {k + 1}/{len(pick)} {e} n1 {n} nmeas {dec[i]['nmeas']} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
