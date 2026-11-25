"""Every ITF observation (MPC isolated tracklet file, newest rsi/data/mpc/<date>/itf.txt.gz) along each body's whole predicted path,
2013 to the file's date: flagged if it lies within max(30", 5 sigma) of the prediction at its own time (prediction 1-sigma < 300").
Fills the gap found 2026-11-17 (LEAD12 = Rubin X05 trkSub RUBIN_TRKSUB): novelty_check.py looks at the ITF only within 0.6 day of the
candidate's own nights, and its Rubin-era step (5) repeats the MPCORB screen, not the ITF one.
Usage: ~/anaconda3/bin/python -I itf_path_check.py BODIES.json OUT.json   (BODIES: [{"id", "arc"}], arc = arc file path)"""
import ast, glob, gzip, json, math, os, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.expanduser("$ALE_HOME/planet-search/linker"))
import orbit_fit as OF
from orbits import unit_to_radec
B = json.load(open(sys.argv[1])); OUT = sys.argv[2]
ITF = sorted(glob.glob(os.path.expanduser("$ALE_HOME/planet-search/rsi/data/mpc/20*-*-*/itf.txt.gz")))[-1]
lit = lambda v: ast.literal_eval(v) if isinstance(v, str) else v
import datetime
MJD0 = datetime.date(1858, 11, 17).toordinal()
def mjd(y, mo, d):
    return datetime.date(y, mo, 1).toordinal() - MJD0 + d - 1
T, RA, DE, ST, TK = [], [], [], [], []
with gzip.open(ITF, "rt", errors="replace") as fh:
    for l in fh:
        if len(l) < 80 or l[14] in "sSrRvxX" or not l[15:19].isdigit():
            continue
        try:
            y = int(l[15:19])
            if y < 2012:
                continue
            m = mjd(y, int(l[20:22]), float(l[23:32])); ra = 15 * (int(l[32:34]) + int(l[35:37]) / 60 + float(l[38:44]) / 3600)
            de = (-1 if l[44] == "-" else 1) * (int(l[45:47]) + int(l[48:50]) / 60 + float(l[51:56]) / 3600)
        except ValueError:
            continue
        T.append(m); RA.append(ra); DE.append(de); ST.append(l[77:80]); TK.append(l[5:12].strip())
T, RA, DE = np.array(T), np.array(RA), np.array(DE); ST = np.array(ST); TK = np.array(TK)
print(ITF, len(T), "observations from 2012", flush=True)
res = {}
for b in B:
    A = json.load(open(b["arc"])); p = np.array(lit(A["p"]), float); cov = np.array(lit(A["cov"]), float); t0 = float(A["epoch_tdb"])

    def where(m):
        R, tdb = OF.observer_bary(np.atleast_1d(np.asarray(m, float)), "W84"); u0, s = OF.predict_sigma(p, cov, t0, np.atleast_1d(tdb), np.atleast_2d(R)); ra, de = unit_to_radec(u0)
        return np.degrees(ra) % 360, np.degrees(de), s
    g = np.arange(56300., T.max() + 5, 2.0); gr, gd, gs = where(g)
    k = np.clip(np.searchsorted(g, T), 1, len(g) - 1)
    near = (np.abs(DE - gd[k]) < 0.5) & (np.abs(((RA - gr[k] + 180) % 360 - 180) * np.cos(np.radians(DE))) < 0.5) & (gs[k] < 300)
    idx = np.where(near)[0]; hits = []
    if len(idx):
        a, d, s = where(T[idx]); sep = np.hypot(((RA[idx] - a + 180) % 360 - 180) * np.cos(np.radians(d)), DE[idx] - d) * 3600
        ok = sep < np.maximum(30, 5 * s)
        for i, sp, sg in zip(idx[ok], sep[ok], s[ok]):
            hits.append(dict(mjd=round(float(T[i]), 5), station=str(ST[i]), trksub=str(TK[i]), sep=round(float(sp), 2), sigma=round(float(sg), 2)))
    by = defaultdict(list)
    for h in hits:
        by[(h["station"], h["trksub"])].append(h)
    res[b["id"]] = dict(n=len(hits), tracklets=[dict(station=k[0], trksub=k[1], n=len(v), first=min(x["mjd"] for x in v), last=max(x["mjd"] for x in v),
                                                    max_sep=max(x["sep"] for x in v), med_sigma=float(np.median([x["sigma"] for x in v]))) for k, v in sorted(by.items(), key=lambda kv: min(x["sep"] for x in kv[1]))])
    print(b["id"][:60], "ITF hits", len(hits), [(t["station"], t["trksub"], t["n"], t["max_sep"]) for t in res[b["id"]]["tracklets"][:5]], flush=True)
json.dump(dict(itf=ITF, bodies=res), open(OUT, "w"), indent=1); print("wrote", OUT)
