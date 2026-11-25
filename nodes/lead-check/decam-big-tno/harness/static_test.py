"""runs/s38/static/static_test.py for any run (2026-11-16): writes runs/<RUN>/static/<lead id>.json. Needs the lead's arc record
(harness/arc_extend.py RUN ...). Usage: DBT_SITE=nersc ~/anaconda3/bin/python -I harness/static_test.py RUN ID_SUBSTRING ...
Static-source test for a lead: for each of its detections, is there a source at the same sky place (< 1.5") in exposures
of other epochs, when the body is predicted > 5" away? A real moving body leaves nothing there; a star or galaxy does."""
import json, os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = "$ALE_HOME/rsi/decam-big-tno"
sys.path.insert(0, os.path.join(HERE, "harness"))
import lead_check as LC, arc_extend as AX
from lead_check import radec_to_unit
RN = sys.argv[1]; RUN = os.path.join(HERE, "runs", RN); cache = os.path.join(RUN, "lead_check")
led = {x["id"]: x for x in json.load(open(os.path.join(RUN, "leads_sent.json")))}
def sep(r1, d1, r2, d2):
    return np.degrees(np.arccos(np.clip(radec_to_unit(np.radians(r1), np.radians(d1)) @ radec_to_unit(np.radians(r2), np.radians(d2)), -1, 1))) * 3600
out = {}
for key in sys.argv[2:]:
    f = [g for g in glob.glob(os.path.join(RUN, "arc", f"dbt-{RN}-*.json")) if key in g][0]
    A = json.load(open(f)); i = A["id"]; sid = led[i]["sid"]
    E = LC.exposures(sid, cache); allt = np.array([e["mjd_mid"] for e in E])
    R = AX.remote({sid: {"rows": A["rows"]}})[sid]["rows"]
    p = np.array(A["p"]); cov = np.array(A["cov"]); t0 = A["epoch_tdb"]
    cand = np.array([k for k in range(len(E)) if E[k].get("ra0") is not None])
    pr, pd, s1 = AX.predict(p, cov, t0, allt[cand])
    print("==", i)
    res = []
    for r in A["rows"]:
        ra, dec, ex, fl, fe, sn = R[str(r)]; ex = int(ex)
        ks = []
        for j, k in enumerate(cand):
            m = E[k]
            if sep(m["ra0"], m["dec0"], ra, dec) < 1.1 * 3600 and sep(pr[j], pd[j], ra, dec) > 5.0:
                ks.append(int(k))
        B = AX.remote({sid: {"ball": {str(k): [[float(ra), float(dec), 60.0]] for k in ks}}})[sid]["ball"] if ks else {}
        on = 0; deep = 0; hit = []; m5d = E[ex].get("m5")
        for k in ks:
            hits = B[str(k)][0]
            if not hits:
                continue
            on += 1
            dd = [(float(sep(h[1], h[2], ra, dec)), h[3]) for h in hits]
            near = [x for x in dd if x[0] <= 1.5]
            dp = E[k].get("m5") is not None and m5d is not None and E[k]["m5"] >= m5d - 0.3 and E[k]["band"] == E[ex]["band"]
            deep += bool(dp)
            if near:
                hit.append((round(allt[k], 2), E[k]["band"], round(min(near)[0], 2), round(min(near)[1], 1), bool(dp)))
        print(f"  det {allt[ex]:10.3f} {E[ex]['band']:2s} snr {sn:5.1f}: other-epoch exposures on CCD {on:3d} (same band, as deep: {deep:3d}); source within 1.5\" in {len(hit):3d}  {hit[:8]}")
        res.append(dict(mjd=float(allt[ex]), band=E[ex]["band"], snr=float(sn), on=on, deep=deep, hits=hit))
    out[i] = res
for i, res in out.items():
    os.makedirs(os.path.join(RUN, "static"), exist_ok=True); json.dump(res, open(os.path.join(RUN, "static", i + ".json"), "w"), indent=1)
