#!/usr/bin/env python3
"""Refit a lead's extended arc without the detections that sit on fixed sources (2026-11-17, asked by the TRKSUB01 session for
its all-archive checks: the arc files hold detection row ids, not positions, so it cannot drop them itself).

Reads runs/<RUN>/arc/<id>.json (harness/arc_extend.py) and runs/<RUN>/static/<id>.json (harness/static_test.py, one entry per
arc row, same order). A detection is dropped when, in other-epoch exposures, a source lies within 1.5" of it in an as-deep
same-band exposure, or in >= 2 exposures of any band (stricter than progress/objects/gather.py's verdict rule, so nothing
doubtful stays in the fit). Refits from the arc's orbit (arc_extend.refit), predicts the same DATES and writes
runs/<RUN>/arc/<id>.refit.json in the arc format plus "dropped" (row, mjd, band, reason) and "base" (the arc file's n, rms).
Needs >= 4 detections on >= 3 nights left. Usage: DBT_SITE=nersc ~/anaconda3/bin/python -I harness/arc_refit.py RUN ID_SUBSTRING ..."""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "harness"))
import lead_check as LC, arc_extend as AX                       # noqa: E402
from lead_check import O                                       # noqa: E402
from astropy.time import Time                                  # noqa: E402

run = sys.argv[1]; RUN = os.path.join(HERE, "runs", run); cache = os.path.join(RUN, "lead_check")
led = {x["id"]: x for x in json.load(open(os.path.join(RUN, "leads_sent.json")))}
for key in sys.argv[2:]:
    fs = [f for f in glob.glob(os.path.join(RUN, "arc", f"dbt-{run}-*.json")) if key in f and not f.endswith(".refit.json")]
    if not fs:
        print(key, "no arc file"); continue
    A = json.load(open(fs[0])); i = A["id"]
    sp = os.path.join(RUN, "static", i + ".json")
    if not os.path.exists(sp):
        print(i, "no fixed-source test"); continue
    S = json.load(open(sp))
    if len(S) != len(A["rows"]):
        print(i, f"static has {len(S)} entries for {len(A['rows'])} rows: rerun harness/static_test.py"); continue
    sid = led[i]["sid"]; E = LC.exposures(sid, cache); allt = np.array([e["mjd_mid"] for e in E])
    keep, dropped = [], []
    for r, x in zip(A["rows"], S):
        dh = sum(1 for h in x["hits"] if h[4]); nh = len(x["hits"])
        why = f"source within 1.5\" in {dh} of {x['deep']} as-deep exposures" if dh else (f"source within 1.5\" in {nh} of {x['on']} exposures" if nh >= 2 else None)
        if why:
            dropped.append(dict(row=r, mjd=round(x["mjd"], 4), band=x["band"], reason=why))
        else:
            keep.append(r)
    if not dropped:
        print(i, "nothing to drop"); continue
    D = {int(k): v for k, v in AX.remote({sid: {"rows": keep}})[sid]["rows"].items()}
    M = np.array([D[r] for r in keep]); ex = M[:, 2].astype(int)
    t_utc, ra, dec, sig = allt[ex], M[:, 0], M[:, 1], 0.08 + 0.6 / np.maximum(M[:, 5], 3.0)
    nights = len(np.unique(np.floor(t_utc - 0.5)))
    if len(keep) < 4 or nights < 3:
        print(i, f"only {len(keep)} detections on {nights} nights left after dropping {len(dropped)}: not refitted"); continue
    p0 = np.array(A["p"], float); t0 = float(A["epoch_tdb"])
    p, cov, rms, mx = AX.refit(p0, t0, t_utc, ra, dec, sig)
    el = O.elements(p, t0, helio=True)
    if not (0 < el["a"] and el["e"] < 1):
        print(i, f"no bound orbit after dropping {len(dropped)} of {A['n']} (a {el['a']:.2f}, e {el['e']:.2f}): not written"); continue
    now = [Time(d).mjd for d in AX.DATES]; nr, nd, ns = AX.predict(p, cov, t0, now, "W84")
    res = dict(id=i, n=len(keep), nights=int(nights), arc_days=round(float(t_utc.max() - t_utc.min()), 1),
               arc_years=round(float(t_utc.max() - t_utc.min()) / 365.25, 2), rms=round(float(rms), 3),
               a=round(float(el["a"]), 3), e=round(float(el["e"]), 4), i=round(float(el["i"]), 3), r=round(float(el["r"]), 2),
               rows=keep, seasons=A.get("seasons"), epoch_tdb=t0, p=[float(v) for v in p], cov=np.asarray(cov).tolist(),
               now=[dict(date=d, ra=round(float(r_), 5), dec=round(float(d_), 5), sigma_arcsec=round(float(s_), 1)) for d, r_, d_, s_ in zip(AX.DATES, nr, nd, ns)],
               dropped=dropped, base=dict(n=A["n"], nights=A["nights"], rms=A["rms"], arc_years=A["arc_years"]))
    json.dump(res, open(os.path.join(RUN, "arc", i + ".refit.json"), "w"), indent=1)
    print(i, f"dropped {len(dropped)} of {A['n']}; {len(keep)} on {nights} nights, arc {res['arc_years']} yr, rms {res['rms']} (was {A['rms']}), "
             f"a {res['a']} e {res['e']} i {res['i']}, now 1-sigma {res['now'][0]['sigma_arcsec']}\" (was {A['now'][0]['sigma_arcsec']}\")", flush=True)
