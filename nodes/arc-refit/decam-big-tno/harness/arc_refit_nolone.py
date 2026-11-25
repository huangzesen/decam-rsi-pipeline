#!/usr/bin/env python3
"""Refit a lead's extended arc without its untied single-detection seasons (2026-11-22, asked by the DECam search session after
the TRKSUB01 session's single-night audit, planet-search/rsi/candidates/allarch/doublecheck/singlenight.json: many multi-year
arcs rest on one season holding one or two detections that the rest of the arc cannot tie, a chance alignment).

Modelled on harness/arc_refit.py. Starts from runs/<RUN>/arc/<id>.refit.json (fixed-source detections already out) if it
exists, else from runs/<RUN>/arc/<id>.json (harness/arc_extend.py). Reads the rows' places on the scenario host (arc_extend.remote;
DBT_SITE=nersc for the 5 % search regions), splits the arc into seasons (detections > 200 d apart, as arc_extend does) and drops
every detection of the season of each given carrier year. With --card the audit's carrier for that year says which end of the
arc it is (lo = first season, hi = last); the season's first detection must fall in that year, and the audit's detection count is
recorded next to ours. Refits from the base orbit (arc_extend.refit); if that orbit goes unbound, falls back to
lead_check.fit_lead's distance grid (the fitter of the original one-season leads). The grid fit is always run and reported as a
check on short arcs, where distance is poorly constrained.

Writes runs/<RUN>/arc/<id>.nolone.json in the arc format plus "dropped_seasons" [years], "dropped" (row, mjd, reason), "base"
(n, nights, rms, arc_years before), "seasons_after", "fit" (which fitter, or null with "reason"), "grid_fit", "audit". Never
touches .json or .refit.json. Needs >= 4 detections on >= 3 nights left, else "fit": null.
--points also writes the kept detections to harness/agents/nolone_points.json under the card id:
[mjd_utc, ra_deg, dec_deg, sigma_arcsec, "W84"].
Usage: DBT_SITE=nersc OMP_NUM_THREADS=1 ~/anaconda3/bin/python -I harness/arc_refit_nolone.py RUN ID_SUBSTRING YEAR[,YEAR] [--card C###] [--points]
"""
import argparse, datetime, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "harness"))
import lead_check as LC, arc_extend as AX                       # noqa: E402
from lead_check import O                                       # noqa: E402
from astropy.time import Time                                  # noqa: E402

AUDIT = os.path.expanduser("$ALE_HOME/planet-search/rsi/candidates/allarch/doublecheck/singlenight.json")
POINTS = os.path.join(HERE, "harness", "agents", "nolone_points.json")
REASON = "untied season (single-night audit)"


def seasons_of(t):
    """Indices of t grouped into seasons (split where consecutive detections are > 200 d apart), in time order."""
    o = np.argsort(t); g, cur = [], [int(o[0])]
    for a, b in zip(o[:-1], o[1:]):
        if t[b] - t[a] > 200:
            g.append(cur); cur = []
        cur.append(int(b))
    g.append(cur)
    return g


def label(t):
    y = Time(np.asarray(t), format="mjd", scale="utc").decimalyear
    return f"{y.min():.2f}-{y.max():.2f}"


def bound(el):
    return 0 < el["a"] < 1e5 and el["e"] < 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("id"); ap.add_argument("years")
    ap.add_argument("--card"); ap.add_argument("--points", action="store_true")
    ap.add_argument("--as-refit", action="store_true", help="the dropped places are fixed sources: write <id>.refit.json "
                    "(only when none exists), so build.py treats it as the card's refit")
    ap.add_argument("--drop-mjd", default="", help="comma list of detection MJDs to drop (foreign places: TRKSUB01's merges_drop / "
                    "split4_drop lists, 2026-11-22); matched within 2e-4 d; writes <id>.drop.json instead of .nolone.json")
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(",") if y]
    RUN = os.path.join(HERE, "runs", a.run); cache = os.path.join(RUN, "lead_check")
    fs = [f for f in glob.glob(os.path.join(RUN, "arc", f"dbt-{a.run}-*.json"))
          if a.id in os.path.basename(f) and not f.endswith((".refit.json", ".nolone.json", ".drop.json"))]
    if len(fs) != 1:
        sys.exit(f"{a.id}: {len(fs)} arc files match")
    A0 = json.load(open(fs[0])); i = A0["id"]
    rp = os.path.join(RUN, "arc", i + ".refit.json")
    B = json.load(open(rp)) if os.path.exists(rp) else A0
    base_file = os.path.basename(rp if os.path.exists(rp) else fs[0])
    led = {x["id"]: x for x in json.load(open(os.path.join(RUN, "leads_sent.json")))}
    sid = led[i]["sid"]; E = LC.exposures(sid, cache); allt = np.array([e["mjd_mid"] for e in E])
    carriers = {}
    if a.card:
        for c in json.load(open(AUDIT))[a.card]["carriers"]:
            carriers[int(c["year"])] = c
    rows = list(B["rows"])
    D = {int(k): v for k, v in AX.remote({sid: {"rows": rows}})[sid]["rows"].items()}
    M = np.array([D[r] for r in rows]); ex = M[:, 2].astype(int)
    t_all, ra_all, dec_all = allt[ex], M[:, 0], M[:, 1]
    sig_all = 0.08 + 0.6 / np.maximum(M[:, 5], 3.0)
    G = seasons_of(t_all); yfirst = [int(np.floor(Time(t_all[g[0]], format="mjd", scale="utc").decimalyear)) for g in G]
    drop_idx, audit, notes = set(), [], []
    dm = [float(x) for x in a.drop_mjd.split(",") if x]
    for m in dm:
        k = np.flatnonzero(np.abs(t_all - m) < 2e-4)
        if len(k) == 0:
            notes.append(f"drop mjd {m}: no detection of the arc at that time")
        drop_idx |= set(int(v) for v in k)
    for y in years:
        c = carriers.get(y)
        if c is not None:
            gi = 0 if c["end"] == "lo" else len(G) - 1
            if yfirst[gi] != y:
                sys.exit(f"{i}: audit carrier {y} is at the {c['end']} end, but that season starts in {yfirst[gi]} ({label(t_all[G[gi]])})")
        else:
            m = [k for k, yy in enumerate(yfirst) if yy == y]
            if len(m) != 1:
                sys.exit(f"{i}: {len(m)} seasons start in {y}; pass --card")
            gi = m[0]
        g = G[gi]; drop_idx |= set(g)
        nn = len(np.unique(np.floor(t_all[g] - 0.5)))
        rec = dict(year=y, season=label(t_all[g]), ours=len(g), ours_nights=int(nn))
        if c is not None:
            own = sum(1 for v in (c.get("own_drop") or []) if v)     # audit detections already out as fixed sources
            want = c["detections"] - (own if base_file.endswith(".refit.json") else 0)
            rec.update(audit_status=c["status"], audit_end=c["end"], audit_kind=c["kind"], audit_detections=c["detections"],
                       audit_own_drop=own, chance_P=c.get("chance_P"))
            if want != len(g):
                notes.append(f"{y}: audit counts {c['detections']} detection(s) ({own} fixed-source), the arc has {len(g)} in that season")
        audit.append(rec)
    if a.card:
        S = json.load(open(AUDIT))[a.card]
        other = sorted(int(c["year"]) for c in S["carriers"] if c["status"] in ("chance", "untied"))
        if other != sorted(years):
            notes.append(f"audit's untied years {other} differ from the years given {sorted(years)}")
    keep = [k for k in range(len(rows)) if k not in drop_idx]
    why = ("fixed source (TRKSUB01 split-off review)" if a.as_refit else "foreign place (TRKSUB01 merges / split4 drop list)") if dm else REASON
    dropped = [dict(row=int(rows[k]), mjd=round(float(t_all[k]), 5), reason=why) for k in sorted(drop_idx, key=lambda k: t_all[k])]
    t_utc, ra, dec, sig = t_all[keep], ra_all[keep], dec_all[keep], sig_all[keep]
    nights = len(np.unique(np.floor(t_utc - 0.5))) if len(keep) else 0
    res = dict(id=i, made=datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"), base_file=base_file,
               dropped_seasons=years, dropped=dropped,
               base=dict(n=B["n"], nights=B["nights"], rms=B["rms"], arc_years=B["arc_years"], a=B.get("a"), e=B.get("e"), i=B.get("i")),
               audit=audit, notes=notes)
    if keep:
        res["seasons_after"] = [dict(years=label(t_utc[g]), n=len(g), nights=int(len(np.unique(np.floor(t_utc[g] - 0.5)))))
                                for g in seasons_of(t_utc)]
    out = os.path.join(RUN, "arc", i + (".drop.json" if dm else ".nolone.json"))
    if dm and a.as_refit:
        if os.path.exists(rp):
            sys.exit(f"{i}: a .refit.json exists; not overwriting")
        out = rp
    pts = [[round(float(t), 6), round(float(r), 7), round(float(d), 7), round(float(max(s, 0.1)), 3), "W84"]
           for t, r, d, s in sorted(zip(t_utc, ra, dec, sig))]
    if a.points and a.card:
        P = json.load(open(POINTS)) if os.path.exists(POINTS) else {}
        P[a.card] = dict(key=i, run=a.run, n=len(pts), dropped_seasons=years, points=pts)
        json.dump(P, open(POINTS, "w"), indent=1)
    arc_days = float(t_utc.max() - t_utc.min()) if len(keep) else 0.0
    res.update(n=len(keep), nights=int(nights), arc_days=round(arc_days, 1), arc_years=round(arc_days / 365.25, 2),
               rows=[int(rows[k]) for k in keep], seasons=B.get("seasons"))
    if len(keep) < 4 or nights < 3:
        res.update(fit=None, reason=f"only {len(keep)} detections on {nights} nights left after dropping {len(dropped)}")
        json.dump(res, open(out, "w"), indent=1)
        print(i, res["reason"], flush=True); return
    p0 = np.array(B["p"], float); t0 = float(B["epoch_tdb"])
    fit = None
    try:
        p, cov, rms, mx = AX.refit(p0, t0, t_utc, ra, dec, sig)
        el = O.elements(p, t0, helio=True)
        first = dict(a=round(float(el["a"]), 3), e=round(float(el["e"]), 4), i=round(float(el["i"]), 3), rms=round(rms, 3))
        if bound(el):
            fit = "arc_extend.refit from the base orbit"
    except Exception as err:                                    # noqa: BLE001
        first = dict(error=str(err)[:200])
    g = LC.fit_lead(t_utc, ra, dec, sig)
    if g is not None:
        grms, gp, gcov, gt0, gel = g
        res["grid_fit"] = dict(a=round(float(gel["a"]), 3), e=round(float(gel["e"]), 4), i=round(float(gel["i"]), 3),
                               r=round(float(gel["r"]), 2), rms=round(float(grms), 3))
    else:
        res["grid_fit"] = None
    if fit is None:
        res["refit_from_base"] = first
        if g is None:
            res.update(fit=None, reason=f"no bound orbit: refit from the base orbit gives {first}; distance grid finds none")
            json.dump(res, open(out, "w"), indent=1)
            print(i, res["reason"], flush=True); return
        p, cov, t0, el, rms = gp, gcov, gt0, gel, grms
        fit = "lead_check.fit_lead distance grid (refit from the base orbit went unbound)"
    now = [Time(d).mjd for d in AX.DATES]; nr, nd, ns = AX.predict(p, cov, t0, now, "W84")
    res.update(fit=fit, rms=round(float(rms), 3), a=round(float(el["a"]), 3), e=round(float(el["e"]), 4), i=round(float(el["i"]), 3),
               r=round(float(el["r"]), 2), epoch_tdb=float(t0), p=[float(v) for v in p], cov=np.asarray(cov).tolist(),
               now=[dict(date=d, ra=round(float(r_), 5), dec=round(float(d_), 5), sigma_arcsec=round(float(s_), 1))
                    for d, r_, d_, s_ in zip(AX.DATES, nr, nd, ns)])
    json.dump(res, open(out, "w"), indent=1)
    print(i, f"dropped {len(dropped)} ({years}); {len(keep)} on {nights} nights, arc {B['arc_years']} -> {res['arc_years']} yr, "
             f"rms {res['rms']} (was {B['rms']}), a {res['a']} e {res['e']} i {res['i']}, now 1-sigma {res['now'][0]['sigma_arcsec']}\" "
             f"[{'grid' if fit.startswith('lead_check') else 'refit'}]" + (f" NOTE {notes}" if notes else ""), flush=True)


if __name__ == "__main__":
    main()
