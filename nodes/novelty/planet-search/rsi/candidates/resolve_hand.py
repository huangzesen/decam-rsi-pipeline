"""Resolve the ITF tracklets behind 'CHECK BY HAND' novelty verdicts (2026-11-22). novelty_check.py flags ITF tracklets near a lead (step 2, and step 5 along a long arc's carried orbit) but never fits them: every such
lead stayed 'CHECK BY HAND' (399 leads on 2026-11-22). Here, per flagged tracklet: our places alone -> n-body orbit (cand9_orbit fit), then ours +
their ITF lines together. Power check as in novelty_check: the same joint fit with their places moved by 5 fixed sky offsets (2-5').
  SAME     joint rms <= 0.3" and no point over 1", and fewer than 2 shifted copies fit within max(1", 2 x joint rms)
  WEAK     the fit is that good but shifted copies fit too (one or two ITF points cannot decide)
  NOT SAME otherwise (the tracklet is something else)
Catalogued bodies named in the old verdict (borderline, untested, failed fit, or matching one season) get the same test with their MPC
observations (MPC get-obs API), from their own orbit and from ours, the better joint fit kept. User, same day: "there is NO check by check, just
check to the best effort... if unsure, show unsure". Lead verdict, written back into the result file (old one kept as verdict_before_resolve):
  KNOWN ITF <trk> (...)     a SAME tracklet: reported to the MPC by another survey, unlinked (tagged 'known to another survey')
  KNOWN <designation> (...) a SAME catalogued body
  UNSURE: <why>             a WEAK link, or a test that could not run (no observations, fit error)
  NOT IN MPC FILES OF <date> (every flagged tracklet and body tested, none the same body)
Output novelty_queue/hand/<id>.json; summary novelty_queue/hand/SUMMARY.tsv. Usage: python -I resolve_hand.py [--jobs N] [--only SUBSTR]"""
import argparse, glob, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

C = os.path.dirname(os.path.abspath(__file__)); Q = C + "/novelty_queue"; OUT = Q + "/hand"
SHIFTS = ((180, 60), (-240, 120), (120, -200), (-90, -260), (300, -90))


def one(path):
    os.chdir(C); sys.path.insert(0, C)
    import novelty_check as N
    R = json.load(open(path)); rid = R.get("id") or os.path.basename(path)[:-12]
    out = os.path.join(OUT, rid + ".json")
    if os.path.exists(out):
        return json.load(open(out))
    pts = [tuple(p) for p in R["points"]]
    res = dict(id=rid, old=R["verdict"], tracklets=[])
    try:
        s0, ep = N.start_state(pts); f0 = N.NB["fit"](pts, s0, ep, max_nfev=150)
        res["ours_rms"] = round(f0["rms"], 3)
    except Exception as ex:
        res["verdict"] = f"ERROR own fit {ex!r}"[:150]; json.dump(res, open(out, "w")); return res
    items = [dict(trk="ITF " + t["trk"], lines=t["lines"]) for t in R.get("itf", [])]
    named = [x.strip() for x in res["old"].split(":", 1)[1].split(",")] if ":" in res["old"] else []
    for x in named:
        if x.startswith("ITF "):
            continue
        name = x.split(" (")[0].strip()
        if name and name not in [i["trk"] for i in items]:
            try:
                items.append(dict(trk=name, lines=N.get_obs(name)))
            except Exception as ex:
                items.append(dict(trk=name, lines=[], err=repr(ex)[:80]))
    for t in items:
        th = N.obs80(t["lines"]); r = dict(trk=t["trk"], n=len(th))
        if not th:
            r["result"] = "no usable lines"; res["tracklets"].append(r); continue
        try:
            f2 = N.NB["fit"](pts + th, np.array(f0["state"]), f0["epoch"], max_nfev=150); alone = None
            if len(th) >= 3:                     # a catalogued body: also start from its own observations' orbit
                try:
                    s1, e1 = N.start_state(th); f1 = N.NB["fit"](th, s1, e1, max_nfev=150); alone = f1["rms"]
                    g2 = N.NB["fit"](pts + th, np.array(f1["state"]), f1["epoch"], max_nfev=150)
                    if g2["rms"] < f2["rms"]:
                        f2 = g2
                except Exception:
                    pass
            rs = np.hypot(*np.array(f2["resid"]).T); ro = rs[:len(pts)]; ours_rms = float(np.sqrt(np.mean(ro ** 2)))
            r.update(joint_rms=round(f2["rms"], 3), theirs_max=round(float(rs[len(pts):].max()), 3), ours_max=round(float(ro.max()), 3),
                     ours_rms=round(ours_rms, 3), theirs_alone_rms=None if alone is None else round(alone, 3))
            # SAME: one orbit fits both tightly; for a body with many (older, poorer) observations of its own, our places must sit on its orbit
            # (ours rms <= 0.3", none over 1") without spoiling its own fit (joint rms within 20 % + 0.1" of its observations alone)
            own = f0["rms"]                       # how well our places fit an orbit of their own: the joint orbit may not do much worse
            ours_ok = ours_rms <= max(.3, 1.3 * own + .1) and ro.max() <= max(1., 3 * own)
            good = (f2["rms"] <= .3 and rs.max() <= 1) or (alone is not None and ours_ok and f2["rms"] <= 1.2 * alone + .1)
            clear_no = f2["rms"] > 1 and (ro.max() > 1.5 or float(rs[len(pts):].max()) > 3)   # the joint orbit spoils our places or misses theirs
            if good:
                ref = max(1.0, 2 * f2["rms"]); fk = []
                for ox, oy in SHIFTS:
                    sh = [(m, ra + ox / 3600 / math.cos(math.radians(de)), de + oy / 3600, sg, site) for m, ra, de, sg, site in th]
                    try:
                        g = N.NB["fit"](pts + sh, np.array(f0["state"]), f0["epoch"], max_nfev=80); fk.append(round(g["rms"], 2))
                    except Exception:
                        fk.append(None)
                r["fake_rms"] = fk; r["fakes_fitting"] = sum(1 for v in fk if v is not None and v <= ref)
                r["result"] = "SAME" if r["fakes_fitting"] < 2 else "WEAK"
            elif clear_no:
                r["result"] = "NOT SAME"
            else:
                r["result"] = "BORDERLINE"     # neither a clear fit nor a clear miss: shown as UNSURE
        except Exception as ex:
            r["result"] = f"error {ex!r}"[:100]
        res["tracklets"].append(r)
    same = [t for t in res["tracklets"] if t.get("result") == "SAME"]; weak = [t for t in res["tracklets"] if t.get("result") == "WEAK"]
    border = [t for t in res["tracklets"] if t.get("result") == "BORDERLINE"]
    bad = [t for t in res["tracklets"] if t.get("result") not in ("SAME", "WEAK", "NOT SAME", "BORDERLINE")]
    if same:
        t = same[0]; res["verdict"] = f"KNOWN {t['trk']} (joint rms {t['joint_rms']}\", {t['n']} of their obs; resolve_hand)"
    elif weak or bad or border:
        res["verdict"] = "UNSURE: " + "; ".join([f"{t['trk']} fits but {t['fakes_fitting']} of 5 shifted copies fit too" for t in weak]
                                                + [f"{t['trk']} borderline (joint rms {t['joint_rms']}\", ours {t['ours_rms']}\", max {t['ours_max']}\")" for t in border]
                                                + [f"{t['trk']} not testable ({t.get('result')})" for t in bad])
    else:
        res["verdict"] = f"NOT IN MPC FILES OF {R.get('mpc_files', '?')} (resolve_hand: {len(res['tracklets'])} flagged tracklet(s)/body(ies) tested, none the same body)"
    json.dump(res, open(out, "w"), indent=0, default=float)
    return res


def apply(path):
    """Resolve one result file and write the final verdict into it (the old one kept as verdict_before_resolve)."""
    R = json.load(open(path))
    if R.get("hand") or not str(R.get("verdict", "")).startswith("CHECK BY HAND"):   # settled by hand (TRKSUB01: allarch/apply_hand.py): leave it
        return R.get("verdict")
    r = one(path)
    if r.get("verdict", "").startswith("ERROR"):
        r["verdict"] = "UNSURE: " + r["verdict"]
    R["verdict_before_resolve"] = R["verdict"]; R["verdict"] = r["verdict"]; R["resolve"] = r.get("tracklets")
    tmp = path + ".tmp"; json.dump(R, open(tmp, "w"), indent=0, default=float); os.replace(tmp, path)
    return R["verdict"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--only", default="")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    files = []
    for p in sorted(glob.glob(Q + "/done/*.result.json")):
        if a.only and a.only not in p:
            continue
        try:
            R = json.load(open(p))
        except Exception:
            continue
        if str(R.get("verdict", "")).startswith("CHECK BY HAND") and R.get("points") and not R.get("hand"):
            files.append(p)
    print(time.strftime("%H:%M:%S"), len(files), "leads with ITF tracklets to resolve", flush=True)
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for k, r in enumerate(ex.map(one, files, chunksize=1)):
            rows.append(r); print(time.strftime("%H:%M:%S"), k + 1, r["id"], "->", r["verdict"], flush=True)
    with open(OUT + "/SUMMARY.tsv", "w") as f:
        for r in rows:
            f.write(f"{r['id']}\t{r['verdict']}\t{r['old']}\n")
    from collections import Counter
    print(Counter(r["verdict"].split(" ")[0].split(";")[0] for r in rows))


if __name__ == "__main__":
    main()
