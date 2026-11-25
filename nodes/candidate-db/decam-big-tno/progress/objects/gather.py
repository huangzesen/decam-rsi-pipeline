#!/usr/bin/env python3
"""Every promising new body from every experiment, in one list (2026-11-17).

Sources
- planet-search (the planet RSI runs, fields F1-F10): planet-search/rsi/candidates/candidates_status.json (hand-kept).
- decam-big-tno runs r1, r2 (hill climbs), s38 (first 5 % search; n10, v28, r2b and its bands), s10a, s10b (second 5 %), sldt5 (fourth 5 %, LDT-weighted; 2026-11-19):
  runs/<run>/lead_check/<tag>.json (verdicts) + .log (fits), novelty_queue/done/<id>.result.json, runs/<run>/arc/<id>.json
  (extended arcs), runs/<run>/static/<id>.json (fixed-source test), runs/<run>/confirm/<id>.json (other archives),
  runs/r2/observer_labels.json (DBT006-) and the DBT001-005 map of runs/r1/LEADS.md.
Promising = filed or sent to OBSERVER_NAME, or: seen again in exposures its program did not use AND not in the MPC's files (or
its novelty check still pending) AND not failed on fixed sources. Re-finds of one body (shared detections, or the same
orbit and size) are merged. No sky positions or dates are written here except "now" places, which stay out of the page.
Usage: python3 -I progress/objects/gather.py  -> progress/objects/objects.json"""
import glob, json, os, re
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PS = os.path.join(HERE, "..", "..", "planet-search", "rsi", "candidates")
NOV = os.path.join(PS, "novelty_queue", "done")
R1MAP = {"LEAD01": "DBT001", "LEAD02": "DBT002", "LEAD03": "DBT003",
         "LEAD04": "DBT004", "LEAD05": "DBT005"}
EXP = {"r1": "decam-big-tno hill climb, first run", "r2": "decam-big-tno hill climb (Run 1 on the tracker)",
       "s38": "first 5 % search", "s10a": "second 5 % search, step A", "s10b": "second 5 % search, step B",
       "sldt5": "fourth 5 % search (LDT-weighted, conveyor)", "sa15": "third 5 % search (archive15, conveyor)", "su05": "fifth 5 % search (conveyor unit u05)", "su06": "sixth 5 % search (conveyor unit u06)"}


def J(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


labels = dict(R1MAP); labels.update(J(os.path.join(HERE, "runs/r2/observer_labels.json"), {}))
sent = {}
for b in J(os.path.join(HERE, "runs/r2/observer_sent.json"), []):
    for i in b.get("ids", []):
        sent.setdefault(i, str(b.get("batch")))
for i in R1MAP:
    sent.setdefault(i, "1")


def static_verdict(S):
    """a detection is on a fixed source when a source sits within 1.5" of it in other epochs: in at least half of the
    as-deep same-band exposures (>= 2), or in >= 2 exposures of any band making >= 1 % of those looked at (calibrated on
    the hand verdicts of 2026-11-15/16: LEAD06, LEAD07, LEAD08 fail, LEAD09 passes). A lead fails
    when fewer than 5 clean detections remain or a third of them are on fixed sources."""
    if S is None:
        return None, "not tested"
    def bad(x):
        dh = sum(1 for h in x["hits"] if h[4]); nh = len(x["hits"])
        return ((x["deep"] >= 2 and dh >= max(2, x["deep"] / 2)) or (nh >= 2 and x["on"] and nh / x["on"] >= 0.01)
                or (x["on"] >= 1 and nh >= x["on"]))     # a source in every other-epoch exposure looked at, even if only one
                                                         # (2026-11-20: C###, a G 19.7 Gaia star, had on=1, hits=1 at 9 of 10 detections)
    b = sum(1 for x in S if bad(x)); n = len(S)
    if b == 0:
        return "clean", f"clean at all {n} detections"
    if n - b < 5 or b * 3 >= n:
        return "failed", f"on fixed sources at {b} of {n} detections"
    return "partly", f"{b} of {n} detections on fixed sources (refit without them)"


HAND = {"r2b-gen014", "n10-gen019", "v28-gen019"}   # first 5 % lists read exposure by exposure on 2026-11-14/16 (runs/s38/LEADS.md)
BAND_A_PASS = {"LEAD09"}                     # band A: only this one passed the hand check
OUT = {"LEAD10"}   # 2026-11-17: no bound orbit once its fixed-source detections are dropped (arc_refit.py)
OUT |= set((J(os.path.join(os.path.dirname(os.path.abspath(__file__)), "not_a_body.json"), {}) or {}).get("leads", {}))   # hand: not bodies (2026-11-20)


# 2026-11-17 (TRKSUB01 session): the novelty check looked for ITF tracklets only near the lead's own nights; its whole-path
# check (allarch/itf_path_check.py -> allarch/itf_path.json) finds unlinked tracklets of other observers on some paths.
ITF = (J(os.path.join(PS, "allarch", "itf_path.json"), {}) or {}).get("bodies", {})


JOINT = J(os.path.join(PS, "allarch", "itf_joint.json"), {}) or {}   # TRKSUB01's identity fits (one orbit through tracklet + arc, shifted copies must fail)
JOINT.update(J(os.path.join(PS, "allarch", "itf_joint_p4_wide.json"), {}) or {})   # the step B bodies (2026-11-17)


def itf_text(i):
    T = (ITF.get(i) or {}).get("tracklets") or []
    if i not in ITF and i not in JOINT:
        return None, "not checked"
    if i in JOINT:
        same = [t for t in JOINT[i] if t.get("verdict", "").startswith("SAME")]
        cons = [t for t in JOINT[i] if t.get("verdict", "").startswith("CONSISTENT")]
        txt = lambda t: f'{t["station"]} {t["trksub"]} ({t["n"]} obs; one orbit fits both at rms {t["rms"]:.2f}")'
        if same:
            return "same body", "proven by a joint orbit: " + "; ".join(txt(t) for t in same)
        if cons:
            return "possible", "consistent but untestable (shifted copies fit too): " + "; ".join(txt(t) for t in cons)
        return None, f"{len(JOINT[i])} tracklet(s) near the path, none the same body by a joint orbit"
    out, kind = [], None
    for t in sorted(T, key=lambda t: t["max_sep"]):
        s = t["med_sigma"]; sep = t["max_sep"]
        k = "same body" if sep <= max(2.0, 3 * s) and s <= 10 else ("possible" if sep <= 3 * s and s <= 60 else None)   # a looser match with 1-sigma > 60" says nothing
        if k:
            out.append(f'{t["station"]} {t["trksub"]} ({t["n"]} obs, {sep:.1f}" off, 1-sigma {s:.1f}"): {k}')
            kind = kind or k
    return kind, "; ".join(out) if out else "none on its path"


# the LDT 5 % search's large leads (H <= 6.6, not known, not yet seen again in other exposures) are candidates too
# (2026-11-19)
LARGE = set()                                    # every search's follow-up list (sldt5 first; sa15 and later since 11-19 19:40)
for _fl in glob.glob(os.path.join(HERE, "runs", "*", "followup_ids.txt")):
    LARGE |= set(open(_fl).read().split())
rows = []
# every run with lead checks, so new searches (conveyor units) join without an edit here (2026-11-19)
RUNS = ["r1", "r2", "s38", "s10a", "s10b", "sldt5"]
RUNS += sorted(os.path.basename(os.path.dirname(d)) for d in glob.glob(os.path.join(HERE, "runs", "*", "lead_check"))
               if os.path.basename(os.path.dirname(d)) not in RUNS
               and not re.match(r"^s38p\d+$", os.path.basename(os.path.dirname(d))))   # s38p*: shards of s38's lead check, merged there
for run in RUNS:
    R = os.path.join(HERE, "runs", run)
    EXP.setdefault(run, f"search {run}")
    led = {x["id"]: x for x in J(os.path.join(R, "leads_sent.json"), [])}
    for jf in sorted(glob.glob(os.path.join(R, "lead_check", "*.json"))):
        tag = os.path.basename(jf)[:-5]
        if tag.endswith("_plan") or tag.endswith("_exposures"):
            continue
        L = J(jf, [])
        if not isinstance(L, list):
            continue
        fit = {}
        lp = jf[:-5] + ".log"
        if os.path.exists(lp):
            for line in open(lp, errors="replace"):
                k = line.find('{"id"')
                if k >= 0:
                    try:
                        d = json.loads(line[k:]); fit[d["id"]] = d
                    except Exception:
                        pass
        for x in L:
            i = x["id"]; v = x.get("verdict", "")
            if i in OUT:
                continue
            lab = labels.get(i)
            # lead check v6b (2026-11-22): 2+ kept hits on one night only now read UNCLEAR; such leads stay in the list at
            # their own stage instead of dropping out (C###, C###, C### had)
            one = v.startswith("UNCLEAR") and x.get("hit_nights") == 1 and (x.get("extra_found") or 0) >= 2
            if not v.startswith("SEEN AGAIN") and not one and not lab and i not in LARGE:
                continue
            nv = J(os.path.join(NOV, i + ".result.json"))
            nov = "pending" if nv is None else str(nv.get("verdict", "?"))
            # 2026-11-21: a match to unlinked ITF
            # tracklets (e.g. Rubin RL00jqX) is known to another survey but has no MPC designation: kept, tagged
            # then (same day): "actually target on MPC should stay, marked as MPC": known bodies are never dropped here,
            # they are tagged below (MPC body / probably an MPC body / known to another survey, not in the MPC)                      # 2026-11-19: a known body leaves the list even if it was sent (DBT011 = 2010 FB49); its card stays, dropped
            A = J(os.path.join(R, "arc", i + ".json"), {}) or {}
            sv, stext = static_verdict(J(os.path.join(R, "static", i + ".json")))
            if sv == "failed" and not lab:
                continue
            C = J(os.path.join(R, "confirm", i + ".json"))
            f = fit.get(i) or {k: x.get(k) for k in ("H", "r", "a", "e", "i", "nights", "arc_days", "rms")}
            src = A if A else f
            m = re.search(r"-(R\d{3}[a-g]?|[A-D])-rank(\d+)-p([\d.]+)$", i)
            rows.append(dict(
                id=i, run=run, experiment=EXP[run], tag=tag, region=m.group(1) if m else "?", rank=int(m.group(2)) if m else None,
                p=float(m.group(3)) if m else None, label=lab, sent=sent.get(i),
                H=f.get("H"), r=src.get("r", f.get("r")), a=src.get("a"), e=src.get("e"), i=src.get("i"),
                nights=src.get("nights"), arc_years=round(A["arc_years"], 2) if A.get("arc_years") is not None else (round(f["arc_days"] / 365.25, 2) if f.get("arc_days") is not None else None),
                rms=src.get("rms"), arc_extended=bool(A), now_sigma=(A.get("now") or [{}])[0].get("sigma_arcsec") if A else None,
                now_place=[(A.get("now") or [{}])[0].get(k) for k in ("date", "ra", "dec", "sigma_arcsec")] if A and A.get("now") else None,
                seen_again=v.startswith("SEEN AGAIN"), one_night=one, lead_check=v[:120], novelty=nov[:80],
                fixed_source=sv, fixed_source_text=stext, archives=(C or {}).get("verdict"), archives_checked=(C or {}).get("checked"),
                itf=itf_text(i)[1], itf_kind=itf_text(i)[0],
                hand=("held" if tag in HAND else "failed" if tag.startswith("r2blowA") and not any(k in i for k in BAND_A_PASS) else None) if not lab else None,
                det=[led[i]["sid"], sorted(led[i]["det"])] if i in led and "det" in led[i] else None))

# merge re-finds: shared detections in one scenario, or the same orbit and size
SAME = {frozenset(x[:2]) for x in (J(os.path.join(os.path.dirname(os.path.abspath(__file__)), "same_body.json"), {}) or {}).get("pairs", [])}
# hand list of leads proven to be different bodies (TRKSUB01, 2026-11-22: clause 3 below joined two cold classicals / two
# Trojans on like orbits): {"pairs": [[id_a, id_b, note], ...]}, written only by the TRKSUB01 session; a lead never joins a
# group holding any lead it is listed against, and a listed pair is never "same" whatever the other clauses say
NOT_SAME = {frozenset(x[:2]) for x in (J(os.path.join(os.path.dirname(os.path.abspath(__file__)), "not_same_body.json"), {}) or {}).get("pairs", [])}


def same(a, b):
    if frozenset((a["id"], b["id"])) in NOT_SAME:
        return False
    if frozenset((a["id"], b["id"])) in SAME:          # proven by a joint orbit fit (same_body.json, 2026-11-20)
        return True
    if a["det"] and b["det"] and a["det"][0] == b["det"][0] and set(a["det"][1]) & set(b["det"][1]):
        return True
    try:
        like = abs(a["a"] - b["a"]) < 0.5 and abs(a["e"] - b["e"]) < 0.02 and abs(a["i"] - b["i"]) < 0.3 and abs(a["H"] - b["H"]) < 0.5
    except TypeError:
        return False
    # 2026-11-22 (TRKSUB01 merges pass): a like orbit and size is not one body. Of 58 leads merged by this clause alone and 6 by chains,
    # the merges pass found leads 20-140 degrees apart in the sky (C### rank73, C### rank22, C###, C###: two cold classicals; C###, C###:
    # two Trojans). So the clause also needs both leads' arcs to put the body in one place now: their "now" places (arc.json, same date
    # within 1 day, combined 1-sigma <= 600") within 60" + 5 x the combined 1-sigma. Without that it cannot merge: shared detections or a
    # joint orbit decide (same_body.json). On the 90 key-lead pairs the pass tested whose link was this clause, the gate keeps 44 of the
    # 75 true merges and none of the 14 wrong ones (the other 31 true ones have no informative "now" place; many also share detections).
    # A missed merge shows up as a duplicate card (crosscard.py); a wrong merge hides a candidate.
    return like and near_now(a, b)


def near_now(a, b):
    import math
    pa, pb = a.get("now_place"), b.get("now_place")
    if not pa or not pb or None in pa or None in pb:
        return False
    s = math.hypot(pa[3] or 0, pb[3] or 0)
    if abs(J_date(pa[0]) - J_date(pb[0])) > 1 or s > 600:      # different dates, or a "now" place that says nothing
        return False
    c = math.cos(math.radians((pa[2] + pb[2]) / 2))
    sep = math.hypot(((pa[1] - pb[1] + 180) % 360 - 180) * c, pa[2] - pb[2]) * 3600
    return sep <= 60 + 5 * s


def J_date(d):
    import calendar, time
    return calendar.timegm(time.strptime(str(d)[:10], "%Y-%m-%d")) / 86400


def rank_key(x):   # the representative: labelled first, then extended arc, then most nights
    return (x["label"] is None, not x["arc_extended"], -(x["nights"] or 0), -(x["p"] or 0))


groups = []
for x in sorted(rows, key=rank_key):
    for g in groups:
        if any(same(x, y) for y in g) and not any(frozenset((x["id"], y["id"])) in NOT_SAME for y in g):
            g.append(x); break
    else:
        groups.append([x])
bodies = []
for g in groups:
    b = dict(g[0]); b["refinds"] = [y["id"] for y in g[1:]]
    b["found_in"] = sorted({y["experiment"] for y in g})
    b["label"] = next((y["label"] for y in g if y["label"]), None)
    b["sent"] = next((y["sent"] for y in g if y["sent"]), None)
    b["archives"] = next((y["archives"] for y in g if y["archives"]), None)
    b["itf_kind"] = next((y["itf_kind"] for y in g if y["itf_kind"]), None); b["itf"] = next((y["itf"] for y in g if y["itf_kind"]), b["itf"])
    del b["det"]
    b["rubin"] = any((J(os.path.join(HERE, "runs", y["run"], "confirm", y["id"] + ".json"), {}) or {}).get("rubin_rights") for y in g)
    V = (b["archives"] or "").upper()
    nvs = [str((J(os.path.join(NOV, y["id"] + ".result.json"), {}) or {}).get("verdict", "")) for y in g]
    kn = (next((v for v in nvs if v.startswith("KNOWN") and not v.startswith("KNOWN ITF")), None)   # an MPC designation beats an ITF match
          or next((v for v in nvs if v.startswith("KNOWN")), None) or next((v for v in nvs if v.startswith("PROBABLY")), None))
    b["known_as"] = kn
    if kn and (kn.startswith("KNOWN ITF") or (kn.startswith("KNOWN:") and "ITF" in kn)):          # not '"ITF" in kn': a designated body's text can mention its old tracklet (C###, 2026-11-22)
        b["stage"] = "known to another survey, not in the MPC"
    elif kn and kn.startswith("KNOWN"):
        b["stage"] = "known MPC body"
    elif kn and kn.startswith("PROBABLY KNOWN ITF") and not any(y.get("label") for y in g):   # a likely match to an unlinked ITF tracklet, not a catalogued body (C###, C###; 2026-11-23)
        b["stage"] = "probably known to another survey, not in the MPC"
    elif kn and not any(y.get("label") for y in g):
        b["stage"] = "probably a known MPC body"
    elif V.startswith("REPORTED BY"):
        b["stage"] = "already reported by another survey"
    elif V.startswith("OTHER OBSERVERS"):
        b["stage"] = "seen; others' unlinked MPC tracklets"
    elif V.startswith("NOT SEEN"):
        b["stage"] = "not seen where expected"
    elif V.startswith("PROBABLY NOT SEEN"):            # 2026-11-22 (TRKSUB01): a missed search with some power left (e.g. C###)
        b["stage"] = "probably not seen where expected"
    elif b["archives"] and b["archives"].upper().startswith("SEEN ONCE"):
        b["stage"] = "seen once in another archive"
    elif b["archives"] and b["archives"].upper().startswith("SEEN"):
        b["stage"] = "seen in another archive"
    elif b["sent"]:
        b["stage"] = "sent to OBSERVER_NAME"
    elif b["hand"] == "failed":
        continue
    elif b["hand"] == "held":
        b["stage"] = "held back after the hand check"
    elif not b["seen_again"] and b.get("one_night"):
        b["stage"] = "seen again on one night only"
    elif not b["seen_again"] and b["id"] in LARGE:
        b["stage"] = "large, not yet tested in other images"
    elif b["fixed_source"] == "clean" and (b["arc_years"] or 0) >= 1:
        b["stage"] = "strong: seen again, clean, multi-year arc"
    elif b["fixed_source"] == "clean":
        b["stage"] = "seen again, clean, short arc"
    elif b["fixed_source"] == "partly":
        b["stage"] = "seen again, needs a refit"
    else:
        b["stage"] = "seen again, fixed-source test pending"
    bodies.append(b)

ps = J(os.path.join(PS, "candidates_status.json"), {})
planet = []
for b in ps.get("bodies", []):
    if b.get("stage") == "known":
        continue
    planet.append(dict(key=b["key"], title=b.get("title"), field=b.get("field"), stage=b.get("stage"), status=b.get("status"),
                       facts=b.get("facts"), summary=b.get("summary")))

ORDER = ["seen in another archive", "seen; others' unlinked MPC tracklets", "seen once in another archive", "sent to OBSERVER_NAME", "strong: seen again, clean, multi-year arc", "seen again, clean, short arc",
         "seen again, needs a refit", "seen again, fixed-source test pending", "large, not yet tested in other images", "seen again on one night only", "held back after the hand check", "not seen where expected", "probably not seen where expected", "already reported by another survey", "known to another survey, not in the MPC", "probably known to another survey, not in the MPC", "known MPC body", "probably a known MPC body"]
bodies.sort(key=lambda b: (ORDER.index(b["stage"]), b["label"] or "ZZ", b["H"] if b["H"] is not None else 99))
out = dict(asof=__import__("time").strftime("%Y-%m-%d %H:%M %Z"), planet_search=planet, decam_big_tno=bodies,
           counts={s: sum(1 for b in bodies if b["stage"] == s) for s in ORDER})
os.makedirs(os.path.join(HERE, "progress", "objects"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "progress", "objects", "objects.json"), "w"), indent=1)
print("planet-search bodies:", len(planet), [p["key"] for p in planet])
print("decam-big-tno bodies:", len(bodies), "(from", len(rows), "lead rows)")
for s in ORDER:
    print(f"  {s}: {out['counts'][s]}")
