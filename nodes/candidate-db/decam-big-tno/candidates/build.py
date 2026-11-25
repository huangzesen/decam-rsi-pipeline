#!/usr/bin/env python3
"""The candidate database: one folder per promising body, with every identity and every piece of evidence
(2026-11-19).

Input: progress/objects/objects.json (progress/objects/gather.py: every promising body of every search, re-finds merged;
new searches with runs/<run>/lead_check/ join by themselves) and planet-search/rsi/candidates/candidates_status.json.
Output, all in candidates/:
  registry.json   permanent catalogue numbers: C###, C###, ... Every lead id, label (DBT..., TRKSUB...) and planet-search
                  alias points at one number. A number is never reused or renumbered; bodies found later get the next one.
  C###/card.json  identities (catalogue number, labels, every lead id that is this body, search, region, rank, scenario
                  and detection ids), orbit and size, stage, the evidence ladder (found, seen again, not in the MPC's
                  files, fixed sources, other archives, reported), text of each verdict, history of stage changes,
                  and the evidence files.
  C###/evidence/  copies of the evidence: lead-check verdict and fit, novelty result, extended arc and refit, fixed-source
                  test, other-archive check, MPC tracklet matches, OBSERVER_NAME batch; planet-search status entry.
  C###/README.md  the card in words.
  index.json      the whole database in one file (every card, active and dropped).
A body that stops passing gather.py's test (a known body after all, failed on fixed sources, failed the hand check) keeps
its folder and number; its card says dropped, and why when that can be told.
Private: cards and evidence carry sky positions and dates of unreported bodies. Pages built from it strip them.
Usage: python3 -I candidates/build.py   (or candidates/refresh.sh: gather, this, observe, page)"""
import collections, glob, json, os, re, shutil, time

C = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(C)
PS = os.path.normpath(os.path.join(HERE, "..", "..", "planet-search", "rsi", "candidates"))
NOBOUND = {}   # filled in main() from allarch/nobound_check.json
MPCORB_KNOWN = {}   # filled in main() from allarch/doublecheck/knownlook_mpc_orbits.json
IMAGETESTS = {}     # filled in main() from allarch/doublecheck/imagetests.json (TRKSUB01's image tests in public frames)
NOV = os.path.join(PS, "novelty_queue", "done")
NOW = time.strftime("%Y-%m-%d %H:%M %Z")


def J(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


def W(p, x):
    tmp = p + ".tmp"
    json.dump(x, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, p)


def size(H, alb):
    return None if H is None else round(1329 / alb ** 0.5 * 10 ** (-H / 5))


def lead_search(run):
    return {"r1": "hill climb run 1", "r2": "hill climb run 2", "s38": "first 5 % search", "s10a": "second 5 % search",
            "s10b": "second 5 % search", "sldt5": "LDT-weighted 5 % search", "sa15": "third 5 % search", "su05": "fifth 5 % search", "su06": "sixth 5 % search"}.get(run, f"search {run}")


def ladder_dbt(b):
    """six steps, each pass / partial / pending / fail / todo, with a short text"""
    nv = b.get("novelty") or "pending"
    arch = (b.get("archives") or "").upper()
    st = []
    st.append(("found", "pass", f"{b['experiment']}, region {b['region']}, rank {b['rank']}, p {b['p']}"
               + (f"; found again in {len(b['refinds'])} other list(s)" if b.get("refinds") else "")))
    if b["seen_again"]:
        st.append(("seen_again", "pass", "detected where predicted in exposures its program did not use"))
    else:
        st.append(("seen_again", "todo", "not yet looked for in exposures its program did not use"))
    if nv.startswith("NOT IN MPC"):
        st.append(("not_in_mpc", "pass", nv))
    elif nv.startswith(("CHECK BY HAND", "UNSURE")):      # 2026-11-22: UNSURE = the automatic identity test could not decide
        st.append(("not_in_mpc", "partial", nv))
    elif nv == "pending":
        st.append(("not_in_mpc", "todo", "novelty check pending"))
    else:
        st.append(("not_in_mpc", "fail", nv))
    fs = b.get("fixed_source")
    st.append(("fixed_sources", {"clean": "pass", "partly": "partial", "failed": "fail"}.get(fs, "todo"), b.get("fixed_source_text") or "not tested"))
    if arch.startswith("SEEN ONCE"):
        st.append(("other_archives", "partial", b["archives"]))
    elif arch.startswith(("SEEN", "OTHER OBSERVERS")):
        st.append(("other_archives", "pass", b["archives"]))
    elif arch.startswith("NO DECISIVE"):
        st.append(("other_archives", "todo", "checked: " + b["archives"]))
    elif arch.startswith(("NOT SEEN", "REPORTED", "PROBABLY NOT SEEN")):
        st.append(("other_archives", "fail", b["archives"]))
    elif b.get("archives"):
        st.append(("other_archives", "partial", b["archives"]))
    else:
        st.append(("other_archives", "todo", "not yet checked in other telescopes' archives"))
    if b.get("sent"):
        st.append(("reported", "pass", f"sent to OBSERVER_NAME (batch {b['sent']}) for follow-up"))
    elif arch.startswith("REPORTED"):
        st.append(("reported", "fail", "already reported by another survey"))
    else:
        st.append(("reported", "todo", "not yet sent or filed"))
    return [dict(step=a, state=s, text=t) for a, s, t in st]


def ladder_planet(p):
    tests = {t[0]: (t[1], t[2]) for t in (p.get("tests") or [])}
    def get(*names):
        for n in names:
            if n in tests:
                return ("pass" if tests[n][0] else "fail", tests[n][1])
        return ("todo", "")
    st = [("found", "pass", p.get("found_by") or "planet-search runs"),
          ("seen_again", *get("Seen in images its program never used")),
          ("not_in_mpc", *get("Not in the catalogues")),
          ("fixed_sources", "na", "not part of the planet-search checks"),
          ("other_archives", *get("Seen in images its program never used", "Followed for years")),
          ("reported", "pass" if p.get("stage") == "filed" else "todo",
           "positions filed with the Minor Planet Center" if p.get("stage") == "filed" else "not yet filed")]
    return [dict(step=a, state=s, text=t) for a, s, t in st]


def orbit_from_facts(facts):
    o = {}
    for k, v in facts or []:
        if k == "Orbit":
            for key, pat in (("a", r"a ([\d.]+) au"), ("e", r"e ([\d.]+)"), ("i", r"i ([\d.]+)")):
                m = re.search(pat, v)
                if m:
                    o[key] = float(m.group(1))
    return o


def evidence_dbt(b, ev):
    """copy each evidence file of every lead id of this body into ev/; return the list"""
    out = []
    for f in glob.glob(os.path.join(ev, "refind_*")):      # leads split off later must not leave their files (2026-11-22)
        os.remove(f)
    ids = [b["id"]] + list(b.get("refinds") or [])
    for i in ids:
        m = re.match(r"dbt-([a-z0-9]+)-", i)
        run = m.group(1) if m else b["run"]
        R = os.path.join(HERE, "runs", run)
        pre = "" if i == b["id"] else f"refind_{len(out)}_"
        for sub, name in (("arc", "arc.json"), ("arc", "arc.refit.json"), ("static", "fixed_sources.json"),
                          ("confirm", "other_archives.json"), ("arc", "arc_prior.json")):
            src = os.path.join(R, sub, i + {"arc.json": ".json", "arc.refit.json": ".refit.json", "arc_prior.json": ".prior.json"}.get(name, ".json"))
            if os.path.exists(src):
                dst = os.path.join(ev, pre + name)
                shutil.copy2(src, dst); out.append(os.path.basename(dst))
        nv = os.path.join(NOV, i + ".result.json")
        if os.path.exists(nv):
            shutil.copy2(nv, os.path.join(ev, pre + "novelty.json")); out.append(pre + "novelty.json")
        lc = []
        for jf in glob.glob(os.path.join(R, "lead_check", "*.json")):
            L = J(jf, [])
            if isinstance(L, list):
                lc += [dict(x, _file=os.path.relpath(jf, HERE)) for x in L if isinstance(x, dict) and x.get("id") == i]
            lp = jf[:-5] + ".log"
            if os.path.exists(lp):
                for line in open(lp, errors="replace"):
                    k = line.find('{"id"')
                    if k >= 0 and i in line:
                        try:
                            d = json.loads(line[k:])
                            if d.get("id") == i:
                                lc.append(dict(d, _file=os.path.relpath(lp, HERE), _kind="fit"))
                        except Exception:
                            pass
        if lc:
            W(os.path.join(ev, pre + "lead_check.json"), lc); out.append(pre + "lead_check.json")
        for sl in glob.glob(os.path.join(R, "leads_sent.json")):
            for x in J(sl, []):
                if x.get("id") == i:
                    W(os.path.join(ev, pre + "lead.json"), x); out.append(pre + "lead.json")
                    break
    itf = {}
    for f in ("itf_path.json", "itf_joint.json", "itf_joint_p4_wide.json"):
        D = J(os.path.join(PS, "allarch", f), {}) or {}
        D = D.get("bodies", D) if f == "itf_path.json" else D
        for i in ids:
            if i in D:
                itf[f + ":" + i] = D[i]
    if itf:
        W(os.path.join(ev, "mpc_tracklets.json"), itf); out.append("mpc_tracklets.json")
    for q in J(os.path.join(HERE, "runs/r2/observer_sent.json"), []) or []:
        if any(i in q.get("ids", []) for i in ids):
            W(os.path.join(ev, "observer.json"), q); out.append("observer.json")
            break
    return sorted(set(out))


def static_failed(S):
    """gather.py's fixed-source rule (static_verdict), repeated here for the dropped reason"""
    if not S:
        return None
    def bad(x):
        dh = sum(1 for h in x["hits"] if h[4]); nh = len(x["hits"])
        return ((x["deep"] >= 2 and dh >= max(2, x["deep"] / 2)) or (nh >= 2 and x["on"] and nh / x["on"] >= 0.01)
                or (x["on"] >= 1 and nh >= x["on"]))     # a source in every other-epoch exposure looked at, even if only one
                                                         # (2026-11-20: C###, a G 19.7 Gaia star, had on=1, hits=1 at 9 of 10 detections)
    nb = sum(1 for x in S if bad(x)); n = len(S)
    return f"failed the fixed-source test: {nb} of {n} detections sit on sources seen at other times" if (n - nb < 5 or nb * 3 >= n) else None


def dropped_reason(i):
    nb = (J(os.path.join(HERE, "progress", "objects", "not_a_body.json"), {}) or {}).get("leads", {})
    if i in nb:
        return "not a body: " + nb[i]
    nv = (J(os.path.join(NOV, i + ".result.json"), {}) or {}).get("verdict", "")
    if str(nv).startswith(("KNOWN", "PROBABLY")):
        return "a known body: " + str(nv)[:120]
    m = re.match(r"dbt-([a-z0-9]+)-", i)
    f = static_failed(J(os.path.join(HERE, "runs", m.group(1) if m else "", "static", i + ".json")))
    if f:
        return f
    return "no longer passes the promising test of progress/objects/gather.py (known, failed fixed-source test or hand check)"


def readme(card):
    L = [f"# {card['cid']} {card['name']}", "", f"Stage: **{card['stage']}**" + ("" if card["active"] else f" (dropped: {card.get('dropped_reason', '')})"), ""]
    L.append("## Identities")
    for k, v in card["identities"].items():
        L.append(f"- {k}: {v if not isinstance(v, list) else ', '.join(map(str, v)) or '-'}")
    o = card["orbit"]
    L += ["", "## Orbit and size",
          f"- a {o.get('a')} au, e {o.get('e')}, i {o.get('i')} deg; distance now {o.get('r')} au",
          f"- H {card.get('H')}; diameter about {card['D_km'][0]} km (albedo 0.10) to {card['D_km'][1]} km (0.05)" if card.get("D_km") else "- H not measured",
          f"- arc {card.get('arc_years')} years over {card.get('nights')} nights", "", "## Evidence"]
    for s in card["ladder"]:
        L.append(f"- {s['step']}: {s['state']} - {s['text']}")
    L += ["", "## History"] + [f"- {h['time']}: {h['stage']}" for h in card["history"]]
    L += ["", "## Files", ""] + [f"- evidence/{f}" for f in card["evidence"]]
    return "\n".join(L) + "\n"


def main():
    NOBOUND.update(J(os.path.join(PS, "allarch", "nobound_check.json"), {}) or {})
    it_ = J(os.path.join(PS, "allarch", "doublecheck", "imagetests.json"), {}) or {}
    prov_ = {p_["pass"] for p_ in it_.get("passes") or [] if not str(p_.get("status", "")).startswith("read") or "still running" in str(p_.get("status", ""))}
    for cid_, rows_ in (it_.get("cards") or {}).items():
        IMAGETESTS[cid_] = [dict(r_, provisional=r_.get("pass") in prov_) for r_ in rows_]
    for f_ in sorted(glob.glob(os.path.join(PS, "allarch", "doublecheck", "*_mpc_orbits.json"))):   # TRKSUB01's MPCORB extracts
        MPCORB_KNOWN.update((J(f_, {}) or {}).get("cards") or {})
    obj = J(os.path.join(HERE, "progress", "objects", "objects.json"))
    reg = J(os.path.join(C, "registry.json"), {"next": 1, "ids": {}, "about": "permanent catalogue numbers; never renumber"})
    ps_all = {b["key"]: b for b in (J(os.path.join(PS, "candidates_status.json"), {}) or {}).get("bodies", [])}
    ps_alias = (J(os.path.join(PS, "candidates_status.json"), {}) or {}).get("aliases", {})

    # who owns each number: the key (or planet-search label) of the card it was first given to. When a group is split
    # (not_same_body.json, 2026-11-22), every half's leads point at the old number; the half holding the owner keeps it,
    # the other half gets a new number and its leads are re-registered (TRKSUB01: halves overwrote each other's card)
    OWNER = {}
    for f in glob.glob(os.path.join(C, "C[0-9][0-9][0-9]*", "card.json")):
        o = J(f, {}) or {}
        k = o.get("key") or (o.get("identities") or {}).get("label")
        if k and o.get("cid"):
            OWNER[o["cid"]] = k
    ALL = [set([p["key"]] + [a for a, k in ps_alias.items() if k == p["key"]]) for p in obj["planet_search"]] + \
          [set([b["id"]] + list(b.get("refinds") or []) + ([b["label"]] if b.get("label") else [])) for b in obj["decam_big_tno"]]

    def valid(c, names):
        k = OWNER.get(c)
        if not k or k in names:
            return True
        return not any(k in nm for nm in ALL if nm is not names and nm != set(names))

    def cid_for(names):
        hit = sorted({reg["ids"][n] for n in names if n in reg["ids"]})
        hit = [h for h in hit if valid(h, set(names))]
        if hit:
            c = hit[0]
            for other in hit[1:]:
                old = J(os.path.join(C, other, "card.json"), {})
                if old:
                    old["active"] = False; old["merged_into"] = c; W(os.path.join(C, other, "card.json"), old)
        else:
            c = f"C{reg['next']:03d}"; reg["next"] += 1
        for n in names:
            reg["ids"][n] = c
        return c

    cards, seen = [], set()
    # planet-search bodies first (C###, C### are the filed TRKSUB01, TRKSUB02)
    for p in obj["planet_search"]:
        full = ps_all.get(p["key"], p)
        names = [p["key"]] + [a for a, k in ps_alias.items() if k == p["key"]]
        c = cid_for(names); seen.add(c)
        folder = os.path.join(C, c); ev = os.path.join(folder, "evidence"); os.makedirs(ev, exist_ok=True)
        W(os.path.join(ev, "planet_search_status.json"), full)
        o = orbit_from_facts(full.get("facts"))
        cards.append(dict(cid=c, name=full.get("title") or p["key"], kind="planet-search", active=True, stage=p["stage"],
                          status=full.get("status"), summary=full.get("summary"),
                          identities=dict(catalogue=c, label=p["key"], field=full.get("field"), aliases=names[1:],
                                          search="planet-search hill climbs (fields F1-F10)", found_by=full.get("found_by")),
                          orbit=o, H=None, D_km=None, arc_years=None, nights=None, facts=full.get("facts"),
                          ladder=ladder_planet(full), evidence=["planet_search_status.json"],
                          sources=["planet-search/rsi/candidates/candidates_status.json", "planet-search/rsi/candidates/NOTES.md"]))
    for b in obj["decam_big_tno"]:
        names = [b["id"]] + list(b.get("refinds") or []) + ([b["label"]] if b.get("label") else [])
        c = cid_for(names); seen.add(c)
        folder = os.path.join(C, c); ev = os.path.join(folder, "evidence"); os.makedirs(ev, exist_ok=True)
        files = evidence_dbt(b, ev)
        lead = J(os.path.join(ev, "lead.json"), {}) or {}
        cards.append(dict(cid=c, name=b["label"] or re.sub(r"^dbt-[a-z0-9]+-", "", b["id"]).rsplit("-p", 1)[0], kind="decam-big-tno",
                          active=True, stage=b["stage"], key=b["id"],
                          identities=dict(catalogue=c, label=b.get("label"), lead_id=b["id"], also_found_as=b.get("refinds") or [],
                                          search=lead_search(b["run"]), experiment=b["experiment"], found_in=b.get("found_in"),
                                          run=b["run"], program=b["tag"], region=b["region"], rank=b["rank"], p=b["p"],
                                          scenario=lead.get("sid"), detections=lead.get("det"), observer_batch=b.get("sent"),
                                          rubin_rights=b.get("rubin"), known_as=b.get("known_as")),
                          orbit=dict(a=b.get("a"), e=b.get("e"), i=b.get("i"), r=b.get("r"), rms=b.get("rms")),
                          H=b.get("H"), D_km=[size(b["H"], 0.10), size(b["H"], 0.05)] if b.get("H") is not None else None,
                          arc_years=b.get("arc_years"), nights=b.get("nights"), now_sigma=b.get("now_sigma"),
                          verdicts=dict(lead_check=b.get("lead_check"), novelty=b.get("novelty"), fixed_sources=b.get("fixed_source_text"),
                                        other_archives=b.get("archives"), archives_checked=b.get("archives_checked"),
                                        mpc_tracklets=b.get("itf"), mpc_tracklets_kind=b.get("itf_kind"), hand=b.get("hand")),
                          ladder=ladder_dbt(b), evidence=files,
                          sources=[f"runs/{b['run']}/", "progress/objects/objects.json"]))
    dup = [c for c, n in collections.Counter(x["cid"] for x in cards).items() if n > 1]
    if dup:
        raise SystemExit(f"build.py: two bodies share a catalogue number {dup}; nothing written")
    # history and dropped bodies
    OVERRIDE = (J(os.path.join(HERE, "progress", "objects", "stage_override.json"), {}) or {}).get("cards", {})
    SINGLENIGHT = J(os.path.join(PS, "allarch", "doublecheck", "singlenight.json"), {}) or {}
    STRONG_BEFORE = set((J(os.path.join(C, "strong_before_review.json"), {}) or {}).get("cards", []))
    SINGLENIGHT2 = J(os.path.join(PS, "allarch", "doublecheck", "singlenight2.json"), {}) or {}
    HAND_DIST = (J(os.path.join(C, "assumed_distance_hand.json"), {}) or {}).get("cards", {})   # by-hand assumed distances (C###, 2026-11-23)
    for card in cards:
        # the arc refit without fixed-source detections (harness/arc_refit.py) is the card's own-place orbit when it exists and is
        # bound: before 2026-11-22 23:4x the card showed the arc file's orbit and nights (TRKSUB01: C### 22 vs 18 nights)
        rf0 = os.path.join(HERE, "runs", card["identities"].get("run") or "", "arc", (card.get("key") or "") + ".refit.json")
        R0 = J(rf0, None) if os.path.exists(rf0) else None
        if isinstance(R0, dict) and (R0.get("a") or 0) > 0 and (R0.get("e") if R0.get("e") is not None else 1) < 1:
            card["orbit"] = dict(card.get("orbit") or {}, a=R0["a"], e=R0.get("e"), i=R0.get("i"), r=R0.get("r", (card.get("orbit") or {}).get("r")),
                                 rms=R0.get("rms"), refit=True)
            card["arc_years"], card["nights"] = R0.get("arc_years"), R0.get("nights")
        # TRKSUB01's joint orbit (our arc + SAME BODY archive rows, refit without fixed sources): the card's orbit
        # and arc when it is longer (2026-11-21)
        jf = os.path.join(PS, "allarch", "joint", card["cid"] + "_joint.json")
        jo = J(jf, None) if os.path.exists(jf) else None
        if isinstance(jo, dict) and jo.get("a") is not None:
            shutil.copy2(jf, os.path.join(C, card["cid"], "evidence", "joint_orbit.json"))
            card["evidence"] = sorted(set(card["evidence"]) | {"joint_orbit.json"})
            card["orbit"] = dict(card.get("orbit") or {}, a=jo["a"], e=jo.get("e"), i=jo.get("i"), r=jo.get("r"),
                                 rms=jo.get("rms"), joint=True)
            if (jo.get("arc_years") or 0) > (card.get("arc_years") or 0):
                card["arc_years"], card["nights"] = jo["arc_years"], jo.get("nights")
            elif (jo.get("arc_years") or 0) >= (card.get("arc_years") or 0) - 0.01 and jo.get("nights"):
                card["nights"] = jo["nights"]   # same arc: the joint file's count, which leaves out fixed-source places (TRKSUB01 2026-11-23: C### 7 vs 9, C### 7 vs 6)
        # foreign places dropped (TRKSUB01's merges_drop / split4_drop lists, harness/arc_refit_nolone.py --drop-mjd, 2026-11-22):
        # the card's orbit and arc come from its own places only
        dfile = os.path.join(HERE, "runs", card["identities"].get("run") or "", "arc", (card.get("key") or "") + ".drop.json")
        D = J(dfile, None) if os.path.exists(dfile) else None
        if isinstance(D, dict) and (card.get("orbit") or {}).get("joint") and isinstance(jo, dict) and jo.get("arc_years") is not None:
            # places dropped and TRKSUB01's joint orbit refit on what is left (u20drop, 2026-11-23: C### C### C### showed the
            # pre-drop 1.25 / 0.86 / 4.11 yr arcs under one-season core orbits): the joint file's arc and nights, whatever their length
            card.setdefault("verdicts", {})["foreign_dropped"] = dict(n=len(D["dropped"]), fit=D.get("fit"), reason=D.get("reason"),
                                                                      arc_years=jo.get("arc_years"), nights=jo.get("nights"))
            card["arc_years"], card["nights"] = jo["arc_years"], jo.get("nights", card.get("nights"))
        elif isinstance(D, dict) and not (card.get("orbit") or {}).get("joint"):    # a joint orbit (with archive rows) is TRKSUB01's to refit
            card.setdefault("verdicts", {})["foreign_dropped"] = dict(n=len(D["dropped"]), fit=D.get("fit"), reason=D.get("reason"),
                                                                      arc_years=D.get("arc_years"), nights=D.get("nights"))
            card["arc_years"], card["nights"] = D.get("arc_years"), D.get("nights")
            ns = ((D.get("now") or [{}])[0] or {}).get("sigma_arcsec") or 0
            # a "now" place uncertain by > 10 deg, or by > 1 deg on a one-season arc under 0.2 yr (TRKSUB01 2026-11-23, C###:
            # 0.1 yr, 15,800": it constrains the season's motion, not the orbit), says the orbit is not constrained
            if D.get("fit") and ns <= 36000 and not ((D.get("arc_years") or 0) < 0.2 and ns > 3600):
                card["orbit"] = dict(card.get("orbit") or {}, a=D["a"], e=D.get("e"), i=D.get("i"), r=D.get("r"), rms=D.get("rms"),
                                     joint=False, foreign_dropped=True)
            else:
                card["orbit"] = dict(a=None, e=None, i=None, r=None, rms=None, foreign_dropped=True,
                                     note="own places alone do not constrain an orbit")
                card["H"], card["D_km"] = None, None          # distance open, so no H or size (2026-11-22)
        # own places fix no orbit: distance under an assumed bound prograde orbit (e <= 0.3), harness/arc_circfit.py
        # (TRKSUB01's request, 2026-11-23: the ten u20 cards left with one season after u20drop). Shown as an assumed range,
        # never as a measured orbit; a, e, i stay blank. Cards whose places fit no such orbit (fit_ok false, e.g. C###) get nothing.
        cfile = os.path.join(HERE, "runs", card["identities"].get("run") or "", "arc", (card.get("key") or "") + ".circ.json")
        CF = J(cfile, None) if os.path.exists(cfile) else None
        if isinstance(CF, dict) and (card.get("orbit") or {}).get("a") is None:
            B = CF.get("bound_emax") or {}
            ok = CF.get("fit_ok") and B.get("r_lo") is not None and (B.get("rms_best") or 9) <= 0.30
            if ok:
                hb = CF.get("H_best"); h1, h2 = B.get("H_at_r_lo"), B.get("H_at_r_hi")
                card["H"] = hb
                card["D_km"] = [size(hb, 0.10), size(hb, 0.05)] if hb is not None else None
                card["assumed_distance"] = dict(model="bound prograde orbit, e <= 0.3 (circular best as point value)",
                                                r_lo=B["r_lo"], r_hi=B["r_hi"], r_best=CF.get("r_best"), H_best=hb,
                                                H_lo=min(h1, h2), H_hi=max(h1, h2), file=os.path.basename(cfile))
                card["orbit"] = dict(card.get("orbit") or {}, a=None, e=None, i=None, r=CF.get("r_best"), assumed=True,
                                     note=f"distance not measured: an assumed bound prograde orbit (e <= 0.3) fits at r {B['r_lo']:.1f}-{B['r_hi']:.1f} au, "
                                          f"H {max(h1, h2):.1f}-{min(h1, h2):.1f} (circular best r {CF.get('r_best'):.1f} au, H {hb:.1f})")
        HD = (HAND_DIST.get(card["cid"]) or None)
        if HD and (card.get("orbit") or {}).get("a") is None and not card.get("assumed_distance"):
            card["H"] = HD["H"]; card["D_km"] = [size(HD["H"], 0.10), size(HD["H"], 0.05)]
            card["assumed_distance"] = dict(model="hand", r_lo=HD["r_lo"], r_hi=HD["r_hi"], H_best=HD["H"], source=HD.get("source"))
            card["orbit"] = dict(card.get("orbit") or {}, a=None, e=None, i=None, r=round((HD["r_lo"] + HD["r_hi"]) / 2, 1),
                                 assumed=True, note=HD["note"])
        if card["cid"] in IMAGETESTS:
            card["image_tests"] = IMAGETESTS[card["cid"]]
        else:
            card.pop("image_tests", None)
        # a known MPC body whose own places do not constrain an orbit shows the MPC catalogue orbit, labelled with its
        # oppositions and uncertainty U (TRKSUB01, 2026-11-23: knownlook_mpc_orbits.json, MPCORB values)
        mo = MPCORB_KNOWN.get(card["cid"])
        if mo and (card.get("orbit") or {}).get("a") is None and card.get("stage") == "known MPC body":
            num = lambda v: int(v) if str(v).strip().isdigit() else 0      # MPCORB fields arrive as strings
            weak = num(mo.get("oppositions")) < 2 or num(mo.get("U")) >= 7
            card["orbit"] = dict(card.get("orbit") or {}, a=mo["a"], e=mo.get("e"), i=mo.get("i"), r=None, rms=None, mpc=True,
                                 note=f"MPC catalogue orbit of {mo.get('body')} ({mo.get('oppositions')} opposition"
                                      f"{'s' if num(mo.get('oppositions')) != 1 else ''}, U {mo.get('U')}"
                                      f"{', arc ' + str(mo.get('arc')) if mo.get('arc') else ''})"
                                      + ("; weak, one-opposition orbit, not a measured one" if weak else "")
                                      + "; our own places alone do not constrain an orbit")
            if mo.get("H") is not None:
                card["H"] = mo["H"]; card["D_km"] = [size(mo["H"], 0.10), size(mo["H"], 0.05)]
        # TRKSUB01's no-bound check (2026-11-21): a free refit without the fixed-source detections slid unbound; the clean
        # detections tested against the original orbit + covariance. CONSISTENT keeps the original orbit.
        nbc = NOBOUND.get(card.get("key") or "")
        if nbc:
            card.setdefault("verdicts", {})["nobound_check"] = nbc
            fs = next((s for s in card["ladder"] if s["step"] == "fixed_sources"), None)
            if fs is not None and nbc.get("verdict") == "CONSISTENT":
                fs["text"] += (f"; the clean detections alone fit the original orbit (rms {nbc.get('rms')}\", "
                               f"the dropped ones lie {max(nbc.get('dropped_offsets') or [0]):.2f}\" or less from it)")
                if (nbc.get("arc_years_keep") or 0) < 0.1:
                    fs["text"] += (f"; caveat: the clean detections span only {nbc.get('arc_years_keep')} years, so the "
                                   "multi-year arc rests on detections that coincide with a fixed source")
        # single-night audit cap (TRKSUB01, 2026-11-22): no card is "strong ... multi-year" when the audit says its multi-year
        # link rests on an untied season and the arc without that season is under a year
        sn = SINGLENIGHT.get(card["cid"])
        sn2 = SINGLENIGHT2.get(card["cid"])
        # a clean refit (harness/arc_refit.py, fixed-source detections dropped) settles "needs a refit" (2026-11-22, TRKSUB01's
        # question): bound, >= 5 detections on >= 3 nights, rms <= 0.2". Multi-year only when the arc is >= 1 yr both in the
        # refit and in the single-night audit's arc without untied seasons; otherwise short arc. Else the stage stays.
        if card.get("stage") == "seen again, needs a refit":
            rf = os.path.join(HERE, "runs", card["identities"].get("run") or "", "arc", (card.get("key") or "") + ".refit.json")
            R = J(rf, None) if os.path.exists(rf) else None
            if isinstance(R, dict) and (R.get("a") or 0) > 0 and (R.get("e") or 1) < 1 and (R.get("n") or 0) >= 5 \
                    and (R.get("nights") or 0) >= 3 and (R.get("rms") or 9) <= 0.2:
                arc = min(R.get("arc_years") or 0, (sn or {}).get("arc_years_without", 99) or 0)
                card.setdefault("verdicts", {})["refit"] = dict(n=R["n"], nights=R["nights"], rms=R["rms"], a=R["a"], e=R.get("e"),
                                                                arc_years=R.get("arc_years"), arc_years_audit=arc,
                                                                dropped=len(R.get("dropped") or []))
                # strong comes only from TRKSUB01's review (override list), 2026-11-22 21:1x: arcs >= 1 yr whose seasons
                # fit one orbit but do not predict each other were promoted by the old rule (7 of 21 demoted on review)
                card["stage"] = "seen again, clean, short arc"
            elif isinstance(R, dict) and R.get("fit") is None and "no bound orbit" in str(R.get("reason", "")) \
                    and (R.get("n") or 0) >= 5 and (R.get("nights") or 0) >= 3:
                # clean places that a days-long arc cannot fit to a bound orbit: the orbit is not constrained (TRKSUB01's
                # boundfit, 2026-11-22: bound orbits fit nearly as well), not a failed body
                card.setdefault("verdicts", {})["refit"] = dict(n=R["n"], nights=R["nights"], fit=None, reason=R.get("reason"),
                                                                arc_years=R.get("arc_years"))
                card["orbit"] = dict(a=None, e=None, i=None, r=None, rms=None, note="own places alone do not constrain an orbit")
                card["H"], card["D_km"] = None, None          # distance open, so no H or size (2026-11-22)
                card["arc_years"], card["nights"] = R.get("arc_years"), R.get("nights")
                card["stage"] = "seen again, clean, short arc"
               # joint test of single detections in 2+ seasons decides those cards
        if sn2 and card.get("stage") == "strong: seen again, clean, multi-year arc" and sn2.get("verdict") == "UNTIED":
            card.setdefault("verdicts", {})["arc_cap"] = dict(was=card["stage"], audit="UNTIED (joint)",
                                                               untied_years=sn2.get("untied_years"))
            card["stage"] = "seen again, clean, short arc"
        elif sn and not sn2 and card.get("stage") == "strong: seen again, clean, multi-year arc" and \
                sn.get("verdict") in ("ARC RESTS ON ONE DETECTION", "UNSURE") and (sn.get("arc_years_without") or 0) < 1 and \
                len([c for c in sn.get("carriers") or [] if c.get("status") == "chance"]) == 1 and \
                any(c.get("status") == "chance" and (c.get("chance_P") or 0) >= 0.42 for c in sn.get("carriers") or []):
            # one untied season with P >= 0.42 (every mislink in the audit's known-body control had P >= 0.42); cards with
            # two or more untied seasons wait for the multi-season audit (singlenight2)
            card.setdefault("verdicts", {})["arc_cap"] = dict(was=card["stage"], audit=sn.get("verdict"),
                                                               arc_years_without=sn.get("arc_years_without"))
            card["stage"] = "seen again, clean, short arc"
        # a card reaches strong only through TRKSUB01's review (override) unless it was strong before the rule (2026-11-22 21:3x;
        # C###, C### had become strong straight from their stored lead check + fixed-source test)
        if card.get("stage") == "strong: seen again, clean, multi-year arc" and card["cid"] not in STRONG_BEFORE:
            card["stage"] = "seen again, clean, short arc"
        ov = OVERRIDE.get(card["cid"])                    # hand stage overrides (progress/objects/stage_override.json, 2026-11-22)
        if ov and card.get("stage") not in ("known MPC body", "probably a known MPC body", "known to another survey, not in the MPC", "probably known to another survey, not in the MPC"):
            card.setdefault("verdicts", {})["override"] = dict(stage=ov["stage"], note=ov.get("note", ""), was=card.get("stage"))
            card["stage"] = ov["stage"]
            # an override that says another archive saw the body also sets the ladder's archive step, with the override's
            # note as its text (TRKSUB01 2026-11-23, C###: one Subaru HSC night, the archive line still read NO DECISIVE DATA)
            if ov["stage"] in ("seen in another archive", "seen once in another archive"):
                for st_ in card["ladder"]:
                    if st_["step"] == "other_archives" and st_["state"] not in ("pass",):
                        st_["state"] = "pass" if ov["stage"] == "seen in another archive" else "partial"
                        st_["text"] = ov.get("note") or st_.get("text")
        # show the audited arc when the single-night audit finds the long arc resting on untied seasons and it is far shorter
        # (TRKSUB01, 2026-11-23: C###, C### showed 1.13 / 1.91 yr where only 0.1 / 0.08 yr is tied); not for strong (reviewed)
        # or known cards, nor where the joint multi-season test tied the seasons
        # "seen in another archive" is set only by TRKSUB01's override after an independent sighting on the long orbit
        # (2026-11-23: C### moved there from strong with a 14-yr joint orbit and was blanked)
        if sn and card.get("stage") not in ("strong: seen again, clean, multi-year arc", "seen in another archive", "known MPC body", "probably a known MPC body",
                                            "known to another survey, not in the MPC", "probably known to another survey, not in the MPC") \
                and sn.get("verdict") in ("ARC RESTS ON ONE DETECTION", "UNSURE") and not (sn2 and sn2.get("verdict") != "UNTIED") \
                and any(c_.get("status") in ("chance", "untied") for c_ in sn.get("carriers") or []):
            aw = sn.get("arc_years_without")
            if aw is not None and (card.get("arc_years") or 0) > 0 and aw < 0.5 * card["arc_years"]:
                card["arc_years_full"] = card["arc_years"]; card["arc_years"] = aw
                card.setdefault("verdicts", {})["arc_shown"] = f"audited arc {aw} yr; {card['arc_years_full']} yr only with untied single detections"
                # the full fit's elements rest on those untied detections too (TRKSUB01, C###: a 181 au from two chance singles)
                card["orbit"] = dict(card.get("orbit") or {}, a=None, e=None, i=None, rms=None, note="orbit not established: "
                                     "the long fit rests on untied single detections")
        f = os.path.join(C, card["cid"], "card.json")
        old = J(f, {}) or {}
        hist = old.get("history") or []
        if not hist or hist[-1]["stage"] != card["stage"]:
            hist.append(dict(time=NOW, stage=card["stage"]))
        card["history"] = hist
        card["added"] = old.get("added") or NOW
        card["updated"] = NOW
        W(f, card)
        open(os.path.join(C, card["cid"], "README.md"), "w").write(readme(card))
    dropped = []
    # 2026-11-22 (TRKSUB01): a dropped card proven the same body as an active one (same_body.json) is marked merged into it, even
    # when gather.py never groups the two (its lead is filtered out, e.g. failed fixed-source test: C### = C### = 2013 TS227)
    same_to = {}
    for pr in (J(os.path.join(HERE, "progress", "objects", "same_body.json"), {}) or {}).get("pairs", []):
        ca, cb = reg["ids"].get(pr[0]), reg["ids"].get(pr[1])
        if ca and cb and ca != cb:
            if ca in seen and cb not in seen:
                same_to[cb] = ca
            elif cb in seen and ca not in seen:
                same_to[ca] = cb
    for f in sorted(glob.glob(os.path.join(C, "C[0-9][0-9][0-9]*", "card.json"))):
        c = os.path.basename(os.path.dirname(f))
        if c in seen:
            continue
        old = J(f, {})
        if c in same_to and old.get("merged_into") != same_to[c]:
            old["active"] = False; old["merged_into"] = same_to[c]; old["dropped_reason"] = f"merged into {same_to[c]} (same body, joint orbit)"
            W(f, old)
        lid = (old.get("identities") or {}).get("lead_id")
        if lid:                                   # keep its evidence current too
            m = re.match(r"dbt-([a-z0-9]+)-", lid)
            old["evidence"] = evidence_dbt(dict(id=lid, refinds=(old["identities"].get("also_found_as") or []), run=m.group(1) if m else ""),
                                           os.path.join(C, c, "evidence"))
            if not old.get("active") and not old.get("merged_into"):
                old["dropped_reason"] = dropped_reason(lid)
            W(f, old)
        if old.get("active"):
            old["active"] = False
            old["dropped_reason"] = old.get("merged_into") and f"merged into {old['merged_into']}" or dropped_reason(old["identities"].get("lead_id", ""))
            old["history"] = (old.get("history") or []) + [dict(time=NOW, stage="dropped")]
            W(f, old); open(os.path.join(C, c, "README.md"), "w").write(readme(old))
        dropped.append(old)
    W(os.path.join(C, "registry.json"), reg)
    allc = sorted(cards + dropped, key=lambda x: x["cid"])
    W(os.path.join(C, "index.json"), dict(asof=NOW, n_active=len(cards), n_dropped=len(dropped), cards=allc))
    new = [c["cid"] for c in cards if c["added"] == NOW]
    print(f"{len(cards)} active candidates, {len(dropped)} dropped; new this run: {len(new)} {new[:20]}")


if __name__ == "__main__":
    main()
