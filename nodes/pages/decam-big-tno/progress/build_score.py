#!/usr/bin/env python3
"""Score tracker data for the claude.ai artifact "Big TNO Hill Climb" (<private-page-url>;
2026-11-13).

Reads runs/<run>/state.json, each round's grade.json, the leads ledger, the novelty queue's RESULTS.md, lead_check results
and the OBSERVER_NAME ledger; writes progress/out/score.json with the lists summary, rounds, regions, funnel, events. Counts
only for candidates: no positions, dates or names of unreported bodies. Region letters only (A=REG_A, B=REG_B, C=REG_C).
Then progress/docs.py BLOB_URL [--page] writes the store documents for the ArtifactData batch.
Usage: python3 -I progress/build_score.py [--run r1]
"""
import argparse, glob, json, os, re
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NQ = os.path.expanduser("$ALE_HOME/planet-search/rsi/candidates/novelty_queue/RESULTS.md")
PHASE = {"new": "agent writing", "running": "agent writing", "ended": "agent finished", "synced": "agent finished",
         "grading": "being graded"}


def load(p, d):
    try:
        return json.load(open(p))
    except Exception:
        return d


def note_method(sub):
    try:
        for line in open(os.path.join(sub, "NOTES.md"), errors="replace").read(20000).splitlines()[:12]:
            if line.lstrip("#*- ").lower().startswith("method:"):
                t = line.split(":", 1)[1].strip()
                return t if len(t) <= 180 else t[:180].rsplit(" ", 1)[0].rstrip(",;:(") + "…"   # cut at a word, never mid-word
    except Exception:
        pass
    return ""


def run1(run="r2"):
    """Run 1 of the task's current form (2026-11-14). Internally this is run r2, design 2:
    two rewards (recall, GPU-hours), four graded regions, nothing hidden from the agent. The page calls it Run 1; folders
    and trial labels keep r2. Lists: run1_summary, run1_rounds, run1_regions, run1_events. Region ids only, body counts only.
    runs/r2/review.json {gen: text} holds what reading that round's program showed (written by the session per round)."""
    RUN = os.path.join(HERE, "runs", run)
    st = load(os.path.join(RUN, "state.json"), {}); cfg = load(os.path.join(RUN, "config.json"), {})
    sids = cfg.get("sids", []); rid = cfg.get("regions", {}); review = load(os.path.join(RUN, "review.json"), {})
    best = (st.get("best") or {}).get("gen")
    # Rounds: one program per round up to gen(pf-1); from gen(pf) two programs per round, one per model (config pairs_from;
    # 2026-11-14). gen003 + gen004 = round 3.
    pf = cfg.get("pairs_from"); pu = cfg.get("pairs_until")   # pairs_until: last gen of the paired rounds (38; one Astra per round after)
    rnd0 = lambda n: n if not pf or n < pf else pf + (n - pf) // 2
    rnd = lambda n: rnd0(n) if not pu or n <= pu else rnd0(pu) + (n - pu)
    def who(gen, g):
        n = int(gen[3:]); name = "" if n == 0 else (g.get("agent") or {}).get("name") or "Claude Fable 5.2"
        short = "Astra" if "Astra" in name else "Fable" if name else ""
        pair = bool(pf) and n >= pf and (not pu or n <= pu)
        return rnd(n), name, (f"round {rnd(n)} {short}" if pair else f"round {rnd(n)}"), ((-1 if short == "Fable" else 1) if pair else 0)
    def plain(text):                                        # the agents' notes name programs by folder; the page names them by round
        text = re.sub(r"(?:prior[- _]run |prior_r1/)gen0*(\d+)", lambda m: f"the earlier run's round {m.group(1)}", text)
        return re.sub(r"\bgen(\d{3})\b", lambda m: who(m.group(0), st.get("gens", {}).get(m.group(0), {}))[2] if int(m.group(1)) else "the starting program", text)
    rounds, regions, top, top_found, pend, cur = [], [], None, set(), [], None
    for gen, g in sorted(st.get("gens", {}).items()):
        rd, agent, label, slot = who(gen, g)
        if rd != cur:                                       # a round's programs are compared with the best before the round
            for key, f in pend:
                if top is None or key[:2] > top[:2]:
                    top, top_found = key, f
            pend, cur = [], rd
        done = g.get("phase") == "done"
        d = load(os.path.join(g.get("grade_dir") or os.path.join(RUN, "gens", gen, "grade"), "grade.json"), {}) if done else {}
        per = d.get("per_scenario", {})
        state = (PHASE.get(g.get("phase")) or {"valid": "graded", "invalid": "scored 0 (rule)", "unchanged": "unchanged"}.get(g.get("status"), g.get("status", "")))
        mins = None
        try:
            js = glob.glob(os.path.join(RUN, "gens", gen, f"job-a{g.get('attempt', 1)}", "*", "result.json"))
            if js:
                ae = load(js[0], {}).get("agent_execution") or {}; t0, t1 = ae.get("started_at"), ae.get("finished_at")
                if t0 and t1:
                    mins = round((datetime.fromisoformat(t1.replace("Z", "+00:00")) - datetime.fromisoformat(t0.replace("Z", "+00:00"))).total_seconds() / 60)
        except Exception:
            pass
        found = {(s, b) for s, v in (d.get("found") or {}).items() for b in v}
        row = {"gen": gen, "round": rd, "agent": agent, "label": label, "slot": slot, "state": state, "recall": g.get("reward") if done else None,
               "gpu_hours": g.get("gpu_hours") if done else None,
               "known_found": sum(v.get("known_found", 0) for v in per.values()) if per else None,
               "known_total": sum(v.get("known_findable", 0) for v in per.values()) if per else None,
               "leads": g.get("leads") if done else None, "agent_minutes": mins, "best": gen == best,
               "started": datetime.fromtimestamp(g["started"]).strftime("%m-%d %H:%M") if g.get("started") else "",
               "vs": top[3] if top and per else "", "gained": len(found - top_found) if top and per else None,
               "lost": len(top_found - found) if top and per else None,
               "method": ("starting program: the earlier run's round 19 program, graded again under Run 1's rules" if gen == "gen000"
                          else plain(note_method(g.get("submission") or ""))),
               "check": review.get(gen, "")}
        for s in sids:
            v = per.get(s, {})
            row["found_" + rid.get(s, s)] = v.get("known_found")
            if v:
                regions.append({"gen": gen, "round": rd, "agent": agent, "label": label, "slot": slot, "region": rid.get(s, s), "found": v.get("known_found"),
                                "findable": v.get("known_findable"), "seconds": round(v.get("seconds") or 0)})
        rounds.append(row)
        if per and g.get("status") == "valid":             # the run's rule: highest recall, then fewer GPU-hours
            pend.append(((g.get("reward") or 0, -(g.get("gpu_hours") or 0), gen, label), found))
    lead = {}                                               # the better program of each round (the region chart's line follows it)
    for r in rounds:
        if r["recall"] is not None and (r["round"] not in lead or (r["recall"], -(r["gpu_hours"] or 0)) > (lead[r["round"]]["recall"], -(lead[r["round"]]["gpu_hours"] or 0))):
            lead[r["round"]] = r
    for x in regions:
        x["lead"] = lead.get(x["round"], {}).get("gen") == x["gen"]
    graded = [r for r in rounds if r["recall"] is not None and r["gen"] != "gen000"]
    byr = {}
    for r in rounds:
        if r["gen"] != "gen000":
            byr.setdefault(r["round"], []).append(r)
    full = lambda rd: 2 if pf and rd >= rnd(pf) and (not pu or rd <= rnd(pu)) else 1
    rounds_done = sum(1 for rd, v in byr.items() if len(v) >= full(rd) and all(r["state"] not in ("agent writing", "agent finished", "being graded") for r in v))
    b = next((r for r in rounds if r["best"]), None); s0 = next((r for r in rounds if r["gen"] == "gen000"), {})
    live = [r for r in rounds if r["state"] in ("agent writing", "agent finished", "being graded")]
    summ = {"asof": datetime.now().strftime("%Y-%m-%d %H:%M"), "best_recall": b["recall"] if b else None, "best_gen": b["gen"] if b else None,
            "best_name": (b["label"] + (f" ({b['agent']})" if b["agent"] else "")) if b else None, "programs_graded": len(graded),
            "best_found": b["known_found"] if b else None, "best_gpu_hours": b["gpu_hours"] if b else None,
            "known_total": s0.get("known_total"), "start_recall": s0.get("recall"), "start_gpu_hours": s0.get("gpu_hours"),
            "start_found": s0.get("known_found"), "rounds_graded": rounds_done, "rounds_target": rnd(max(st.get("target") or 0, cfg.get("target_planned") or 0)),
            "limit_gpu_hours": round(len(sids) * cfg.get("budget_s", 0) / 3600, 2),
            "writing": ", ".join(r["label"] for r in live if r["state"] in ("agent writing", "agent finished")) or "none",
            "grading": ", ".join(r["label"] for r in live if r["state"] == "being graded") or "none"}
    labels = {r["gen"]: r["label"] for r in rounds}
    events = []
    try:
        lines = open(os.path.join(RUN, "climb.log"), errors="replace").read().splitlines()[-400:]
    except Exception:
        lines = []
    for line in lines:
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d):\d\d (gen\d{3}) (.*)", line)
        if not m:
            continue
        x = m.group(3)
        a = re.match(r"started dbt-\S+-a(\d+)", x); r_ = re.match(r"reward ([\d.]+) \(NERSC A100\) gpu_hours ([\d.]+) leads (\d+)", x)
        if a:
            txt = "agent started" + (f" (attempt {a.group(1)})" if a.group(1) != "1" else "")
            ag = next((r["agent"] for r in rounds if r["gen"] == m.group(2)), "")
            txt += f": {ag}" if ag else ""
        elif r_:
            txt = f"graded: recall {float(r_.group(1)):.4f}, {r_.group(2)} GPU-hours, {r_.group(3)} leads"
        elif x.startswith("written;"):
            txt = "program written; queued for grading on NERSC"
        elif "node granted" in x:
            txt = "NERSC node granted"
        elif re.match(r"(is the best|left |unchanged|invalid|attempt |agent stopped|10 errors)", x):
            txt = re.sub(r"dbt-\S+", "", x).strip()[:160]
        else:
            continue
        txt = re.sub(r"\bgen(\d{3})\b", lambda k: labels.get(k.group(0), k.group(0)), txt)
        events.append({"t": m.group(1), "gen": m.group(2), "who": labels.get(m.group(2), m.group(2)), "text": txt})
    events = events[-40:][::-1]
    for i, e in enumerate(events):
        e["n"] = len(events) - i
    return {"run1_summary": [summ], "run1_rounds": rounds, "run1_regions": regions, "run1_events": events}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="r1"); a = ap.parse_args()
    RUN = os.path.join(HERE, "runs", a.run)
    st = load(os.path.join(RUN, "state.json"), {}); cfg = load(os.path.join(RUN, "config.json"), {})
    sids = cfg.get("sids", []); letter = {s: "ABC"[i] for i, s in enumerate(sids)}
    best = (st.get("best") or {}).get("gen")
    rounds, regions = [], []
    for gen, g in sorted(st.get("gens", {}).items()):
        gd = g.get("grade_dir") or os.path.join(RUN, "gens", gen, "grade")
        d = load(os.path.join(gd, "grade.json"), {})
        per = d.get("per_scenario", {})
        done = g.get("phase") == "done"
        state = (PHASE.get(g.get("phase")) or {"valid": "graded", "invalid": "scored 0 (rule)", "unchanged": "unchanged"}.get(g.get("status"), g.get("status", "")))
        mins = None
        try:
            js = glob.glob(os.path.join(RUN, "gens", gen, f"job-a{g.get('attempt', 1)}", "*", "result.json"))
            if js:
                r = load(js[0], {}); ae = r.get("agent_execution") or {}
                t0, t1 = ae.get("started_at"), ae.get("finished_at")
                if t0 and t1:
                    mins = round((datetime.fromisoformat(t1.replace("Z", "+00:00")) - datetime.fromisoformat(t0.replace("Z", "+00:00"))).total_seconds() / 60)
        except Exception:
            pass
        row = {"gen": gen, "round": int(gen[3:]), "state": state,
               "score": g.get("reward") if done and g.get("grader") in ("nersc", "scan") else None,
               "score_gb200": g.get("reward_vista") if g.get("reward_vista") is not None else (g.get("reward") if done and g.get("grader") not in ("nersc", "scan") else None),
               "known_found": sum(v.get("known_found", 0) for v in per.values()) if per and g.get("grader") == "nersc" else None,
               "known_total": sum(v.get("known_findable", 0) for v in per.values()) if per else None,
               "leads": g.get("leads") if done else None, "base": g.get("seed") or ("starter" if gen == "gen001" else ""),
               "agent_minutes": mins, "best": gen == best,
               "started": datetime.fromtimestamp(g["started"]).strftime("%m-%d %H:%M") if g.get("started") else "",
               "method": "starter: the plain search" if gen == "gen000" else note_method(g.get("submission") or "")}
        for s in sids:
            v = per.get(s, {}) if g.get("grader") == "nersc" else {}
            row[f"found_{letter[s]}"] = v.get("known_found")
            if v:
                regions.append({"gen": gen, "round": int(gen[3:]), "region": letter[s], "found": v.get("known_found"),
                                "findable": v.get("known_findable"), "seconds": round(v.get("seconds") or 0)})
        rounds.append(row)
    # candidates funnel (counts only)
    ledger = load(os.path.join(RUN, "leads_sent.json"), [])
    sent_ids = {x["id"] for x in ledger}
    nov = {}
    if os.path.exists(NQ):
        for line in open(NQ, errors="replace"):
            m = re.match(r"\|\s*[\d-]+ [\d:]+\s*\|\s*(dbt-%s-\S+)\s*\|\s*([^|]+)\|" % re.escape(a.run), line)
            if m and m.group(1) in sent_ids:
                nov[m.group(1)] = m.group(2).strip()
    lc = {}
    for f in [f for f in glob.glob(os.path.join(RUN, "lead_check", "gen*.json")) if re.search(r"gen\d{3}\.json$", f)]:
        for x in load(f, []):
            lc[x["id"]] = x.get("verdict", "")
    conf = {}
    for f in glob.glob(os.path.join(RUN, "confirm", "*.json")):
        x = load(f, {}); conf[x.get("id")] = x.get("verdict", "")
    qz = load(os.path.join(RUN, "observer_sent.json"), [])
    total_leads = sum((r["leads"] or 0) for r in rounds if r["gen"] != "gen000")
    notin = [i for i, v in nov.items() if v.startswith("NOT IN MPC")]
    real = [i for i in notin if lc.get(i, "").startswith(("SEEN AGAIN", "LIKELY REAL"))]
    funnel = [
        {"step": 1, "stage": "Leads in the scored top list (all graded rounds)", "count": total_leads},
        {"step": 2, "stage": "Sent to the novelty check", "count": len(sent_ids)},
        {"step": 3, "stage": "Checked: known to the MPC", "count": sum(1 for v in nov.values() if v.startswith(("KNOWN", "PROBABLY", "OURS")))},
        {"step": 4, "stage": "Checked: not in the MPC's files", "count": len(notin)},
        {"step": 5, "stage": "Seen again where its orbit predicts, in the region's exposures the program did not use", "count": len(real)},
        {"step": 6, "stage": "Seen in images the program never used", "count": sum(1 for i in real if conf.get(i, "").startswith("SEEN"))},
        {"step": 7, "stage": "Sent to OBSERVER_NAME for follow-up", "count": sum(len(b.get("ids", [])) for b in qz)},
    ]
    # one row per candidate (2026-11-13);
    # no sky positions, dates or image ids: only what the checks measured
    nres = {}
    for x in ledger:
        rp = os.path.join(os.path.dirname(NQ), "done", x["id"] + ".result.json")
        nres[x["id"]] = load(rp, {})
    lcr = {}
    for f in [f for f in glob.glob(os.path.join(RUN, "lead_check", "gen*.json")) if re.search(r"gen\d{3}\.json$", f)]:
        for y in load(f, []):
            lcr[y["id"]] = y
    qsent = {}
    for bt in qz:
        for i in bt.get("ids", []):
            qsent[i] = bt.get("at", "")
    cands = []
    for x in ledger:
        cid = x["id"]; m = re.match(r"dbt-\w+-(gen\d{3})-([A-C])-rank(\d+)-p([\d.]+)", cid)
        if not m:
            continue
        gen, reg, rank, p = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
        nv = nov.get(cid, ""); n_ = nres.get(cid, {}); c = lcr.get(cid, {}); cf = conf.get(cid, "")
        novelty = ("waiting" if not nv else "not in the MPC's files" if nv.startswith("NOT IN MPC") else
                   "check by hand: a catalogued body is near" if nv.startswith("CHECK BY HAND") else "known to the MPC")
        lv = c.get("verdict", "")
        image = ("waiting" if not c else "seen again" if lv.startswith(("SEEN AGAIN", "LIKELY REAL")) else
                 "missing where predicted" if lv.startswith(("MISSING", "NOT CONFIRMED")) else
                 "no unused exposure to test" if lv.startswith("NO EXTRA") else
                 "no bound orbit" if lv.startswith("no bound") else "no exposure to test" if lv.startswith("no test") else
                 "failed to run" if "failed" in lv else "unclear")
        if cid in qsent:
            stage, status = 6, "sent to OBSERVER_NAME"
        elif novelty == "known to the MPC":
            stage, status = 0, "known body"
        elif novelty == "waiting":
            stage, status = 1, "waiting for the novelty check"
        elif image == "seen again" and novelty == "not in the MPC's files":
            stage, status = (5, "seen in other images") if cf.startswith("SEEN") else (4, "plausible: new, and seen again where predicted")
        elif image in ("missing where predicted", "no bound orbit"):
            stage, status = 0, "failed the image check"
        else:
            stage = 2
            status = ("new; image check " + ("pending" if image == "waiting" else image)) if novelty == "not in the MPC's files" else novelty
        D = c.get("D_km") or {}
        a_, e_ = c.get("a"), c.get("e")
        cands.append({"key": f"{gen}-{reg}-{rank}", "round": int(gen[3:]), "gen": gen, "region": reg, "rank": rank, "p": p,
                      "stage": stage, "status": status, "novelty": novelty, "image_check": image,
                      "other_images": cf or "not run yet", "observer": qsent.get(cid, ""),
                      "seen_in": ", ".join([gen] + x.get("seen", [])), "n_rounds": 1 + len(x.get("seen", [])),
                      "detections": len(x["det"]), "nights": c.get("nights"), "arc_days": n_.get("arc_days") or c.get("arc_days"),
                      "snr_med": c.get("snr_med"), "rms": c.get("rms"), "r": c.get("r"), "a": a_, "e": e_, "i": c.get("i"),
                      "q": round(a_ * (1 - e_), 1) if a_ and e_ is not None else None,
                      "Q": round(a_ * (1 + e_), 1) if a_ and e_ is not None else None,
                      "H": c.get("H"), "H_spread": c.get("H_spread"),
                      "D20": D.get("0.20"), "D10": D.get("0.10"), "D05": D.get("0.05"),
                      "expected": c.get("expected"), "found": c.get("found"), "found_new": c.get("found_not_in_lead"),
                      "extra": c.get("extra"), "extra_found": c.get("extra_found"),
                      "null_hits": c.get("null_hits")})
    graded = [r for r in rounds if r["score"] is not None and r["gen"] != "gen000"]
    b = max((r for r in rounds if r["score"] is not None), key=lambda r: r["score"], default=None)
    s0 = next((r["score"] for r in rounds if r["gen"] == "gen000"), None)
    live = [r for r in rounds if r["state"] in ("agent writing", "agent finished", "being graded")]
    summ = {"asof": datetime.now().strftime("%Y-%m-%d %H:%M"), "best_score": b["score"] if b else None, "best_gen": b["gen"] if b else None,
            "best_known": b["known_found"] if b else None, "known_total": next((r["known_total"] for r in rounds if r["known_total"]), None),
            "starter_score": s0, "rounds_graded": len(graded), "rounds_target": st.get("target"),
            "writing": ", ".join(r["gen"] for r in live if r["state"] in ("agent writing", "agent finished")) or "none",
            "grading": ", ".join(r["gen"] for r in live if r["state"] == "being graded") or "none",
            "leads": total_leads, "not_in_mpc": len(notin), "likely_real": len(real),
            "sent_observer": sum(len(x.get("ids", [])) for x in qz)}
    events = []
    for line in open(os.path.join(RUN, "climb.log"), errors="replace").read().splitlines()[-400:]:
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d):\d\d (gen\d{3}) (reward [\d.]+ \(NERSC A100\)|is the best.*|written; queued.*|started .*|left .*|scan: .*)", line)
        if m:
            txt = re.sub(r"dbt-\S+", "", m.group(3)).strip()
            events.append({"t": m.group(1), "gen": m.group(2), "text": txt[:160]})
    events = events[-40:][::-1]
    for i, e in enumerate(events):
        e["n"] = len(events) - i
    summ["plausible"] = sum(1 for c in cands if c["stage"] >= 4)
    summ["candidates"] = len(cands)
    summ["new_candidates"] = sum(1 for c in cands if c["novelty"] == "not in the MPC's files")
    # the data the programs read (2026-11-13): per sealed region, its
    # detection lists' size and its exposures by year; from the scenario copies on Crucial (no truth, no candidates)
    SC = "$DATA_DRIVE/ale-astronomy/rsi/decam-big-tno/build_tmp/scen_sealed"
    A5 = load(os.path.expanduser("$ALE_HOME/planet-search/archive5/regions.json"), {})
    a5 = {r.get("id"): r for r in (A5 if isinstance(A5, list) else A5.get("regions", []))} if A5 else {}
    findable = {s: (next((r for r in rounds if r["gen"] == "gen002"), {}) or {}) for s in sids}
    data_regions, data_years = [], []
    rg = cfg.get("regions") or {}
    rid = rg if isinstance(rg, dict) else {s: r for s, r in zip(sids, rg)}
    for s in sids:
        R = load(os.path.join(SC, s, "region.json"), {}); E = load(os.path.join(SC, s, "exposures.json"), [])
        if not R:
            continue
        nbytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(SC, s, "det_*.npy")))
        ar = a5.get(rid.get(s, ""), {})
        per = (load(os.path.join(RUN, "gens", "gen002", "grade", "grade.json"), {}).get("per_scenario") or {}).get(s, {})
        yrs = sorted({int(2000 + (e["mjd_mid"] - 51544.5) / 365.25) for e in E})
        data_regions.append({"region": letter[s], "area_deg2": ar.get("deg2"), "exposures": R.get("n_exposures"),
                             "detections": R.get("n_detections"), "det_gb": round(nbytes / 1e9, 1), "images_tb": ar.get("tb"),
                             "years": f"{yrs[0]}-{yrs[-1]}" if yrs else "", "seasons": len(yrs),
                             "known_findable": per.get("known_findable"),
                             "centre": f"RA {R['centre_radec'][0]:.0f}, Dec {R['centre_radec'][1]:+.0f}" if R.get("centre_radec") else ""})
        by = {}
        for e in E:
            y = int(2000 + (e["mjd_mid"] - 51544.5) / 365.25); by.setdefault(y, []).append(e)
        for y, L in sorted(by.items()):
            m5 = sorted(x["m5"] for x in L if isinstance(x.get("m5"), (int, float)) and x["m5"] == x["m5"])
            data_years.append({"region": letter[s], "year": y, "exposures": len(L), "m5_median": round(m5[len(m5) // 2], 2) if m5 else None,
                               "nights": len({int(x["mjd_mid"] - 0.5) for x in L})})
    out = {"summary": [summ], "rounds": rounds, "regions": regions, "funnel": funnel, "events": events, "candidates": cands,
           "data_regions": data_regions, "data_years": data_years}
    out.update(run1())                                    # Run 1 = internal run r2 (the earlier run above is r1)
    os.makedirs(os.path.join(HERE, "progress", "out"), exist_ok=True)
    p = os.path.join(HERE, "progress", "out", "score.json"); json.dump(out, open(p, "w"), indent=1)
    print(p, summ); print("run 1:", out["run1_summary"][0])


if __name__ == "__main__":
    main()
