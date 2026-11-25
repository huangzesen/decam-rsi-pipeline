"""Write the RSI progress dashboard's data files (out/*.json) from the run's private record.

Friend-safe by construction: no body names, scenario ids, paths or keys; hidden fields are labelled F1..Fn.
Usage: python -I make_progress.py [RUN]
"""
import datetime as dt, glob, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RSI = os.path.dirname(HERE)
# first plain argument; skips build_page.py's "--out PATH" when imported from it
RUN = ([a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and not sys.argv[i].startswith("--")] or ["r1"])[0]
PRIV = os.path.join(RSI, "runs", RUN)
OUT = os.path.join(HERE, "out")
PLAN = []   # open-ended until the deadline: no placeholders for rounds not yet started
TIER = {1: ("Seen in single exposures", 1), 2: ("Only in a one-night stack", 2), 3: ("Only when nights are combined", 4),
        0: ("Not findable in these images", 0), None: ("Orbit too uncertain to grade", 0)}


def iso(t):
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def tool_times(path):
    """(first timestamp, [epoch of each tool call]) of one transcript."""
    first = last = None; calls = []
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        ts = d.get("timestamp") if isinstance(d, dict) else None
        if ts:
            last = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            first = first or last
        if isinstance(d, dict) and d.get("type") == "assistant":
            for c in d.get("message", {}).get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_use" and last:
                    calls.append(last)
    return first, calls


def transcript_stats(path, sub_paths=()):
    """(start epoch, tool calls, [(minute, cumulative calls)]) of the session, its subagents' calls included
    (from gen103 most of the work runs in subagents, so the main transcript alone shows a flat line)."""
    t0, calls = tool_times(path)
    for sp in sub_paths:
        calls += tool_times(sp)[1]
    per = {}
    for n, t in enumerate(sorted(c for c in calls if t0 and c >= t0), 1):
        per[int((t - t0) // 60)] = n
    return t0, len(calls), sorted(per.items())


def found(pg):
    """Counts of hidden bodies found by difficulty (no names), from the private grade."""
    f = os.path.join(pg, "grade", "grade.json")
    if not os.path.exists(f):
        return dict(found1=None, found2=None, found4=None, found_new=None)
    g = json.load(open(f))
    c = {t: sum(1 for h in g["hits"] if h["tier"] == t) for t in (1, 2, 3)}
    return dict(found1=c[1], found2=c[2], found4=c[3], found_new=len(g.get("confirmed", [])))


def notes_summary(path, limit=420):
    """First prose of a submitted NOTES.md (written after the session), for programs without a hand-written note."""
    if not os.path.exists(path):
        return ""
    parts = []
    for line in open(path, errors="replace"):
        t = re.sub(r"[*`_]", "", line.strip()).lstrip("-0123456789. ").strip()
        t = re.sub(r"\bs[0-9a-f]{8}\b", "a practice field", t)     # no field ids on the page
        if not t or line.lstrip().startswith("#"):
            continue
        parts.append(t)
        if sum(map(len, parts)) > limit:
            break
    s = " ".join(parts)
    return ("From its notes: " + (s[:limit].rsplit(" ", 1)[0] + "…" if len(s) > limit else s)) if s else ""


MODEL_NAME = {"claude-opus-5-5": "Opus 5.5", "claude-fable-5-2": "Fable 5.2"}
# How each run was set up, in words for the page (2026-11-09, when run 2 started)
RUN_ABOUT = {
    "r1": "Opus 5.5 at effort xhigh. Started from a simple starter program we wrote. Generation 1 was one 3-hour session; "
          "generations 2-14 had three agents each in 1-hour sessions; from generation 15 one agent per generation with "
          "subagent workflows (ultracode); 2-hour sessions from generation 103. Stopped on Nov 9 after generation 151.",
    "r2": "Fable 5.2 at effort xhigh, no subagent workflows. Started from nothing: the first agent got an empty folder and "
          "wrote the finder from scratch. One agent per generation, 2-hour sessions, up to 50 generations (set to 20 on Nov 9 "
          "after generation 4, raised to 50 on Nov 10 during generation 7). From generation 14 the agents write and test on a cloud machine without a GPU instead of the Mac; grading is unchanged. Same task, hidden fields and grading as run 1, but its agents cannot see anything from run 1.",
}


def run_ids():
    """Every run with a config and a gens folder (r1, r2, ...)."""
    base = os.path.join(RSI, "runs")
    return sorted(r for r in os.listdir(base)
                  if os.path.exists(os.path.join(base, r, "config.json")) and os.path.isdir(os.path.join(base, r, "gens")))


def run_gens(run, approach):
    """(generations, activity) of one run; every row carries its run id."""
    priv = os.path.join(RSI, "runs", run)
    cfg = json.load(open(os.path.join(priv, "config.json")))
    gens, activity = [], []
    for g in sorted(os.listdir(os.path.join(priv, "gens"))):
        pg = os.path.join(priv, "gens", g)
        if not os.path.isdir(pg):  # gens/ also holds grading logs and job outputs; only folders are generations
            continue
        rw = json.load(open(os.path.join(pg, "reward.json"))) if os.path.exists(os.path.join(pg, "reward.json")) else None
        ss = json.load(open(os.path.join(pg, "session.json"))) if os.path.exists(os.path.join(pg, "session.json")) else None
        tr = os.path.join(pg, "transcript.jsonl")
        subs = sorted(glob.glob(os.path.join(pg, "subagents", "**", "*.jsonl"), recursive=True))
        t0, calls, per = transcript_stats(tr, subs) if os.path.exists(tr) else (None, None, [])
        model = (ss or {}).get("model") or cfg.get("model", "claude-opus-5-5")
        who = "Starter (written by us)" if g == "gen000" else MODEL_NAME.get(model, model) + " agent"
        if os.path.exists(os.path.join(pg, "skipped.txt")):
            state = "skipped"
        elif rw and rw.get("reward") is not None:
            state = "graded"
        elif ss or g == "gen000":
            state = "grading"
        else:
            state = "writing"
        if state in ("writing", "grading") and os.path.exists(os.path.join(priv, "STOPPED")):
            state = "unfinished"        # the run was stopped before this round was graded
        mins = round(ss["seconds"] / 60) if ss else (round((dt.datetime.now().timestamp() - t0) / 60) if t0 else None)
        gens.append(dict(run=run, name=g, gen=int(g[3:6]), agent=who, state=state,
                         reward=rw.get("reward") if rw else None, status=(rw or {}).get("status", "") if state == "graded" else "",
                         finder_s=(rw or {}).get("finder_seconds_total") if state == "graded" else None,
                         started=iso(t0) if t0 else None, session_min=mins, tool_calls=calls,
                         approach=(approach.get(g) if run == "r1" else None) or notes_summary(os.path.join(pg, "submission", "NOTES.md")),
                         graded_at=iso(os.path.getmtime(os.path.join(pg, "reward.json"))) if state == "graded" else None,
                         graded_on=(rw or {}).get("graded_on") if state == "graded" else None,
                         hours=(ss or {}).get("hours"),
                         **found(pg)))
        activity += [dict(run=run, name=g, minute=m, calls=c) for m, c in per]
    return gens, activity


COST_CACHE = os.path.join(HERE, "out", "compare_cache.json")
WEIGHT = {1: 1, 2: 2, 3: 4}


def session_cost(path, cache):
    """(list-price dollars, output tokens) of one finished session, from its transcript's last result line; cached by size and mtime."""
    st = os.stat(path); key = f"{st.st_size}:{int(st.st_mtime)}"
    if cache.get(path, {}).get("key") == key:
        return cache[path]["cost"], cache[path]["out"]
    last = None
    for line in open(path, errors="replace"):
        if '"result"' in line:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("type") == "result":
                last = d
    cost = (last or {}).get("total_cost_usd") or 0.0
    out = sum(u.get("outputTokens", 0) for u in ((last or {}).get("modelUsage") or {}).values())
    cache[path] = dict(key=key, cost=cost, out=out)
    return cost, out


def compare(runs, label_of):
    """The runs' best programs side by side, and each run's best-so-far against rounds, agent hours and list price (user,
    2026-11-09: "i want a comparison of break down of best of both runs", "rolling comparison"). Counts and points only:
    bodies are matched across runs by their sealed designation here, and only the counts leave this function."""
    try:
        cache = json.load(open(COST_CACHE))
    except (OSError, ValueError):
        cache = {}
    per, hits_of, ever = {}, {}, {}
    for run in runs:
        gd = os.path.join(RSI, "runs", run, "gens")
        rows = []
        for g in sorted(os.listdir(gd)):
            pg = os.path.join(gd, g)
            if not (os.path.isdir(pg) and re.fullmatch(r"gen\d{3}[a-z]?", g)):
                continue
            rw = json.load(open(os.path.join(pg, "reward.json"))) if os.path.exists(os.path.join(pg, "reward.json")) else {}
            ss = json.load(open(os.path.join(pg, "session.json"))) if os.path.exists(os.path.join(pg, "session.json")) else None
            tr = os.path.join(pg, "transcript.jsonl")
            cost, out = session_cost(tr, cache) if ss and os.path.exists(tr) else (0.0, 0)
            gr = os.path.join(pg, "grade", "grade.json")
            hits = [h for h in json.load(open(gr))["hits"] if h.get("tier") in WEIGHT] if os.path.exists(gr) else None
            lines = 0
            for p in glob.glob(os.path.join(pg, "submission", "**", "*.py"), recursive=True):
                if not os.path.islink(p):
                    lines += sum(1 for _ in open(p, errors="replace"))
            rows.append(dict(g=g, k=int(g[3:6]), reward=rw.get("reward"), finder_s=rw.get("finder_seconds_total"),
                             sec=(ss or {}).get("seconds", 0), cost=cost, out=out, hits=hits, lines=lines))
        graded = [r for r in rows if r["reward"] is not None and r["hits"] is not None]
        if not graded:
            continue
        top = max(r["reward"] for r in graded)
        b = min((r for r in graded if r["reward"] >= top - 1e-9), key=lambda r: (r["k"], r["g"]))   # the round that first reached it
        before = [r for r in rows if r["k"] <= b["k"]]
        tiers = {str(t): sum(1 for h in b["hits"] if h["tier"] == t) for t in WEIGHT}
        fpts = {}
        for h in b["hits"]:
            if h["sid"] in label_of:
                fpts[label_of[h["sid"]]] = fpts.get(label_of[h["sid"]], 0) + WEIGHT[h["tier"]]
        curve, bs = [], None
        for k in sorted({r["k"] for r in rows}):
            done = [r["reward"] for r in graded if r["k"] <= k]
            if not done or k == 0 or k > max(r["k"] for r in graded):     # a round still writing or grading adds no point yet
                continue
            upto = [r for r in rows if r["k"] <= k]
            curve.append([k, round(max(done), 4), round(sum(r["sec"] for r in upto) / 3600, 2), round(sum(r["cost"] for r in upto), 2)])
        per[run] = dict(best_gen=b["g"], best_k=b["k"], reward=top, points=sum(WEIGHT[h["tier"]] for h in b["hits"]), tiers=tiers,
                        fields=fpts, hours_to_best=round(sum(r["sec"] for r in before) / 3600, 1),
                        cost_to_best=round(sum(r["cost"] for r in before)), session_min=round(b["sec"] / 60),
                        cost_best=round(b["cost"], 2), out_tokens_best=b["out"], finder_s=b["finder_s"], lines=b["lines"],
                        graded=len(graded), hours=round(sum(r["sec"] for r in rows) / 3600, 1), cost=round(sum(r["cost"] for r in rows)),
                        curve=curve)
        hits_of[run] = {(h["sid"], h["pdes"]): h["tier"] for h in b["hits"]}
        ever[run] = {(h["sid"], h["pdes"]): h["tier"] for r in graded for h in r["hits"]}
    try:
        json.dump(cache, open(COST_CACHE, "w"))
    except OSError:
        pass
    count = lambda keys, tiers: {str(t): sum(1 for k in keys if tiers[k] == t) for t in WEIGHT}
    pair = None
    if len(hits_of) >= 2:
        a, b = sorted(hits_of)[:2]
        A, B = hits_of[a], hits_of[b]; tiers = {**A, **B}
        EA, EB = ever[a], ever[b]; et = {**EA, **EB}
        pair = dict(a=a, b=b, both=count(A.keys() & B.keys(), tiers), only_a=count(A.keys() - B.keys(), tiers),
                    only_b=count(B.keys() - A.keys(), tiers), union_points=sum(WEIGHT[t] for t in tiers.values()),
                    ever_both=count(EA.keys() & EB.keys(), et), ever_only_a=count(EA.keys() - EB.keys(), et),
                    ever_only_b=count(EB.keys() - EA.keys(), et),
                    best_b_never_a=count(B.keys() - EA.keys(), tiers), best_a_never_b=count(A.keys() - EB.keys(), tiers))
    return [dict(runs=per, pair=pair)]


def spotlight(runs, label_of):
    """The faintest body gained by the newest record of any run, exposure by exposure (2026-11-09, after r2 gen005): its S/N as the program read it along its own orbit, and as forced photometry
    reads it at the body's catalogued place. Counts, magnitudes and S/N only: no names, field ids or dates leave here."""
    recs = []
    for run in runs:
        gd = os.path.join(RSI, "runs", run, "gens")
        top, prev, ever = -1.0, {}, set()
        for g in sorted(os.listdir(gd)):
            gr, rw = os.path.join(gd, g, "grade", "grade.json"), os.path.join(gd, g, "reward.json")
            if not (re.fullmatch(r"gen\d{3}[a-z]?", g) and os.path.exists(gr) and os.path.exists(rw)):
                continue
            r = json.load(open(rw)).get("reward")
            hits = {(h["sid"], h["pdes"]): h for h in json.load(open(gr))["hits"] if h.get("tier") in WEIGHT}
            if r is not None and r > top + 1e-9:
                gained = [h for k, h in hits.items() if k not in prev]
                if gained:
                    recs.append(dict(run=run, g=g, at=os.path.getmtime(rw), gained=gained, before=set(ever)))
                top, prev = r, hits
            ever |= set(hits)
    if not recs:
        return []
    rec = max(recs, key=lambda x: x["at"])
    run, g = rec["run"], rec["g"]
    others = {}
    for o in runs:
        if o == run:
            continue
        seen, n = set(), 0
        for gr in glob.glob(os.path.join(RSI, "runs", o, "gens", "gen*", "grade", "grade.json")):
            seen |= {(h["sid"], h["pdes"]) for h in json.load(open(gr))["hits"]}
            n += 1
        others[o] = (seen, n)
    rows = []
    for h in rec["gained"]:
        body = next((b for b in json.load(open(os.path.join(RSI, "private", "truth", h["sid"] + ".json")))["bodies"]
                     if b["pdes"] == h["pdes"]), None)
        vis = [x for x in (body or {}).get("recs", []) if not x.get("held") and x.get("snr") is not None]
        if not vis or h["sid"] not in label_of:
            continue
        rows.append((max(x["snr"] for x in vis), h, vis))
    if not rows:
        return []
    peak, h, vis = min(rows, key=lambda x: x[0])
    cand, allp = None, []
    for f in glob.glob(os.path.join(RSI, "runs", run, "gens", g, "*", "*", "candidates.json")):
        cs = json.load(open(f))["candidates"]
        allp += [c["p"] for c in cs]
        if os.path.basename(os.path.dirname(f)) == h["sid"]:
            cand = next((c for c in cs if abs(c["p"] - h["p"]) < 5e-6), cand)
    allp.sort(reverse=True)
    K = 250
    nights = sorted({x["night"] for x in vis})
    prog = (cand or {}).get("snr") or []
    if len(prog) != len(vis):
        prog = [None] * len(vis)      # the program's per-exposure list doesn't line up with the visible exposures
    exp = [[nights.index(x["night"]) + 1, None if p is None else round(p, 1), round(x["snr"], 1)] for x, p in zip(vis, prog)]
    stack = lambda v: round(sum(v) / math.sqrt(len(v)), 1) if v else None
    vmag = sorted(x["V"] for x in vis if x.get("V") is not None)
    return [dict(run=run, gen=g, field=label_of[h["sid"]], tier=h["tier"], weight=WEIGHT[h["tier"]], p=round(h["p"], 2),
                 mag=round(vmag[len(vmag) // 2], 1) if vmag else None, r_au=round(vis[0]["r"], 1) if vis[0].get("r") else None,
                 exp=exp, peak_truth=round(peak, 1), peak_prog=max((p for p in prog if p is not None), default=None),
                 stack_prog=stack([p for p in prog if p is not None]), stack_truth=stack([x["snr"] for x in vis]),
                 rank=allp.index(h["p"]) + 1 if h["p"] in allp else None, n_cand=len(allp), kept=K,
                 cut_p=round(allp[K - 1], 2) if len(allp) >= K else None,
                 new_in_run=(h["sid"], h["pdes"]) not in rec["before"],
                 others_never=[o for o, (seen, n) in others.items() if (h["sid"], h["pdes"]) not in seen])]


def run_meta(run, gens):
    """One row per run for the page: model, setup in words, state, best."""
    priv = os.path.join(RSI, "runs", run)
    cfg = json.load(open(os.path.join(priv, "config.json")))
    mine = [g for g in gens if g["run"] == run]
    graded = [g for g in mine if g["state"] == "graded"]
    stopped = os.path.join(priv, "STOPPED")
    live = [g for g in mine if g["state"] in ("writing", "grading")]
    last = max((g["gen"] for g in mine), default=0)
    model = cfg.get("model", "claude-opus-5-5")
    return dict(run=run, label="Run " + run.lstrip("r"), model=MODEL_NAME.get(model, model), about=RUN_ABOUT.get(run, ""),
                start="our starter program" if cfg.get("start") != "empty" else "an empty folder",
                state="stopped" if os.path.exists(stopped) else ("running" if live else
                      "finished" if cfg.get("last_gen") and any(g["gen"] >= cfg["last_gen"] for g in graded) else "between rounds"),
                last_planned=cfg.get("last_gen"),
                made=dt.datetime.strptime(cfg["made"], "%Y-%m-%d %H:%M").astimezone().isoformat(timespec="seconds"),
                stopped_at=iso(os.path.getmtime(stopped)) if os.path.exists(stopped) else None,
                hours=next((g["hours"] for g in sorted(mine, key=lambda g: g["name"], reverse=True) if g.get("hours")), cfg.get("hours")),
                agents=sum(1 for g in mine if g["gen"] == last) or 1, last_gen=last,
                graded=len(graded), best=max((g["reward"] for g in graded), default=None))


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(os.path.join(PRIV, "config.json")))
    approach = json.load(open(os.path.join(HERE, "approach.json")))
    gens, activity = [], []
    for r in run_ids():
        gs, act = run_gens(r, approach)
        gens += gs; activity += act
    runs = [run_meta(r, gens) for r in run_ids()]
    cur = ([m for m in runs if m["state"] != "stopped"] or runs)[-1]     # the run the page leads with
    # hidden fields: tier make-up, F1.. by weight
    fields, tiers = [], {}
    for sid in cfg["eval"]:
        t = json.load(open(os.path.join(RSI, "private", "truth", sid + ".json")))
        m = json.load(open(os.path.join(RSI, "scen_sealed", sid, "meta.json")))
        cnt = {k: 0 for k in TIER}
        for b in t["bodies"]:
            cnt[b.get("tier")] += 1
        for k in TIER:
            tiers[k] = tiers.get(k, 0) + cnt[k]
        fields.append(dict(w1=cnt[1], w2=cnt[2] * 2, w4=cnt[3] * 4, n1=cnt[1], n2=cnt[2], n4=cnt[3],
                           exposures=m["n_exposures"], nights=m["n_nights"],
                           span_days=round(m["last_mjd"] - m["first_mjd"], 1), bands=m["bands"], _sid=sid))
    fields.sort(key=lambda f: -(f["w1"] + f["w2"] + f["w4"]))
    label_of = {}
    for i, f in enumerate(fields):
        f["field"] = f"F{i + 1}"
        f["weight"] = f["w1"] + f["w2"] + f["w4"]
        label_of[f.pop("_sid")] = f["field"]        # the field id stays here; the page gets F1..Fn only
    truth = [dict(tier=lab, bodies=tiers[k], weight_each=w, weight=tiers[k] * w, findable=w > 0)
             for k, (lab, w) in TIER.items()]
    # downloads
    downloads = []
    for name, log in (("Batch 1", "fetch.log"), ("Batch 2", "fetch_add.log")):
        lines = open(os.path.join(RSI, log)).read().splitlines()
        prog = [re.search(r"^(\d\d):(\d\d):(\d\d) (\d+)/(\d+) \S+ \S+ \| ([\d.]+)/(\d+) GB, \d+ MB/s avg, failed (\d+)", l)
                for l in lines]
        prog = [m for m in prog if m]
        m = prog[-1]
        done = lines[-1].startswith("DONE")
        # recent speed: GB over the last ~10 min of "ok" lines (a restart's skipped files would inflate the average)
        sec = lambda x: int(x[1]) * 3600 + int(x[2]) * 60 + int(x[3])
        rate = None
        for x in reversed(prog[:-1]):
            dt_s = (sec(m) - sec(x)) % 86400
            if dt_s >= 600 and float(m[6]) >= float(x[6]):
                rate = round((float(m[6]) - float(x[6])) * 1000 / dt_s)
                break
        downloads.append(dict(batch=name, files_done=int(m[4]), files_total=int(m[5]), gb_done=float(m[6]),
                              gb_total=float(m[7]), mb_s=None if done else rate, failed=int(m[8]), finished=done))
    nwin = sum(len(json.load(open(os.path.join(RSI, f)))["windows"]) for f in ("files.json", "files_add.json"))
    # match_arcsec, min_nights: as in harness/grade.py
    setup = [dict(fields=len(cfg["eval"]), slots_per_field=cfg["k_per_scen"], slots=cfg["k_per_scen"] * len(cfg["eval"]),
                  cap_s=round(cfg["cap_s"]), practice=len(os.listdir(os.path.join(RSI, "scen_dev"))),
                  sealed=len(os.listdir(os.path.join(RSI, "scen_sealed"))),
                  truth_built=len(glob.glob(os.path.join(RSI, "private", "truth", "*.json"))),
                  windows=nwin, agents_per_gen=cur["agents"], session_h=cur["hours"] or 2, match_arcsec=2, min_nights=3,
                  updated=iso(dt.datetime.now().timestamp()))]
    events = json.load(open(os.path.join(HERE, "events.json")))
    for m in runs:
        mine = [g for g in gens if g["run"] == m["run"]]
        pre = m["label"] + ": " if len(runs) > 1 else ""
        for k in sorted({g["gen"] for g in mine if g["gen"] >= 1}):
            gs = [g for g in mine if g["gen"] == k]
            st = [g["started"] for g in gs if g["started"]]
            if k >= 2 and st and len(gs) > 1:
                events.append(dict(at=min(st), what=f"{pre}generation {k}: {len(gs)} agents started writing, independently."))
            elif st and (k >= 2 or m["run"] != "r1"):
                events.append(dict(at=min(st), what=f"{pre}generation {k} started writing."))
            if all(g["state"] == "graded" for g in gs):
                r = ", ".join(f"{g['name'][-1]} {g['reward']:.2f}" for g in gs) if len(gs) > 1 else f"{gs[0]['reward']:.2f}"
                events.append(dict(at=max(g["graded_at"] for g in gs),
                                   what=f"{pre}generation {k} graded: reward {r}." if len(gs) == 1 else f"{pre}generation {k} graded: rewards {r}."))
    events.sort(key=lambda e: e["at"])
    for name, rows in (("generations", gens), ("activity", activity), ("fields", fields), ("truth", truth),
                       ("downloads", downloads), ("setup", setup), ("events", events), ("runs", runs),
                       ("compare", compare(run_ids(), label_of)), ("spotlight", spotlight(run_ids(), label_of))):
        json.dump(rows, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
        print(name, len(rows))


if __name__ == "__main__":
    main()
