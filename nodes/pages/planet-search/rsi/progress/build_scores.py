"""Build the per-round score breakdown page (2026-11-05).

Reads each round's reward.json and grade/grade.json (sealed: used here only for counts and points, never for names or
field ids), its NOTES.md change line, and verdicts.json (Claude's verdict on each round, written by hand; rounds
without one show "no verdict yet"; run N's are in verdicts_rN.json). Writes one page per run: run 1 at
$PAGES_OUT/scores/index.html, run N at scores/rN/index.html (since 2026-11-09),
which $PAGE_PUBLISHER publishes on its next tick; build_page.py calls this after every tracker build. The ten sealed
fields are shown as Field A to J. Stops without writing if a key, sealed name, field id or path would reach the page.

Usage: python3 -I build_scores.py [--dump]     (--dump prints the table used to write verdicts.json)
"""
import importlib.util, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RSI = os.path.dirname(HERE)
PUB = os.path.expanduser("$PAGES_OUT")
# one page per run (2026-11-09 after run 2 started): run 1 keeps scores/, run N gets scores/rN/
MODEL_NAME = {"claude-opus-5-5": "Opus 5.5", "claude-fable-5-2": "Fable 5.2"}
RUN = GENS = None   # set per run by use_run()


def use_run(run):
    global RUN, GENS
    RUN = os.path.join(RSI, "runs", run)
    GENS = os.path.join(RUN, "gens")


def run_ids():
    base = os.path.join(RSI, "runs")
    return sorted(r for r in os.listdir(base) if os.path.exists(os.path.join(base, r, "config.json")) and os.path.isdir(os.path.join(base, r, "gens")))


def page_of(run):
    """(output file, path back to the tracker root, verdicts file) of one run's page."""
    if run == "r1":
        return os.path.join(PUB, "scores", "index.html"), "../", os.path.join(HERE, "verdicts.json")
    return os.path.join(PUB, "scores", run, "index.html"), "../../", os.path.join(HERE, f"verdicts_{run}.json")
WEIGHT = {1: 1, 2: 2, 3: 4}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_json(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def order(g):
    m = re.match(r"gen(\d+)(.*)", g)
    return (int(m.group(1)), m.group(2))


def collect():
    bt = load("build_traces", os.path.join(HERE, "build_traces.py"))
    pg = load("programs_git", os.path.join(RSI, "harness", "programs_git.py"))
    pg.GENS = GENS                       # its base_from_prompt reads this run's transcripts
    rl = load("rsi_loop", os.path.join(RSI, "harness", "rsi_loop.py"))
    clean = bt.Cleaner()
    sids = json.load(open(os.path.join(RUN, "config.json")))["eval"]
    label = {s: "ABCDEFGHIJ"[i] for i, s in enumerate(sorted(sids))}
    field_total, tier_total = {label[s]: 0 for s in sids}, {1: 0, 2: 0, 3: 0}
    for s in sids:
        for b in json.load(open(os.path.join(RSI, "private", "truth", s + ".json")))["bodies"]:
            if b.get("tier") in WEIGHT:
                tier_total[b["tier"]] += 1
                field_total[label[s]] += WEIGHT[b["tier"]]
    names = sorted((g for g in os.listdir(GENS) if os.path.isdir(os.path.join(GENS, g)) and re.fullmatch(r"gen\d{3}[a-h]?", g)), key=order)
    rewards = {}
    for g in names:
        rw = read_json(os.path.join(GENS, g, "reward.json"))
        if rw is not None and rw.get("reward") is not None:
            rewards[g] = rw["reward"]
    rows, best = [], -1.0
    for g in names:
        if g not in rewards:
            continue
        d = os.path.join(GENS, g)
        rw = read_json(os.path.join(d, "reward.json"))
        gr = read_json(os.path.join(d, "grade", "grade.json"))
        fam, base_line, chg, prose = rl.notes_head(os.path.join(d, "submission", "NOTES.md")) if g != "gen000" else ("starter", "", "the starter program", "")
        base = pg.base_from_prompt(g) if g != "gen000" else None
        if base is None and g != "gen000":
            base = pg.best_before(g, rewards)
        r = dict(g=g, reward=rewards[g], status=rw.get("status", ""), graded_on=rw.get("graded_on", ""),
                 base=base, base_reward=rewards.get(base), family=clean(fam or ""), change=clean((chg or prose or "")[:600]))
        if gr:
            hits = gr["hits"]
            r["tiers"] = {str(k): sum(1 for h in hits if h.get("tier") == k) for k in WEIGHT}
            fp = {label[s]: 0 for s in sids}
            for h in hits:
                if h.get("tier") in WEIGHT and h["sid"] in label:
                    fp[label[h["sid"]]] += WEIGHT[h["tier"]]
            r["fields"] = [fp[k] for k in sorted(fp)]
            r["confirmed"] = gr.get("confirmed_weight", 0)
            r["points"] = gr.get("hit_weight")
        r["record"] = rewards[g] > best + 1e-9
        best = max(best, rewards[g])
        r["best"] = best
        rows.append(r)
    return clean, rows, dict(fields=[field_total[k] for k in sorted(field_total)], tiers={str(k): v for k, v in tier_total.items()},
                             total=sum(field_total.values()))


def build(run, runs):
    use_run(run)
    out_path, root, vfile = page_of(run)
    clean, rows, totals = collect()
    if "--dump" in sys.argv:
        for r in rows:
            t = r.get("tiers", {})
            print(f"{run} {r['g']} {r['reward']:.4f} base {r['base']} ({r['base_reward']}) t{t.get('1', '-')}/{t.get('2', '-')}/{t.get('3', '-')} "
                  f"{'REC ' if r['record'] else ''}{r['status'] if r['status'] != 'ok' else ''} | {r['change'][:330]}")
        return
    if not rows:
        return
    vj = read_json(vfile) or {}
    if not vj.get("kinds"):     # the kinds of change are the same for every run; run 1's file defines them
        vj["kinds"] = (read_json(os.path.join(HERE, "verdicts.json")) or {}).get("kinds", {})
    verdicts = vj.get("rounds", {})
    for r in rows:
        v = verdicts.get(r["g"])
        r["kind"], r["verdict"] = (v["kind"], clean(v["text"])) if v else ("", "")
    # stamped with the newest input, not the clock, so an unchanged page is not rewritten (and not republished) every tick
    newest = max([os.path.getmtime(os.path.join(GENS, r["g"], "reward.json")) for r in rows] +
                 [os.path.getmtime(f) for f in (vfile, os.path.join(HERE, "scores.html")) if os.path.exists(f)])
    cfg = json.load(open(os.path.join(RUN, "config.json")))
    me = os.path.basename(RUN)
    data = dict(rows=rows, totals=totals, overall=[[clean(a), clean(b)] for a, b in vj.get("overall", [])], kinds=vj.get("kinds", {}),
                built=time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(newest)),
                run=me, label="Run " + me.lstrip("r"), model=MODEL_NAME.get(cfg.get("model", "claude-opus-5-5"), cfg.get("model", "")),
                root=root, traces=me == "r1",
                runs=[dict(run=r, label="Run " + r.lstrip("r"), href=root + ("scores/" if r == "r1" else f"scores/{r}/")) for r in runs])
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = open(os.path.join(HERE, "scores.html")).read().replace("/*SCORES_DATA*/", blob)
    bad = clean.check(html)
    if bad:
        raise SystemExit(f"stopped: {bad} would reach the {me} scores page; nothing written")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and open(out_path).read() == html:
        return
    tmp = out_path + ".tmp"
    open(tmp, "w").write(html)
    os.replace(tmp, out_path)
    print("wrote", out_path, len(html), "rounds", len(rows))


def main():
    runs = run_ids()
    for run in runs:
        build(run, runs)


if __name__ == "__main__":
    main()
