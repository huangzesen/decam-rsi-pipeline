#!/usr/bin/env python3
"""Every earlier engineering result of this task, laid out for run r2's starting history.

User, 2026-11-14, right after the go for r2: "The agent should be given all previous engineering results in the zeroth gen".
Before this, the first agent of r2 got the earlier run's programs with notes and one number each. This builds the rest:

  runs/r2/history_extra/                    overlaid on history/ by climb.handoff in every round
    prior_r1/RESULTS.md, BODIES.md, bodies.json
    prior_r1/genNNN/{out/<sid>/candidates.json, log/<sid>.log, timing.json, grade.json, bodies.md}
    engineering/README.md                   index and measured facts
    engineering/gen019_all_regions/         gen019's outputs on every open region, scored, per body
    engineering/leads/{README.md, leads.json}   every unmatched candidate that was checked, with the verdicts
  runs/r2/history_extra_note.md             lines appended to the agent's session note

Outputs of the earlier run are scored again with this run's scoring code and label files (harness/grade.py =
/app/tools/score.py, private/truth_all23 = /app/open/labels), so body indices match what the agent has. The earlier
run's own grade.json files are left out: their body indices refer to older label files. Left out on purpose:
correspondence, filings, predicted positions (runs/*/LEADS.md, arc/, observer/): not results about the method.

Needs runs/r2/eng/loss/bodies.json for all 23 regions (python -I harness/body_props.py r2/eng --add).
Usage: ~/anaconda3/bin/python -I harness/eng_history.py
"""
import json, os, re, shutil, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "harness"))
import climb                                              # noqa: E402

R2 = os.path.join(HERE, "runs", "r2"); ENG = os.path.join(R2, "eng")
OUT = os.path.join(R2, "history_extra"); TRUTH = os.path.join(HERE, "private", "truth_all23")
R1S = ["scen13", "scen10", "scen07"]             # the earlier run's regions: REG_A, REG_B, REG_C
PY = sys.executable
NOVELTY = os.path.expanduser("$ALE_HOME/planet-search/rsi/candidates/novelty_queue/RESULTS.md")

J = lambda p: json.load(open(p))
REG = J(os.path.join(ENG, "regions.json"))                # the agent's /app/open/labels/regions.json
OPEN = list(J(os.path.join(ENG, "config.json"))["sids"])
assert set(OPEN) == set(REG) and len(OPEN) == 23
GRADED = [s for s in OPEN if REG[s]["graded"]]
RID = {"scen13": "REG_A", "scen10": "REG_B", "scen07": "REG_C"}
NEXP = {"scen13": 729, "scen10": 682, "scen07": 787}
for r in J(os.path.join(HERE, "harness", "search38", "regions38.json")):
    RID[r["sid"]] = r["rid"]; NEXP[r["sid"]] = r["n_exp"]
P = J(os.path.join(ENG, "loss", "bodies.json"))
missing = [s for s in OPEN if REG[s]["findable"] and len(P.get(s, {})) != REG[s]["findable"]]
assert not missing, f"body properties missing for {missing}: run body_props.py r2/eng --add first"


def write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(text if text.endswith("\n") else text + "\n")


def score(out, grade, sids, timing):
    os.makedirs(grade, exist_ok=True)
    r = subprocess.run([PY, "-I", os.path.join(HERE, "harness", "grade.py"), out, TRUTH, grade, "--sids", ",".join(sids),
                        "--timing", timing], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    os.remove(os.path.join(grade, "public.json"))
    return J(os.path.join(grade, "grade.json"))


def fake_run(name, sids):
    """A folder that climb.body_feedback reads as a run: config.json (sids) and loss/bodies.json."""
    d = os.path.join(ENG, name); os.makedirs(os.path.join(d, "loss"), exist_ok=True)
    json.dump(dict(sids=sids), open(os.path.join(d, "config.json"), "w"))
    json.dump({s: P.get(s, {}) for s in sids}, open(os.path.join(d, "loss", "bodies.json"), "w"))
    return d


def reword(path, what):
    t = open(path).read()
    t = t.replace("of the graded regions", what).replace("history/BODIES.md", "BODIES.md")
    open(path, "w").write(t)


shutil.rmtree(OUT, ignore_errors=True)

# ---------------------------------------------------------------------------------------------- 1. the earlier run
st = J(os.path.join(HERE, "runs", "r1", "state.json"))["gens"]
gens = {k: v for k, v in st.items() if v.get("phase") == "done" and v.get("status") == "valid" and v.get("reward") is not None
        and v.get("submission") and os.path.isdir(v["submission"])}       # the same filter as climb.prior_history
PR = os.path.join(OUT, "prior_r1"); done = {}; rows = []
for k, v in sorted(gens.items()):
    gd = v.get("grade_dir") or os.path.join(HERE, "runs", "r1", "gens", k, "grade"); runs = gd + "_runs"
    d = os.path.join(PR, k)
    for s in R1S:
        os.makedirs(os.path.join(d, "out", s)); os.makedirs(os.path.join(d, "log"), exist_ok=True)
        shutil.copy(os.path.join(runs, "out", s, "candidates.json"), os.path.join(d, "out", s))
        lg = os.path.join(runs, "log", s + ".log")
        if os.path.isfile(lg):
            shutil.copy(lg, os.path.join(d, "log"))
    shutil.copy(os.path.join(runs, "timing.json"), d)
    g = score(os.path.join(d, "out"), d, R1S, os.path.join(d, "timing.json"))
    done[k] = dict(phase="done", status="valid", grade_dir=d, reward=g["reward"], gpu_hours=g["gpu_hours"])
    ps = g["per_scenario"]
    rows.append((k, v["reward"], g["reward"], g["gpu_hours"], [f"{ps[s]['known_found']}/{ps[s]['known_findable']}" for s in R1S],
                 [round(ps[s]["seconds"]) for s in R1S], g["candidates"], sum(ps[s]["leads"] for s in R1S),
                 climb.note_head(v["submission"]).get("Method", "")[:160]))
line = climb.body_feedback(fake_run("r1x", R1S), done, PR)
where = "of regions REG_A, REG_B and REG_C (the earlier run's three; REG_A and REG_B are also graded in this run)"
for f in [os.path.join(PR, "BODIES.md")] + [os.path.join(PR, k, "bodies.md") for k in gens]:
    reword(f, where)
L = [f"# The earlier run's {len(rows)} programs: measured results", "",
     "Each program ran on regions REG_A (`scen13`), REG_B (`scen10`) and REG_C (`scen07`) with a budget of 3,600 s per",
     "region, one A100 per region (the GPU this run is graded on). REG_A and REG_B are graded in this run too, at 1,800 s.", "",
     "- `recall then`: the reward that run recorded (known bodies with H <= 8 only, older label files).",
     "- `recall now`: the same outputs scored again with this run's scoring code and label files (`/app/tools/score.py`,",
     "  `/app/open/labels`: fainter known bodies included), the three regions ranked together. Not this run's recall: other",
     "  regions and twice the time. Use it to compare those programs with each other.",
     "- `found`: bodies found / findable per region in the new scoring. `seconds`: run time per region. `unmatched`: candidates",
     "  in the scored slots that match no labelled body.", "",
     "Per program, `genNNN/` holds (next to the code and notes): `out/<region>/candidates.json` (what it returned),",
     "`log/<region>.log` (what it printed while it ran on the GPU: stage timings, counts), `timing.json`, `grade.json` (the new",
     "scoring in full) and `bodies.md` (its outcome for every body). `BODIES.md` and `bodies.json` here hold one row per body",
     "over all these programs: 69 bodies, which of the 37 programs found each.", "",
     "| program | recall then | recall now | GPU-hours | found REG_A | REG_B | REG_C | seconds REG_A | REG_B | REG_C | candidates | unmatched | method (from its notes) |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, r0, r1, gh, fd, sec, nc, nl, m in sorted(rows, key=lambda x: (-x[2], x[3], x[0])):
    L.append(f"| {k} | {r0:.4f} | {r1:.4f} | {gh} | {fd[0]} | {fd[1]} | {fd[2]} | {sec[0]} | {sec[1]} | {sec[2]} | {nc} | {nl} | {m.replace('|', '/')} |")
write(os.path.join(PR, "RESULTS.md"), "\n".join(L))
bj = J(os.path.join(PR, "bodies.json")); n69 = len(bj["bodies"])
never = sum(1 for b in bj["bodies"].values() if b["found_by"] == 0); allf = sum(1 for b in bj["bodies"].values() if b["found_by"] == len(gens))
print(f"prior_r1: {len(gens)} programs, {n69} bodies, {never} found by none, {allf} found by all")

# ------------------------------------------------------------------------- 2. gen019 on every open region
EG = os.path.join(OUT, "engineering"); GA = os.path.join(EG, "gen019_all_regions")
g19 = gens["gen019"]; r19 = (g19.get("grade_dir") or os.path.join(HERE, "runs", "r1", "gens", "gen019", "grade")) + "_runs"
S38 = {}                                                   # sid -> (group, budget)
for grp, sub in (("v28", ["v28_normal"]), ("n10", ["n10_big", "n10_normal"])):
    for sb in sub:
        for s in os.listdir(os.path.join(HERE, "runs", "search38", sb, "gen019", "out")):
            S38[s] = (grp, sb, 7200 if sb == "n10_big" else 3600)
timing = {}; t19 = J(os.path.join(r19, "timing.json")); budget = {}
for s in OPEN:
    os.makedirs(os.path.join(GA, "out", s)); os.makedirs(os.path.join(GA, "log"), exist_ok=True)
    if s in R1S:
        shutil.copy(os.path.join(r19, "out", s, "candidates.json"), os.path.join(GA, "out", s))
        shutil.copy(os.path.join(r19, "log", s + ".log"), os.path.join(GA, "log")); timing[s] = t19[s]; budget[s] = 3600
    else:
        grp, sb, budget[s] = S38[s]; gd = os.path.join(HERE, "runs", "s38", "gens", grp + "-gen019")
        shutil.copy(os.path.join(gd, "out", s, "candidates.json"), os.path.join(GA, "out", s))
        lg = os.path.join(HERE, "runs", "search38", sb, "gen019", "log", s + ".log")
        if os.path.isfile(lg):
            shutil.copy(lg, os.path.join(GA, "log"))
        timing[s] = J(os.path.join(gd, "timing.json"))[s]
json.dump(timing, open(os.path.join(GA, "timing.json"), "w"), indent=1)
ga = score(os.path.join(GA, "out"), GA, OPEN, os.path.join(GA, "timing.json"))
climb.body_feedback(fake_run("allx", OPEN), dict(gen019=dict(phase="done", status="valid", grade_dir=GA, reward=ga["reward"],
                                                              gpu_hours=ga["gpu_hours"])), GA)
reword(os.path.join(GA, "BODIES.md"), "of all 23 regions in `/app/open`")
ps = ga["per_scenario"]; nf = sum(ps[s]["known_found"] for s in OPEN); nk = sum(ps[s]["known_findable"] for s in OPEN)
T = ["| region | folder | graded here | exposures | detections (M) | budget (s) | seconds | findable | found | unmatched in slots |", "|---|---|---|---|---|---|---|---|---|---|"]
for s in OPEN:
    T.append(f"| {RID[s]} | `{s}` | {'yes' if REG[s]['graded'] else ''} | {NEXP[s]} | "
             f"{REG[s]['detections'] / 1e6:.1f} | {budget[s]} | {round(ps[s]['seconds'])} | {ps[s]['known_findable']} | {ps[s]['known_found']} | {ps[s]['leads']} |")
rest = []                                                  # search regions the agent does not have: counts only
for grp in ("v28", "n10"):
    gg = J(os.path.join(HERE, "runs", "s38", "gens", grp + "-gen019", "grade", "grade.json"))
    for s, x in gg["per_scenario"].items():
        if s not in OPEN:
            rest.append((RID[s], NEXP[s], S38[s][2], round(x["seconds"]), x["known_findable"], x["known_found"], x["leads"]))
write(os.path.join(GA, "README.md"), "\n".join([
    "# gen019 (this run's gen000) on every region in `/app/open`", "",
    "The program was run once on each of the 23 regions, on one A100 per region, with the budgets below (3,600 s; 7,200 s for",
    f"{', '.join(RID[s] + ' with ' + format(NEXP[s], ',') + ' exposures' for s in OPEN if budget[s] == 7200)}). This run grades at 1,800 s, so these are not this run's numbers:",
    "they show how the same program does across the archive with two to four times the time. `history/gen000/` has its",
    "1,800 s run on the four graded regions.", "",
    f"Scored with `/app/tools/score.py` over the 23 regions together (slots = 2 x {nk}): recall {ga['reward']:.4f}, "
    f"{nf} of {nk} findable bodies, {ga['gpu_hours']} GPU-hours, cut p {ga['cut_p']}.", "",
    "- `out/<region>/candidates.json`: what it returned. `log/<region>.log`: what it printed while it ran. `timing.json`.",
    "- `grade.json`: the scoring in full (per body under `feedback`, unmatched candidates in the scored slots under `leads`).",
    "- `BODIES.md`, `bodies.json`: one row per findable body of the 23 regions (label index, name, H, nights per season, S/N,",
    "  fitted orbit) and what the program did with it.", ""] + T + ["",
    "## The search regions you do not have", "",
    "The same program also ran on 18 more regions (same budgets). Their data and labels are not in `/app/open`, so only the",
    "counts are here; they were scored against known bodies with H <= 8 only.", "",
    "| region | exposures | budget (s) | seconds | findable (H <= 8) | found | unmatched in slots |", "|---|---|---|---|---|---|---|"] +
    [f"| {a} | {b} | {c} | {d} | {e} | {f} | {g} |" for a, b, c, d, e, f, g in sorted(rest)]))
print(f"gen019 on 23 regions: recall {ga['reward']}, {nf}/{nk}, {ga['gpu_hours']} GPU-hours; {len(rest)} regions with counts only")

# ------------------------------------------------------------------------------------------------ 3. checked leads
nov = {}
for ln in open(NOVELTY):
    c = [x.strip() for x in ln.split("|")]
    if len(c) > 4 and c[2].startswith("dbt-"):
        nov[c[2]] = c[3]
sent = {x["id"]: x for f in ("runs/r1/leads_sent.json", "runs/s38/leads_sent.json") for x in J(os.path.join(HERE, f))}
checks = []
for run in ("r1", "s38"):
    d = os.path.join(HERE, "runs", run, "lead_check")
    for f in sorted(os.listdir(d)):
        if re.fullmatch(r"(gen\d+|v28-gen019|n10-gen019)\.json", f):
            checks += [dict(x, program=("gen019" if run == "s38" else f[:6]), run=("earlier run" if run == "r1" else "gen019 on the other regions"))
                       for x in J(os.path.join(d, f))]
labels = {}; LEADS = []; cnt = {}; lab_cnt = dict(unlabelled=0, known_body=0, no_data=0)
for x in checks:
    s = sent.get(x["id"])
    if not s:
        continue
    sid = s["sid"]; m = re.search(r"-rank(\d+)-p([\d.]+)$", x["id"])
    e = dict(id=x["id"], found_by=x["program"], also_returned_by=sorted(set(s.get("seen", []))), source=x["run"], region=RID.get(sid, sid),
             folder=sid if sid in OPEN else None, rank=int(m.group(1)), p=float(m.group(2)),
             fit=dict(rms_arcsec=x.get("rms"), r_au=x.get("r"), a=x.get("a"), e=x.get("e"), i=x.get("i"), H=x.get("H"), H_spread=x.get("H_spread")),
             detections=x.get("n"), nights=x.get("nights"), snr_med=x.get("snr_med"), arc_days=x.get("arc_days"),
             repeat_test=dict(verdict=x["verdict"], exposures_predicted=x.get("expected"), detected=x.get("found"),
                              exposures_not_in_the_linkage=x.get("extra"), detected_there=x.get("extra_found"), control_hits=x.get("null_hits")),
             catalogue_check=nov.get(x["id"], "not checked"))
    if sid in OPEN:
        if sid not in labels:
            z = np.load(os.path.join(TRUTH, sid + ".npz"), allow_pickle=True)
            labels[sid] = (z["det_body"], json.loads(str(z["known"])))
        db, kn = labels[sid]; det = np.asarray(s["det"], dtype=np.int64); b = db[det]
        e["det"] = [int(v) for v in det]
        e["labels_now"] = [dict(index=int(u), name=kn[int(u)].get("name"), H=kn[int(u)].get("H"), rows=int((b == u).sum())) for u in np.unique(b) if u >= 0]
        e["rows_unlabelled"] = int((b < 0).sum())
        lab_cnt["known_body" if e["labels_now"] and e["rows_unlabelled"] <= 0.2 * len(det) else "unlabelled"] += 1
    else:
        lab_cnt["no_data"] += 1
    v = re.sub(r" \(.*", "", x["verdict"]); cnt[v] = cnt.get(v, 0) + 1
    LEADS.append(e)
LD = os.path.join(EG, "leads")
os.makedirs(LD)
json.dump(LEADS, open(os.path.join(LD, "leads.json"), "w"), indent=1)
ncat = {}
for e in LEADS:
    c = re.sub(r"\s*OF 2026.*|\s*\(.*|:.*", "", e["catalogue_check"]).split(" 20")[0].split(" 19")[0].strip()
    ncat[c] = ncat.get(c, 0) + 1
write(os.path.join(LD, "README.md"), "\n".join([
    f"# What became of {len(LEADS)} unmatched candidates", "",
    "Candidates in the scored slots that matched no labelled body were all followed up: the earlier run's programs on REG_A-REG_C",
    "and gen019 on the other regions. `leads.json` has one entry per distinct candidate (a candidate returned again by a",
    "later program is listed once, with those programs under `also_returned_by`). This is the measured answer to 'how many of",
    "the confident unknowns are real', the second goal of the search.", "",
    "Two checks per candidate:", "",
    "1. **Catalogue check** (`catalogue_check`): is it a body the Minor Planet Center already has (any size, any H)?",
    "   " + "; ".join(f"{k}: {v}" for k, v in sorted(ncat.items(), key=lambda kv: -kv[1])) + ".",
    "2. **Repeat test** (`repeat_test`): fit an orbit to the linkage, predict where the body was in every exposure of the",
    "   region that covers that place and is deep enough, and look for a detection there. Exposures already in the linkage",
    "   prove nothing; the ones not in it are the test. `control_hits` is the same count at shifted positions (chance).",
    "   - SEEN AGAIN: detected in at least 2 exposures outside the linkage, at least half of them, at most 1 control hit.",
    "   - MISSING: at least 3 exposures outside the linkage and under a quarter detected, or under 40 % of all predicted.",
    "   - NO EXTRA EXPOSURE: fewer than 2 exposures outside the linkage, so the data cannot test it.",
    "   - UNCLEAR: in between. `no test`: no exposure predicted. `no bound orbit`: the fit is not a bound orbit.", "",
    "| repeat test | candidates |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])] + ["",
    f"So of {len(LEADS)} confident unmatched candidates, {cnt.get('SEEN AGAIN', 0)} were seen again and {cnt.get('MISSING', 0)} were shown to be",
    "false by the same data the program had. A program could run this test on its own candidates before giving them a `p`.", "",
    "Fields: `region`, `folder` (the region's folder in `/app/open`, null if you do not have it), `rank` and `p` in that",
    "program's scored list, `fit` (orbit fitted to the linkage; H from its fluxes), `det` (row numbers, only for regions you",
    f"have), `labels_now` and `rows_unlabelled`: what `/app/open/labels` says about those rows today. The labels now include",
    f"fainter known bodies that were not labelled when these ran: {lab_cnt['known_body']} of the {lab_cnt['known_body'] + lab_cnt['unlabelled']} candidates in your regions are",
    "such a body (at least 80 % of their rows are labelled)."] +
    ([] if os.path.isfile(os.path.join(HERE, "runs", "s38", "lead_check", "n10-gen019.json")) else
     ["Not included: gen019's candidates on 10 of the regions (588), whose checks were still running when this was written."])))
print(f"leads: {len(LEADS)}; repeat test {cnt}; catalogue {ncat}; labels {lab_cnt}")

# -------------------------------------------------------------------------------------------------- 4. index + note
g36 = max(rows, key=lambda r: r[1])
write(os.path.join(EG, "README.md"), f"""# Earlier engineering results of this task

Everything measured before this run started, so that nothing has to be found out twice. Nothing here is scored.

| Where | What |
|---|---|
| `../prior_r1/RESULTS.md` | the earlier run's {len(rows)} programs: recall then and now, bodies found and seconds per region |
| `../prior_r1/BODIES.md`, `bodies.json` | the {n69} findable bodies of REG_A-REG_C and which of those programs found each ({never} were found by none, {allf} by all) |
| `../prior_r1/genNNN/` | per program: code, notes, `out/` (candidates), `log/` (its own GPU run log), `timing.json`, `grade.json`, `bodies.md` |
| `gen019_all_regions/` | gen019 (= this run's gen000) on all 23 regions of `/app/open` at 3,600-7,200 s: outputs, logs, scoring, per-body table; counts for 18 more regions |
| `leads/` | {len(LEADS)} unmatched candidates and what the catalogue check and the repeat test said about each |

## What was measured

**The data**
- REG_B (`scen10`): 682 exposures, 64.4 M detections. 81.2 % of them are static (a detection within 1-2" on at least 2
  other nights). After a plain static cut, S/N >= 5 and point-like, 5.75 M are left (8,436 per exposure): mostly faint
  static sources and noise.
- The detection lists came with NaN pointings; `ra0`/`dec0` in `exposures.json` and the centre and radius in
  `region.json` were filled from the detections (mean direction per exposure; radius = farthest exposure centre + 1.1 deg).
- Known bodies with H <= 8 crossing REG_A-REG_C: 370. Detected on at least 5 nights (findable): 62. Imaged on fewer than
  5 nights: 155. 75 more would reach S/N 7 in a one-season stack of the images, which single-exposure lists cannot give.
  So most known bodies in a region are out of reach of any program working on these lists; the findable ones are the target.

**The first programs**
- The starter (a vendored GPU pair linker) wrote 0 candidates: it passed degrees to a linker that works in radians, and
  its static cut alone took 1,067 s on REG_B in single-threaded numpy. Fixed, it found 1 body in a full hour per region
  (recall 0.0202). The first agent replaced it entirely.
- gen001: 0.2828 (16 of 62; under 1,000 s of 3,600 per region). Short-window triplet search per distance, then two-body
  growth across all years.

**The earlier run (recall under its rules: H <= 8 bodies of REG_A-REG_C, 3,600 s per region)**
- gen001 0.2828, gen019 0.5859 (35 of 62: 12/27, 18/27, 5/8), best {g36[0]} {g36[1]:.4f} (36 of 62: 13/27, 18/27, 5/8; it adds
  'unit triples' in the sparse pass on top of gen035).
- Of the twelve programs gen020-gen031, nine scored 0.5859 and three 0.5758. Their additions
  differ (two-plus-two-night sparse passes across seasons, star-frame astrometry, faint-seed repeats, inner passes) and
  the nine at 0.5859 all find 35 bodies. Recall on those regions has barely moved since gen019; `../prior_r1/BODIES.md`
  shows which bodies nobody finds and what they look like.
- Nearly every program there fills its 3,600 s. Nobody was asked to be fast, so run time is unexplored.

**gen019 at this run's budget and across the archive**
- 1,800 s per region on the four graded regions (`history/gen000/`): recall 0.479 (51 of 113), 1.9411 GPU-hours: it fills
  the ceiling on every region; 60 bodies are in no candidate. Against the H <= 8 bodies alone: 46 of 98.
- 3,600-7,200 s on all 23 open regions: {nf} of {nk} findable bodies (`gen019_all_regions/`).
- On 28 other-region runs scored together it found 68 of 187 findable H <= 8 bodies (36 %), against 35 of 62 on REG_A-REG_C:
  the program was tuned on three regions and does worse elsewhere. Small regions (92 or 323 exposures) gave many
  unmatched candidates and no known body.

**Unmatched candidates**
- Of {len(LEADS)} confident unmatched candidates followed up, the repeat test saw {cnt.get('SEEN AGAIN', 0)} again and showed {cnt.get('MISSING', 0)} to be false
  with the region's own exposures (`leads/`). High `p` on an unmatched linkage has so far meant little.

**CPU tests on your machine (8 cores)**
- gen019 on region R037 (17.2 M detections): `subset.py --near-bodies 0.1` keeps 0.67 M detections in 5.5 s and the
  search takes 38 s (2 of 4 bodies); `--near-bodies 0.5` keeps 6.75 M and the search takes 9 min at a 600 s budget
  (1 of 4). A cut around the bodies is easier than the whole region: fewer false pairs, so use it to see why a body is
  lost, and expect less on the full region.
""")
write(os.path.join(R2, "history_extra_note.md"),
      f"- Every earlier result of this task is in the history as well. Start at `/app/history/engineering/README.md` (index and\n"
      f"  measured facts). `/app/history/prior_r1/RESULTS.md`: the earlier run's {len(rows)} programs with bodies found and seconds per\n"
      f"  region; each `prior_r1/genNNN/` also holds that program's candidates, its GPU run log, and its outcome for every body\n"
      f"  scored with this run's code and labels (`prior_r1/BODIES.md`: {n69} bodies of REG_A-REG_C, {never} found by no program).\n"
      f"  `/app/history/engineering/gen019_all_regions/`: gen000's program run on all 23 regions of `/app/open` ({nf} of {nk} bodies)\n"
      f"  with a per-body table. `/app/history/engineering/leads/`: {len(LEADS)} unmatched candidates that were followed up\n"
      f"  ({cnt.get('SEEN AGAIN', 0)} seen again in other exposures, {cnt.get('MISSING', 0)} shown false).\n")
sz = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
print(f"{OUT}: {sz / 1e6:.1f} MB")
