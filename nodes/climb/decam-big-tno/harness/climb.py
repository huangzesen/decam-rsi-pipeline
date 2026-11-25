#!/usr/bin/env python3
"""Hill climb on decam-big-tno: each round's agent improves the program that finds large distant bodies; graded on Vista GPUs.

2026-11-12. Adapted from ../magnetosphere-nn/harness/hill_climb.py.

A round, run from the Mac:
 1. history (earlier code, notes, rewards) -> the trial host; the task is staged there with /app/submission = the best program
 2. one Harbor trial (agent MODEL at xhigh, 8 CPUs, AGENT_S, WebSearch/WebFetch denied, verification off)
 3. the job is copied to runs/<run>/gens/genNNN/; the submission is scanned (network use, designations of known bodies: 0)
 4. the submission goes to Vista $SCRATCH/dbt/runs/<run>/<gen>/sub; run_gb.sbatch runs it on every sealed patch (one GB200
    each, BUDGET_S, no network); only outputs and timings come back; harness/grade.py scores them here against private/truth
 5. the best by reward is kept on the host for the next round; leads (confident unmatched candidates) are kept in the grade

Nothing starts a round without the user's go: `run` refuses unless runs/<run>/GO exists.
    python3 -I harness/climb.py run --run r1 --target 20
    python3 -I harness/climb.py status --run r1
    python3 -I harness/climb.py starter --run r1 --grade DIR     record the graded starter as gen000 (the first best)
    touch runs/<run>/STOP          stop before the next round; a running agent is left to finish
Python standard library only (grade.py needs numpy).
"""
import argparse, fcntl, glob, hashlib, json, os, queue, re, shlex, shutil, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nersc_grade  # noqa: E402
LOCK = threading.RLock()
LEADS_LOCK = threading.Lock()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = os.path.join(HERE, "harbor/decam-big-tno")
TRUTH = os.path.join(HERE, "private/truth")
NPY = os.path.expanduser("~/anaconda3/bin/python")         # grade.py needs numpy
VM, W = "$VM_HOST", "$VM_WORK"
VM_BASE = f"{W}/tasks-ale/decam-big-tno"
VISTA = "$TACC_USER@vista.tacc.utexas.edu"
VSOCK = os.path.expanduser("$SSH_CM_DIR/vista-$TACC_USER.sock")
VD = "$TACC_SCRATCH/dbt"
MODEL = "claude-fable-5-2"
# Two models per round (2026-11-14): design 2 with --workers 2 runs one Claude Code agent on
# Fable and one Codex agent on Astra per round, both at effort xhigh, with the same history (every earlier program of
# this run and of the earlier run). The next round starts when both programs are graded.
PAIR = [dict(agent="claude-code", model=MODEL, name="Claude Fable 5.2"),
        dict(agent="codex", model="openai/gpt-6-astra", name="GPT 6 Astra")]
AGENT_S = 7200
# From gen022 (2026-11-13): /app/submission starts empty, agents run rolling (a new one starts
# whenever one finishes, no pairs), and a program identical to any graded one takes that grade instead of a NERSC node.
NO_BASE = True
BUDGET_S = 3600
MAX_ATTEMPTS = 12
# Design 2 (run config "design": 2; 2026-11-14): two rewards (recall, GPU-hours), one row of feedback per graded body in the
# history, open labelled regions mounted at /app/open, one agent per round that starts after the previous grade.
DESIGN = 1
OPEN = None                                               # host folder of the open labelled regions (design 2)
AGENT_OVERRIDE = None                                     # one agent for every new generation (config "agent_override")
BINARY = (".pt", ".pth", ".npz", ".npy", ".ckpt", ".bin", ".safetensors", ".pkl", ".pickle", ".onnx", ".h5", ".gz", ".zip", ".tar")
SSH = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"]
VSSH = ["-S", VSOCK, "-o", "ControlMaster=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
# 2026-11-21: code, not words: gen069 was scored 0 for "requests" in a comment ("The negative sign requests the broad
# transverse-velocity prior"). Imports, includes, calls, module loads by name, URLs and quoted wget/curl.
NET = re.compile(r"^\s*(?:import|from)\s+(?:urllib\d?|requests|http|socket|aiohttp|httpx|pycurl|ftplib|smtplib|websocket)\b"
                 r"|#\s*include\s*<(?:sys/socket|netdb|arpa/inet|netinet/in|curl/curl)\.h>"
                 r"|\b(?:__import__|import_module)\(\s*['\"](?:urllib|requests|http|socket|aiohttp|httpx|pycurl|ftplib|smtplib)"
                 r"|\b(?:socket\.(?:socket|create_connection)|urllib\d?\.request|requests\.(?:get|post|put|head|Session))\b"
                 r"|https?://|['\"](?:wget|curl)\b", re.M)
DESIG = re.compile(r"\b(19|20)\d\d[ _]?[A-HJ-Y][A-HJ-Z]\d{0,3}\b")


def configure(run):
    """Run settings from runs/<run>/config.json: truth folder, time cap per region, the task's folder on the host, design."""
    global TRUTH, BUDGET_S, VM_BASE, DESIGN, OPEN, AGENT_OVERRIDE
    p = os.path.join(HERE, "runs", run, "config.json")
    cfg = json.load(open(p)) if os.path.exists(p) else {}
    TRUTH = os.path.join(HERE, cfg.get("truth", "private/truth")); nersc_grade.TRUTH = TRUTH
    BUDGET_S = int(cfg.get("budget_s", BUDGET_S)); VM_BASE = cfg.get("vm_base", VM_BASE)
    DESIGN = int(cfg.get("design", 1)); OPEN = cfg.get("open")
    # 2026-11-17: config "agent_override" =
    # {"agent", "model", "name", "effort"} makes every new generation use that one agent (use with --workers 1).
    AGENT_OVERRIDE = cfg.get("agent_override")
    return cfg


def paths(run):
    return os.path.join(HERE, "runs", run), f"{W}/tasks-ale/runs/dbt-{run}", f"{VD}/runs/{run}"


def log(RUN, *a):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with LOCK, open(os.path.join(RUN, "climb.log"), "a") as f:
        f.write(line + "\n")


def load(RUN):
    p = os.path.join(RUN, "state.json")
    return json.load(open(p)) if os.path.exists(p) else dict(target=20, best=None, gens={})


def save(RUN, st):
    p = os.path.join(RUN, "state.json")
    for i in range(20):                                   # the grading thread writes too
        try:
            with LOCK:
                txt = json.dumps(st, indent=1)
            break
        except RuntimeError:
            time.sleep(0.05)
    with LOCK:
        with open(p + ".tmp", "w") as f:
            f.write(txt)
        os.replace(p + ".tmp", p)


def ssh(cmd, tries=30, vista=False):
    """Run on the trial host (or Vista). Waits through a dropped connection; any other exit code is returned."""
    for i in range(tries):
        argv = ["ssh", *VSSH, VISTA, cmd] if vista else ["ssh", *SSH, VM, cmd]
        p = subprocess.run(argv, capture_output=True, text=True)
        if p.returncode != 255:
            return p.returncode, p.stdout
        time.sleep(60 if not vista else 300)          # a lapsed Vista login waits for the user's MFA
    raise RuntimeError(("Vista" if vista else "the trial host") + " cannot be reached")


def rsync(src, dst, vista=False):
    # 2026-11-19 23:50: the shared $VM_HOST ControlMaster ran at ~40 KB/s while a fresh connection moved 8 MB in 2 s, so
    # copies to and from the trial host open their own connection (ControlPath=none); Vista keeps its MFA socket.
    e = "ssh " + " ".join(VSSH if vista else SSH + ["-o", "ControlPath=none"])
    for i in range(10):
        p = subprocess.run(["rsync", "-a", "--partial", "-e", e, src, dst], capture_output=True, text=True)
        if p.returncode == 0:
            return
        time.sleep(60)
    raise RuntimeError(f"rsync {src} {dst} failed: {p.stderr[-200:]}")


def tree_hash(path):
    h = hashlib.sha1()
    for d, ds, fs in sorted(os.walk(path)):
        ds.sort()
        for f in sorted(fs):
            full = os.path.join(d, f)
            if f == ".keep" or os.path.islink(full) or "__pycache__" in full:
                continue
            h.update(os.path.relpath(full, path).encode())
            with open(full, "rb") as g:
                for chunk in iter(lambda: g.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


def scan(sub):
    """Network use or designations of known bodies in the submission's text files: either scores 0."""
    hits = []
    for d, _, fs in os.walk(sub):
        for f in fs:
            p = os.path.join(d, f)
            if os.path.islink(p):
                hits.append(f"{os.path.relpath(p, sub)}: symbolic link"); continue
            if f.lower().endswith(BINARY) or os.path.getsize(p) > 20 << 20:
                continue
            txt = open(p, errors="replace").read()
            if "\0" in txt[:8192]:
                continue
            rel = os.path.relpath(p, sub)
            if rel.endswith(".md"):
                continue                               # notes may name what they found
            hits += [f"{rel}: network '{m.group(0)}'" for m in NET.finditer(txt)][:3]
            hits += [f"{rel}: designation '{m.group(0)}'" for m in DESIG.finditer(txt)][:3]
    return hits


def note_head(sub):
    p = os.path.join(sub, "NOTES.md") if sub else ""
    want = {"Method": "", "Base": "", "Change": ""}
    if p and os.path.isfile(p) and not os.path.islink(p):
        for line in open(p, errors="replace").read(65536).splitlines()[:12]:
            m = re.match(r"\W*(Method|Base|Change)\W*:\W*(.*)", line)
            if m and not want[m.group(1)]:
                want[m.group(1)] = m.group(2).strip()[:300]
    return want


def copy_code(sub, dst):
    total = 0
    for d, ds, fs in os.walk(sub):
        ds[:] = sorted(x for x in ds if not x.startswith("__pycache__") and not os.path.islink(os.path.join(d, x)))
        for f in sorted(fs):
            full = os.path.join(d, f)
            if os.path.islink(full) or f.lower().endswith(BINARY) or os.path.getsize(full) > 5 << 20 or total > 30 << 20:
                continue
            with open(full, "rb") as g:
                if b"\0" in g.read(8192):
                    continue
            to = os.path.join(dst, os.path.relpath(full, sub))
            os.makedirs(os.path.dirname(to), exist_ok=True)
            shutil.copyfile(full, to); total += os.path.getsize(full)


def loss_text(RUN, st):
    """Where the best graded program loses, as counts only (2026-11-14):
    its missed findable known bodies by how their nights fall into observing seasons (nights 150+ days apart start a new
    season). No names, places, dates or regions."""
    import numpy as np
    best = st.get("best")
    try:
        g = json.load(open(os.path.join(RUN, "gens", best["gen"], "grade", "grade.json")))
    except Exception:
        return ""
    cnt = {}
    for sid in g.get("found", {}):
        z = np.load(os.path.join(TRUTH, sid + ".npz")); K = json.loads(str(z["known"])); body, night = z["det_body"], z["det_night"]
        found = set(int(b) for b in g["found"][sid])
        for b, k in enumerate(K):
            if k["nights"] < 5:
                continue
            ns = np.sort(np.unique(night[body == b])); best_s, cur = 1, 1
            for x0, x1 in zip(ns[:-1], ns[1:]):
                cur = cur + 1 if x1 - x0 < 150 else 1; best_s = max(best_s, cur)
            key = ("found" if b in found else "missed", min(best_s, 5))
            cnt[key] = cnt.get(key, 0) + 1
    later = [v for k, v in st["gens"].items() if k > best["gen"] and v.get("phase") == "done" and v.get("reward") is not None]
    ties = sum(abs(v["reward"] - best["reward"]) < 1e-9 for v in later)
    same = f" {ties} of the {len(later)} generations graded after it scored exactly the same." if later else ""
    nf = sum(v for (c, _), v in cnt.items() if c == "found"); nm = sum(v for (c, _), v in cnt.items() if c == "missed")
    if not nm:
        return ""
    m = lambda c, s: cnt.get((c, s), 0)
    return (f"- Where the best program ({best['gen']}) loses, counted on the sealed patches (counts only): it finds {nf} of the "
            f"{nf + nm} findable known bodies. Of the {nm} it misses, {m('missed', 5)} have >= 5 nights inside one observing "
            f"season; {nm - m('missed', 5)} have their >= 5 nights spread over several seasons with at most 4 in any one season "
            f"({m('missed', 2)} at most 2, {m('missed', 3)} at most 3, {m('missed', 4)} at most 4; a season = nights less than "
            f"150 days apart). Of the {nf} it finds, {nf - m('found', 5)} are of the spread kind." + same +
            " The gain is in linking bodies seen on a few nights per season across years.\n")


def loss_table(RUN, st, hist):
    """history/LOSS.md: one row per findable known body of the sealed patches, found or missed by the best graded program,
    with its nights per season, S/N and fitted orbit (runs/<run>/loss/bodies.json from harness/body_props.py); no names,
    places or dates (2026-11-14)."""
    best = st.get("best")
    try:
        B = json.load(open(os.path.join(RUN, "loss", "bodies.json")))
        g = json.load(open(os.path.join(RUN, "gens", best["gen"], "grade", "grade.json")))
    except Exception:
        return False
    sids = json.load(open(os.path.join(RUN, "config.json")))["sids"]
    rows = []
    for pi, sid in enumerate(sids, 1):
        found = set(g.get("found", {}).get(sid, {}))
        for b, m in B.get(sid, {}).items():
            rows.append((b in found, -m["weight"], pi, m))
    rows.sort(key=lambda x: (x[0], x[1], x[2], -x[3]["nights"]))
    L = [f"# Findable known bodies of the sealed patches, and which ones the best program ({best['gen']}, reward {best['reward']:.4f}) finds", "",
         "One row per body with detections on >= 5 nights in its patch (the grader's 'findable'); missed ones first. Weight 2 =",
         "H <= 7, 1 = 7 < H <= 8. Seasons: its nights per observing season (nights 150+ days apart start a new season). S/N: of",
         "its detections in the lists. Orbit: two-body fit to its detections (r = heliocentric distance in the middle of the arc).",
         "No names, places or dates. Use it to see which kinds of bodies your method must reach; tune on your own planted bodies.", "",
         "| # | found | weight | patch | nights | nights per season | arc (yr) | detections | S/N median | S/N max | r (au) | a | e | i (deg) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, (f, w, pi, m) in enumerate(rows, 1):
        L.append(f"| {k} | {'yes' if f else '**no**'} | {m['weight']} | {pi} | {m['nights']} | {'+'.join(map(str, m['seasons']))} | "
                 f"{m['arc_yr']} | {m.get('n_det', '')} | {m.get('snr_med', '')} | {m.get('snr_max', '')} | {m.get('r_au', '')} | "
                 f"{m.get('a', '')} | {m.get('e', '')} | {m.get('i', '')} |")
    open(os.path.join(hist, "LOSS.md"), "w").write("\n".join(L) + "\n")
    return True


WHAT = dict(found="found", below_cut="linked, ranked below the cut", few_nights="linked on too few nights",
            mixed="mixed with other detections", none="in no candidate")


def body_id(sid, b):
    return "k" + hashlib.sha1(f"{sid}:{b}".encode()).hexdigest()[:5]


def rank_key(v):
    """Design 2 order: recall first, then fewer GPU-hours."""
    return (-(v.get("reward") or 0), v.get("gpu_hours") if v.get("gpu_hours") is not None else 9e9)


def body_feedback(RUN, done, hist):
    """Design 2: what every graded generation did with every findable known body of the graded regions.
    history/genNNN/bodies.md (one generation), history/BODIES.md and history/bodies.json (all). Nothing is hidden from the
    agent (2026-11-14): each body comes with
    its region, its index in that region's label file (/app/open/labels/<sid>.npz), name and H. The short id (kxxxxx) is
    only a handle that stays the same from round to round; shared pages show nothing but it."""
    sids = json.load(open(os.path.join(RUN, "config.json")))["sids"]
    try:
        P = json.load(open(os.path.join(RUN, "loss", "bodies.json")))
    except Exception:
        P = {}
    FB = {}
    for k, v in sorted(done.items()):
        if v["phase"] != "done" or v.get("status") != "valid":
            continue
        try:
            g = json.load(open(os.path.join(v.get("grade_dir") or os.path.join(RUN, "gens", k, "grade"), "grade.json")))
        except Exception:
            continue
        if "feedback" in g:
            FB[k] = (g, v)
    if not FB:
        return False
    g0 = next(iter(FB.values()))[0]
    bodies = [(pi, sid, b) for pi, sid in enumerate(sids, 1) for b in sorted(g0["feedback"].get(sid, {}), key=int)]
    ids = {(sid, b): body_id(sid, b) for _, sid, b in bodies}
    assert len(set(ids.values())) == len(ids)
    gens = sorted(FB, key=lambda k: rank_key(FB[k][1]) + (k,))
    best = gens[0]
    J = dict(note="one entry per findable known body of the graded regions; outcome per graded generation; region + index "
                  "point into /app/open/labels/<region>.npz (det_body == index)", generations={
        k: dict(reward=FB[k][1].get("reward"), gpu_hours=FB[k][1].get("gpu_hours")) for k in gens}, bodies={})
    trow = lambda v: "; ".join(f"{n}: {x['pure_nights']}/{x['any_nights']}" for n, x in v.get("trace", {}).items())
    for k in gens:                                         # one file per generation
        g, v = FB[k]
        L = [f"# {k}: what it did with each findable known body (recall {v['reward']:.4f}, GPU-hours {v.get('gpu_hours')})", "",
             "Outcome over all valid candidates of the body's region; nights linked / purity / rank describe the candidate that",
             "got furthest with it. Trace (only if the program wrote trace.json): per step, the most of the body's nights in one",
             "group that is >= 80 % this body / in any group. Region and index point into `/app/open/labels/<region>.npz`",
             "(`det_body == index` are the body's detections). The program's candidates are in `out/<region>/candidates.json`.", "",
             "| body | region | index | name | H | weight | nights | outcome | nights linked | purity | rank (p) | trace |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        order = dict(none=0, mixed=1, few_nights=1, below_cut=3, found=4)
        for pi, sid, b in sorted(bodies, key=lambda x: (order[g["feedback"][x[1]][x[2]]["stage"]], x[0], int(x[2]))):
            f = g["feedback"][sid][b]; m = P.get(sid, {}).get(b, {})
            rk = "" if f["stage"] == "none" else (f"{f['rank']} ({f['p']:.3f})" if f.get("rank") else
                                                  f"not read: {f['rank_in_scenario']} in its region ({f['p']:.3f})")
            L.append(f"| {ids[(sid, b)]} | {sid} | {b} | {m.get('name', '')} | {m.get('H', '')} | {m.get('weight', '')} | {m.get('nights', '')} | {WHAT[f['stage']]} | "
                     f"{f.get('nights_linked', '')} | {f.get('purity', '')} | {rk} | {trow(f)} |")
        if os.path.isdir(os.path.join(hist, k)):
            open(os.path.join(hist, k, "bodies.md"), "w").write("\n".join(L) + "\n")
    rows = []
    for pi, sid, b in bodies:
        out = {k: FB[k][0]["feedback"][sid][b] for k in gens}
        nf = sum(1 for f in out.values() if f["stage"] == "found")
        m = P.get(sid, {}).get(b, {})
        J["bodies"][ids[(sid, b)]] = dict(region=sid, index=int(b), **m, found_by=nf, outcomes={k: {x: f[x] for x in f if x != "size"} for k, f in out.items()})
        furthest = max(out.items(), key=lambda kv: (dict(none=0, mixed=1, few_nights=1, below_cut=3, found=4)[kv[1]["stage"]],
                                                    kv[1].get("nights_linked", 0)))
        rows.append((nf, -(m.get("weight") or 0), pi, ids[(sid, b)], dict(m, region=sid, index=b), out[best], furthest))
    rows.sort(key=lambda x: x[:4])
    n = len(gens)
    L = [f"# The {len(rows)} findable known bodies of the graded regions and what {n} graded generation{'s' if n > 1 else ''} did with them", "",
         "One row per body with detections on >= 5 nights in its region (the grader's 'findable'); the ones found by the fewest",
         "generations first. Weight 2 = H <= 7, 1 = fainter (any H). Seasons: nights per observing season (nights 150+ days apart",
         "start a new season). S/N: of its detections in the lists. Orbit: two-body fit to its detections (r = heliocentric",
         "distance in the middle of the arc). `found by`: generations whose scored list held it. `best`: outcome in the best",
         f"program ({best}). `furthest`: the generation that got furthest with it, with the nights it linked. Region and index",
         "point into `/app/open/labels/<region>.npz`: the rows with `det_body == index` are the body's detections, so you can look",
         "at every missed body in the data. The id in the first column is only a short handle that stays the same from round",
         "to round. Each generation's own table is `genNNN/bodies.md`; everything here is also in `bodies.json`.", "",
         f"| body | region | index | name | H | weight | found by (of {n}) | best ({best}) | furthest | nights | nights per season | arc (yr) | detections | S/N median | S/N max | r (au) | a | e | i (deg) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for nf, _, pi, bid, m, fb, (fk, ff) in rows:
        far = "" if ff["stage"] == "none" else f"{fk}: {WHAT[ff['stage']]}" + (f", {ff['nights_linked']} nights" if ff["stage"] != "found" else "")
        L.append(f"| {bid} | {m['region']} | {m['index']} | {m.get('name', '')} | {m.get('H', '')} | {m.get('weight', '')} | {nf} | {WHAT[fb['stage']]} | {far} | {m.get('nights', '')} | "
                 f"{'+'.join(map(str, m.get('seasons', [])))} | {m.get('arc_yr', '')} | {m.get('n_det', '')} | {m.get('snr_med', '')} | "
                 f"{m.get('snr_max', '')} | {m.get('r_au', '')} | {m.get('a', '')} | {m.get('e', '')} | {m.get('i', '')} |")
    open(os.path.join(hist, "BODIES.md"), "w").write("\n".join(L) + "\n")
    json.dump(J, open(os.path.join(hist, "bodies.json"), "w"), indent=1)
    g, v = FB[best]; cnt = {}
    for _, sid, b in bodies:
        s_ = g["feedback"][sid][b]["stage"]; cnt[s_] = cnt.get(s_, 0) + 1
    never = sum(1 for r in rows if r[0] == 0)
    return (f"- What the best program ({best}) does with the {len(rows)} findable known bodies of the graded regions: "
            f"{cnt.get('found', 0)} found; {cnt.get('none', 0)} in no candidate; {cnt.get('mixed', 0)} mixed with other detections; "
            f"{cnt.get('few_nights', 0)} linked on too few nights; {cnt.get('below_cut', 0)} linked but ranked below the cut. "
            f"{never} bodies were found by no generation so far.\n"
            "  `/app/history/BODIES.md` has one row per body (region and index in the label file, name, H, nights per season, S/N,\n"
            "  fitted orbit, which generations found it); `/app/history/genNNN/bodies.md` has each generation's outcome per body,\n"
            "  `genNNN/grade.json` the grader's full output and `genNNN/out/<region>/candidates.json` what the program returned.\n")


def prior_history(RUN, hist):
    """Design 2: the programs of an earlier run (run config "prior": {"run": "r1", "rules": "...", "gen000": "gen019"}) as
    history/prior_<run>/genNNN/ (code, NOTES.md, reward.json) with a SUMMARY.md, so that what that run learned is not lost.
    Their rewards were measured under other rules and are labelled so. Returns the line for the session note."""
    pr = json.load(open(os.path.join(RUN, "config.json"))).get("prior")
    if not pr:
        return ""
    name = "prior_" + pr["run"]; dst = os.path.join(hist, name); os.makedirs(dst)
    ps = json.load(open(os.path.join(HERE, "runs", pr["run"], "state.json")))
    gens = {k: v for k, v in ps["gens"].items() if v.get("phase") == "done" and v.get("status") == "valid" and v.get("reward") is not None
            and v.get("submission") and os.path.isdir(v["submission"])}
    L = [f"# The {len(gens)} graded programs of an earlier run of this task, best first", "",
         f"Graded under other rules: {pr['rules']} Their recall is not comparable with this run's numbers; the code and the",
         "notes are what matters. Each `genNNN/` holds the full code, `NOTES.md` and `reward.json`. Notes there that mention",
         "`/app/history/genNNN` mean the folders here.", ""]
    for k, v in sorted(gens.items(), key=lambda kv: (-kv[1]["reward"], kv[0])):
        d = os.path.join(dst, k); os.makedirs(d); copy_code(v["submission"], d)
        json.dump(dict(generation=k, run=pr["run"], recall_under_the_earlier_rules=v["reward"], rules=pr["rules"]),
                  open(os.path.join(d, "reward.json"), "w"), indent=1)
        L.append(f"## {k}: recall {v['reward']:.4f} (earlier rules)" + (" = gen000 of this run" if k == pr.get("gen000") else ""))
        L += [f"- {key}: {val}" for key, val in note_head(v["submission"]).items() if val] + [""]
    open(os.path.join(dst, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    same = (f" This run's gen000 is that run's {pr['gen000']}, graded again under this run's rules; `prior_r1/RESULTS.md` shows which of"
            f" those programs scored highest (gen036, 0.6061, graded after this run was prepared).") if pr.get("gen000") else ""
    return (f"- `/app/history/{name}/` holds the {len(gens)} graded programs of an earlier run of this task (code, notes, `SUMMARY.md`).\n"
            f"  They were graded under other rules ({pr['rules'].rstrip('.')}), so their recall is not comparable with this run's.{same}\n")


def handoff(RUN, st, gen, seed_gen=None):
    out = os.path.join(RUN, "handoff", gen)
    shutil.rmtree(out, ignore_errors=True)
    hist = os.path.join(out, "history"); os.makedirs(hist)
    with LOCK:
        done = {k: dict(v) for k, v in st["gens"].items() if v.get("phase") in ("done", "grading")}
    for k, v in done.items():
        d = os.path.join(hist, k); os.makedirs(d)
        if v.get("submission") and os.path.isdir(v["submission"]) and v.get("status") != "unchanged":
            copy_code(v["submission"], d)
        json.dump(dict(generation=k, reward=v.get("reward") if v["phase"] == "done" else None,
                       status=v.get("status") if v["phase"] == "done" else "being graded", invalid_reason=v.get("reason"),
                       graded_seconds_per_patch=v.get("seconds") if v["phase"] == "done" else None,
                       **(dict(gpu_hours=v.get("gpu_hours") if v["phase"] == "done" else None,
                               unmatched_candidates_in_scored_slots=v.get("leads") if v["phase"] == "done" else None) if DESIGN == 2 else {})),
                  open(os.path.join(d, "reward.json"), "w"), indent=1)
        if DESIGN == 2 and v["phase"] == "done" and v.get("grade_dir"):     # the grader's full output and the program's own
            gd = v["grade_dir"]; od = v.get("out_dir") or gd + "_runs/out"  # outputs on the graded regions: nothing is hidden
            if os.path.exists(os.path.join(gd, "grade.json")):
                shutil.copy(os.path.join(gd, "grade.json"), os.path.join(d, "grade.json"))
            for sid in json.load(open(os.path.join(RUN, "config.json")))["sids"]:
                c = os.path.join(od, sid, "candidates.json")
                if os.path.exists(c):
                    os.makedirs(os.path.join(d, "out", sid)); shutil.copy(c, os.path.join(d, "out", sid, "candidates.json"))
    lines = ["# Earlier generations, best first", "",
             "Each `genNNN/` holds that generation's full code, `NOTES.md` (method, base, change, measurements) and `reward.json`",
             "(reward and graded run time per patch). Read them and decide what to build on.", ""]
    if DESIGN == 2:
        lines[2:4] = ["Each `genNNN/` holds that generation's full code, `NOTES.md` (method, base, change, measurements), `reward.json`",
                      "(recall, GPU-hours, run time per region), `bodies.md` (what it did with each findable known body), `grade.json`",
                      "(the grader's full output) and `out/<region>/candidates.json` (what it returned). Order: recall, then fewer GPU-hours."]
    order = ((lambda kv: (kv[1]["phase"] == "done",) + (rank_key(kv[1]) if kv[1]["phase"] == "done" else (-9, 0)) + (kv[0],)) if DESIGN == 2 else
             (lambda kv: (kv[1]["phase"] == "done", -((kv[1].get("reward") or 0) if kv[1]["phase"] == "done" else 9), kv[0])))
    for k, v in sorted(done.items(), key=order):
        if v["phase"] == "grading":
            lines.append(f"## {k}: being graded (no reward yet)")
        elif DESIGN == 2 and v.get("reward") is not None:
            gh = v.get("gpu_hours")
            lines.append(f"## {k}: recall {v['reward']:.4f}, GPU-hours {gh if gh is not None else 'not measured'}")
        else:
            lines.append(f"## {k}: reward {v['reward']:.4f}" if v.get("reward") is not None else f"## {k}: not graded")
        if v.get("status") == "unchanged":
            lines.append("Left the submission unchanged.")
        if v.get("reason"):
            lines.append(f"Scored 0: {v['reason']}")
        if v.get("seconds") and v["phase"] == "done":
            secs = [x for x in v["seconds"].values() if x]
            lines.append(f"- graded run time per {'region' if DESIGN == 2 else 'patch'} (s): " + ", ".join(f"{x:.0f}" for x in secs) + f" (limit {BUDGET_S})")
        if v.get("started") and v.get("written"):
            lines.append(f"- agent session: {(v['written'] - v['started']) / 60:.0f} min")
        if v.get("seed"):
            lines.append(f"- started from: {v['seed']}")
        for key, val in (note_head(v.get("submission")) if v.get("status") != "unchanged" else {}).items():
            if val:
                lines.append(f"- {key}: {val}")
        lines.append("")
    if not done:
        lines.append("None yet: you are the first.")
    open(os.path.join(hist, "SUMMARY.md"), "w").write("\n".join(lines) + "\n")
    bodies_txt = ""
    try:
        if DESIGN == 2:
            has_table = False; bodies_txt = body_feedback(RUN, done, hist) or ""
        else:
            has_table = loss_table(RUN, st, hist)
    except Exception as e:
        log(RUN, gen, "loss table failed:", repr(e)[:200]); has_table = False
    best = st.get("best")
    btxt = f"{best['gen']} with a reward of {best['reward']:.4f}" if best else "the starter"
    if best and DESIGN == 2:
        btxt = f"{best['gen']}: recall {best['reward']:.4f}, {best.get('gpu_hours')} GPU-hours"
    if NO_BASE:
        start = ("`/app/submission` starts empty: there is no default program. " +
                 (f"The best graded program so far is {btxt}. " if best else ""))
    else:
      start = (f"`/app/submission` starts as {seed_gen}'s program, the most recently finished generation, which is still being graded "
             f"(agents do not wait for grades); the best graded program so far is {btxt}." if seed_gen else
             f"`/app/submission` starts as the best program so far, {btxt}." if best else
             "`/app/submission` starts as the starter (the plain search).")
    intro = (f"""You are one generation in a sequence of agents improving a program for this task. {start}Every earlier
generation's full program, method notes, reward and graded run time per patch are in `/app/history/` (the ones still being
graded too, without a reward): read them and decide what to build on. Copy one, combine ideas from several, or write a new
method; nothing is given to you as a base, the choice is yours. Another agent works at the same time with the same
history, so a different approach from the other generations is valuable. A program identical to an earlier one gets that
one's grade without being run again.""" if NO_BASE else
             f"""You are one generation in a sequence of agents improving this program. {start} Keep improving it, combine ideas from
other generations, or replace it entirely. Generations run in pairs: another agent starts from the same program at the
same time, so a different approach from the other generations is valuable. Every earlier generation's full program,
method notes, reward and graded run time per patch are in `/app/history/` (and the ones still being graded): you decide
what to build on. Start from any of them, combine them, or start over; the program you were given is only the default.""")
    if DESIGN == 2:
        other = " Another agent works at the same time with the same\nhistory, so a different approach from the other generations is valuable."
        assert other in intro
        intro = intro.replace(other, " Each round has two agents on two different models: another agent works at the same time with the same\n"
                              "history, and neither sees the other's work, so a different approach from the other generations is valuable."
                              if (st.get("workers") or 1) >= 2 else "")
        intro = intro.replace("reward and graded run time per patch", "two rewards (recall and GPU-hours) and its outcome for every graded body")
    loss = bodies_txt if DESIGN == 2 else loss_text(RUN, st)
    if DESIGN == 2:
        try:
            loss += prior_history(RUN, hist)
        except Exception as e:
            log(RUN, gen, "prior history failed:", repr(e)[:200])
        extra = os.path.join(RUN, "history_extra")          # every earlier result of the task (harness/eng_history.py;
        if os.path.isdir(extra):                            # 2026-11-14)
            shutil.copytree(extra, hist, dirs_exist_ok=True)
            loss += open(os.path.join(RUN, "history_extra_note.md")).read()
    judge = ("To judge your work before it is graded, run it on regions (or cuts of\n  them) in `/app/open` and score it with "
             "`python3 /app/tools/score.py`: every known body is labelled, the graded regions\n  included, and the score and the "
             "per-body table are computed exactly as the grader does. Your result is trusted and your\n  program is read: what "
             "counts is a method that works on regions it has never seen." if DESIGN == 2 else "To judge your work, plant your own bodies in the sample.")
    if loss and has_table:
        loss += "  The full table, one row per findable body (found or missed) with its nights per season, S/N and fitted orbit, is\n  `/app/history/LOSS.md`.\n"
    note = f"""

## This session: generation {int(gen[3:])} of a hill climb

{intro}

- `/app/history/SUMMARY.md` lists every earlier generation, best first, with its reward and main change;
  `/app/history/genNNN/` holds each one's code, `NOTES.md` and `reward.json`.
- The run always keeps the best program found so far{" (highest recall; at equal recall, fewer GPU-hours)" if DESIGN == 2 else ""}. A low score costs the run nothing and tells later generations what
  does not work, so a bold new method is welcome.
- Your own reward is given only to later generations. {judge}
{loss}- You have {AGENT_S // 3600} hours; the session is stopped at the limit and whatever is in `/app/submission` then is graded,
  so keep it valid: run `python3 /app/tools/check.py` before you finish.
- Before you finish, replace `/app/submission/NOTES.md`. Start it with three lines: `Method:` in a few words, `Base:` the
  generation(s) you built on (or none), `Change:` your main change in one sentence. Then what you tried, what you measured, and what
  you would try next.
"""
    open(os.path.join(out, "note.md"), "w").write(note)
    return out


def launch(RUN, VM_RUN, st, gen, label, seed, seed_gen=None, ag=None):
    out = handoff(RUN, st, gen, seed_gen)
    ssh(f"mkdir -p {VM_RUN}/incoming")
    rsync(out + "/", f"{VM}:{VM_RUN}/incoming/{gen}/")
    S = f"{VM_RUN}/stage/{gen}/decam-big-tno"
    script = f"""set -e
rm -rf {VM_RUN}/stage/{gen}; mkdir -p {S} {VM_RUN}/launched
rsync -a {VM_BASE}/ {S}/
mkdir -p {S}/environment/seed
rsync -a {shlex.quote(seed)}/ {S}/environment/seed/
rsync -a {VM_RUN}/incoming/{gen}/history/ {S}/environment/history/
printf 'COPY history /app/history\\nCOPY seed/ /app/submission/\\nRUN chmod -R a+rwX /app/submission && chmod -R a+rX /app\\n' >> {S}/environment/Dockerfile
cat {VM_RUN}/incoming/{gen}/note.md >> {S}/instruction.md
"""
    rc, o = ssh(script)
    if rc != 0:
        raise RuntimeError(f"staging {gen} failed: {o[-300:]}")
    mount = ""
    if DESIGN == 2 and OPEN:                              # open labelled regions, read-only (never declare networks here)
        y = f"{VM_RUN}/open_mount.yaml"
        rc, o = ssh(f"test -d {OPEN}/scen && test -d {OPEN}/labels && printf '%s\\n' 'services:' '  main:' '    volumes:' "
                    f"'      - \"{OPEN}:/app/open:ro\"' > {y}.tmp && mv {y}.tmp {y} && echo ok")
        if "ok" not in o:
            raise RuntimeError(f"open data folder {OPEN} is not complete on the host")
        mount = f"--extra-docker-compose {y} "
    ag = ag or PAIR[0]
    if ag["agent"] == "codex":                            # Codex through the relay (run_trial.sh picks the credential file); no web
        how = f"codex {ag['model']} {label} --ak reasoning_effort={ag.get('effort', 'xhigh')} --ak web_search=disabled --disable-verification {mount}".rstrip()
    else:
        ae = " ".join(f"--ae {k}={ag['model']}" for k in ("ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"))
        how = f"claude-code {ag['model']} {label} --ak reasoning_effort={ag.get('effort', 'xhigh')} --ak disallowed_tools=WebSearch,WebFetch --disable-verification {mount}{ae}"
    cmd = (f"mkdir {VM_RUN}/launched/{label} 2>/dev/null || {{ echo already; exit 0; }}; cd {W}; "
           f"(HARNESS=plain MAX_AGENT_SEC={AGENT_S} setsid nohup {W}/run_trial.sh {S} {label} {how} > {W}/logs/{label}.out 2>&1 < /dev/null &); echo started")
    rc, o = ssh(cmd)
    if rc != 0:
        raise RuntimeError(f"launch of {label} failed: {o[-300:]}")
    log(RUN, gen, "started", label, ag.get("name", ""))


def wait_end(RUN, gen, g):
    label = g["label"]; gone = 0
    while True:
        rc, o = ssh(f"grep -F ' {label} job=' {W}/logs/trials.log | tail -1")
        m = re.search(r"job=(\S+) harbor_exit=(\S+)", o)
        if m:
            return m.group(1), m.group(2)
        rc, o = ssh(f"pgrep -f '[_]_{label}__p' | head -1")
        gone = gone + 1 if not o.strip() else 0
        if gone >= 5:
            log(RUN, gen, label, "has no process and no end line")
            return None, None
        time.sleep(120)


def grade_gpu(RUN, VR, st, gen, g, sub, sids):
    """Run the submission on every sealed patch on Vista, copy outputs back, score here. Returns the grade dict.
    Resumable: the upload and each job id are kept in the state; run_gb.sbatch skips patches that already have a timing,
    so a resubmitted job finishes the rest. A round whose jobs never finish raises (retried later), never scores 0."""
    vr = f"{VR}/{gen}"
    if not g.get("uploaded"):
        ssh(f"rm -rf {vr}; mkdir -p {vr}/sub", vista=True)
        rsync(sub + "/", f"{VISTA}:{vr}/sub/", vista=True)
        ssh(f"printf '%s\\n' {' '.join(sids)} > {vr}/list.txt", vista=True)
        g["uploaded"] = True; save(RUN, st)
    lanes = 4
    wall = int((len(sids) + lanes - 1) // lanes * (BUDGET_S + 90) + 900)
    t = f"{wall // 3600:02d}:{wall % 3600 // 60:02d}:00"
    while True:
        job = (g.get("vista_jobs") or [None])[-1]
        rc, o = ssh(f"test -f {vr}/DONE && echo done; " + (f"squeue -h -j {job} -o %T 2>/dev/null" if job else "true"), vista=True)
        if "done" in o:
            break
        if job and o.strip():
            time.sleep(300); continue
        if len(g.get("vista_jobs", [])) >= 3:
            raise RuntimeError(f"{gen}: 3 Vista jobs ended without finishing; not graded")
        if job:
            log(RUN, gen, "Vista job", job, "ended without DONE; resubmitting for the patches left")
        rc, o = ssh(f"cd {VD} && sbatch -A $VISTA_ACCOUNT -t {t} harness/run_gb.sbatch {vr} {BUDGET_S} 2>&1 | tail -1", vista=True)
        m = re.search(r"Submitted batch job (\d+)", o)
        if not m:                                    # Vista gb takes 3 submitted jobs per user: wait; a refusal is not a grade
            log(RUN, gen, "sbatch refused, waiting:", o.strip()[-150:]); time.sleep(600); continue
        g.setdefault("vista_jobs", []).append(m.group(1)); save(RUN, st)
        log(RUN, gen, "grading on Vista, job", m.group(1), "wall", t)
    job = g["vista_jobs"][-1]
    local = os.path.join(RUN, "gens", gen, "vista")
    shutil.rmtree(local, ignore_errors=True); os.makedirs(local)
    for d in ("out", "timing", "log"):
        rsync(f"{VISTA}:{vr}/{d}/", os.path.join(local, d) + "/", vista=True)
    timing = {}
    for sid in sids:
        p = os.path.join(local, "timing", sid)
        if os.path.exists(p):
            timing[sid] = float(open(p).read().split()[0])
    json.dump(timing, open(os.path.join(local, "timing.json"), "w"))
    gd = os.path.join(RUN, "gens", gen, "grade")
    p = subprocess.run([NPY, "-I", os.path.join(HERE, "harness/grade.py"), os.path.join(local, "out"), TRUTH, gd, "--sids", ",".join(sids),
                        "--timing", os.path.join(local, "timing.json")], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"grade.py failed: {p.stderr[-400:]}")
    ssh(f"rm -rf {vr}/out {vr}/work", vista=True)    # candidates stay only on the Mac
    return json.load(open(os.path.join(gd, "grade.json"))), job


def vm_art(g, RUN, gen):
    """Path of a generation's submission on the trial host (the job's artifact)."""
    jobdir = os.path.join(RUN, "gens", gen, f"job-a{g['attempt']}")
    return f"{W}/jobs/{g['job']}/{os.path.relpath(g['submission'], jobdir)}"


def choose_seed(RUN, VM_RUN, st, n):
    """The program the next agent starts from. Pairs (2026-11-13): always the best graded program. Otherwise the most recently written generation while it is still
    being graded (pipelining: agents do not wait for grades), else the best graded program."""
    if NO_BASE:                                           # user, 2026-11-13: no base, every generation's code + reward
        ssh(f"mkdir -p {VM_RUN}/empty_seed && printf '%s\\n' 'Empty on purpose: there is no default program. Every earlier generation is in /app/history/genNNN/ (code, NOTES.md, reward.json); build yours here.' > {VM_RUN}/empty_seed/README.md")
        return f"{VM_RUN}/empty_seed", None
    if (st.get("workers") or 1) >= 2:
        rc, o = ssh(f"test -d {VM_RUN}/best && echo yes")
        return (f"{VM_RUN}/best" if "yes" in o else f"{VM_BASE}/solution/main"), None
    with LOCK:
        ung = [(v.get("written") or 0, int(k[3:]), k, v["vm_art"]) for k, v in st["gens"].items()
               if v.get("phase") == "grading" and v.get("vm_art") and int(k[3:]) < n]
    if ung:
        _, _, k, art = max(ung)
        return art, k
    rc, o = ssh(f"test -d {VM_RUN}/best && echo yes")
    return (f"{VM_RUN}/best" if "yes" in o else f"{VM_BASE}/solution/main"), None


def agent_phase(RUN, VM_RUN, st, n):
    """new -> running -> ended -> synced -> grading (or done when unchanged / invalid / nothing)."""
    gen = f"gen{n:03d}"
    gdir = os.path.join(RUN, "gens", gen); os.makedirs(gdir, exist_ok=True)
    g = st["gens"].setdefault(gen, dict(attempt=1, phase="new"))
    while g["phase"] in ("new", "running", "ended", "synced"):
        if g["phase"] == "new":
            g["label"] = f"dbt-{os.path.basename(RUN)}-{gen}-a{g['attempt']}"
            if AGENT_OVERRIDE and not g.get("agent"):
                g["agent"] = dict(AGENT_OVERRIDE); save(RUN, st)
            if DESIGN == 2 and (st.get("workers") or 1) >= 2 and not g.get("agent"):
                with LOCK:                                # a running generation without the field is a Fable agent (before pairs)
                    used = {(v.get("agent") or PAIR[0])["model"] for k, v in st["gens"].items()
                            if k != gen and (v.get("phase") == "running" or (v.get("phase") == "new" and v.get("agent")))}
                    g["agent"] = next((a for a in PAIR if a["model"] not in used), PAIR[n % len(PAIR)])
                save(RUN, st)
            rc, o = ssh(f"if [ -d {VM_RUN}/launched/{g['label']} ] || grep -qF 'label={g['label']} ' {W}/logs/trials.log; then echo yes; else echo no; fi")
            if o.strip() != "yes":
                seed, seed_gen = choose_seed(RUN, VM_RUN, st, n)
                g["seed"] = "none" if NO_BASE else (seed_gen or (st.get("best") or {}).get("gen") or "starter")
                launch(RUN, VM_RUN, st, gen, g["label"], seed, seed_gen, g.get("agent"))
            g["started"] = time.time(); g["phase"] = "running"; save(RUN, st)
        elif g["phase"] == "running":
            g["job"], g["harbor_exit"] = wait_end(RUN, gen, g); g["phase"] = "ended"; save(RUN, st)
        elif g["phase"] == "ended":
            jobdir = os.path.join(gdir, f"job-a{g['attempt']}")
            if g["job"]:
                ssh(f"sudo -n chmod -R a+rX {W}/jobs/{g['job']} 2>/dev/null; true")
                rc, o = ssh(f"test -d {W}/jobs/{g['job']} && echo yes")
                if o.strip() == "yes":
                    os.makedirs(jobdir, exist_ok=True); rsync(f"{VM}:{W}/jobs/{g['job']}/", jobdir + "/")
            g["phase"] = "synced"; save(RUN, st)
        elif g["phase"] == "synced":
            jobdir = os.path.join(gdir, f"job-a{g['attempt']}")
            arts = glob.glob(os.path.join(jobdir, "*", "artifacts", "submission"))
            sub = arts[0] if arts else None
            nothing = not sub or not os.path.isfile(os.path.join(sub, "finder.py"))
            ssh(f"docker images --format '{{{{.Repository}}}}:{{{{.Tag}}}}' | grep -F '{g['label']}' | xargs -r docker rmi >/dev/null 2>&1; true")
            ssh(f"rm -rf {VM_RUN}/stage/{gen} {VM_RUN}/incoming/{gen}")
            shutil.rmtree(os.path.join(RUN, "handoff", gen), ignore_errors=True)
            if nothing:
                log(RUN, gen, "attempt", g["attempt"], "left no finder.py; harbor_exit", g.get("harbor_exit"))
                if g["attempt"] >= MAX_ATTEMPTS:
                    raise SystemExit(f"{gen}: {MAX_ATTEMPTS} attempts left nothing; the service looks down. Stopping.")
                time.sleep(min(3600, 300 * g["attempt"])); g["attempt"] += 1; g["phase"] = "new"; save(RUN, st); continue
            h = tree_hash(sub); best = st.get("best")
            g["submission"], g["hash"] = sub, h
            g["vm_art"] = vm_art(g, RUN, gen)
            # 2026-11-19 (gen054, gen055): the agent's model stream dropped (Codex gave up after 5 quick retries) and the round
            # ended with the program unchanged; such a round is run again instead of counted (user: "another 10 rounds").
            died = False
            for cf in glob.glob(os.path.join(jobdir, "*", "agent", "codex.txt")):
                t = open(cf, errors="replace").read()[-20000:]
                died = died or ('"type":"turn.failed"' in t and "stream disconnected" in t)
            if died and best is not None and h == best["hash"] and g["attempt"] < MAX_ATTEMPTS:
                log(RUN, gen, "attempt", g["attempt"], "lost its model connection with the program unchanged; running the round again")
                time.sleep(60); g["attempt"] += 1; g["phase"] = "new"; save(RUN, st); continue
            with LOCK:
                same = [(k, v) for k, v in st["gens"].items() if k != gen and v.get("phase") == "done" and v.get("hash") == h
                        and v.get("reward") is not None and v.get("status") not in ("invalid",)]
            if best is not None and h == best["hash"]:
                g.update(status="unchanged", reward=best["reward"], reason=None, phase="done", grader=best.get("grader"))
                log(RUN, gen, "left the best program unchanged")
            elif same:
                k0, v0 = same[0]
                g.update(status="unchanged", reward=v0["reward"], reason=f"identical to {k0}", phase="done", grader=v0.get("grader"),
                         seconds=v0.get("seconds"))
                log(RUN, gen, f"identical to {k0}; takes its grade {v0['reward']:.4f} (no NERSC node)")
            elif (hits := scan(sub)):
                g.update(status="invalid", reward=0.0, reason="; ".join(hits[:5]), phase="done", grader="scan")
                log(RUN, gen, "scan:", g["reason"])
            else:
                g["phase"] = "grading"; g["written"] = time.time(); log(RUN, gen, "written; queued for grading on NERSC")
            save(RUN, st)


def keep_best(RUN, VM_RUN, st):
    """Best = highest reward among valid generations graded on NERSC; its program is copied to the host's best/."""
    with LOCK:
        cand = [(v["reward"], k) for k, v in st["gens"].items() if v.get("phase") == "done" and v.get("status") == "valid"
                and v.get("grader") == "nersc" and isinstance(v.get("reward"), (int, float))]
        if not cand:
            return
        gh = lambda k: st["gens"][k].get("gpu_hours") if st["gens"][k].get("gpu_hours") is not None else 9e9
        r, k = max(cand, key=(lambda x: (x[0], -gh(x[1]), -int(x[1][3:]))) if DESIGN == 2 else (lambda x: (x[0], -int(x[1][3:]))))
        old = st.get("best") or {}
        if old.get("gen") == k and old.get("grader") == "nersc":
            return
        v = st["gens"][k]
    src = f"{VM_BASE}/solution/main" if k == "gen000" else v.get("vm_art") or vm_art(v, RUN, k)
    rc, o = ssh(f"set -e; rm -rf {VM_RUN}/best.new; cp -r {shlex.quote(src)} {VM_RUN}/best.new; rm -rf {VM_RUN}/best; mv {VM_RUN}/best.new {VM_RUN}/best; chmod -R a+rX {VM_RUN}/best")
    if rc != 0:
        raise RuntimeError(f"could not keep {k} as the best on the host: {o[-200:]}")
    with LOCK:
        st["best"] = dict(gen=k, reward=r, hash=v.get("hash"), submission=v.get("submission"), grader="nersc")
        if DESIGN == 2:
            st["best"]["gpu_hours"] = v.get("gpu_hours")
    save(RUN, st)
    log(RUN, k, f"is the best ({r:.4f}" + (f", {v.get('gpu_hours')} GPU-hours" if DESIGN == 2 else "") + ")" + (f", was {old.get('gen')} {old.get('reward')}" if old else ""))


def grade_one(RUN, VM_RUN, st, gen, sids):
    """Grade a written generation on NERSC (with any earlier valid generation not yet graded there, so every reward the
    best is chosen from comes from the same machine), then keep the best and send the leads on. Never scores 0 on an
    infrastructure failure: an exception leaves the generation in 'grading' and the worker retries."""
    with LOCK:
        g = st["gens"][gen]
        redo = [k for k, v in sorted(st["gens"].items()) if v.get("phase") == "done" and v.get("status") == "valid" and v.get("grader") != "nersc"]
        items = [(gen, g["submission"], os.path.join(RUN, "gens", gen, "grade"))]
        for k in redo:
            v = st["gens"][k]
            sub = v.get("submission") or os.path.join(HERE, "harness/task/starter")   # design 2: gen000 is an earlier run's program, not the starter
            items.append((k, sub, os.path.join(RUN, "gens", k, "grade_nersc")))
    log(RUN, gen, "grading on NERSC" + (f" with regrades of {', '.join(redo)}" if redo else ""))
    res = nersc_grade.grade(f"{os.path.basename(RUN)}-{gen}", items, sids, BUDGET_S, lambda m: log(RUN, gen, m))
    with LOCK:
        for k, d in res.items():
            v = st["gens"][k]
            if k != gen:
                v.setdefault("reward_vista", v.get("reward"))
                v["grade_dir"] = os.path.join(RUN, "gens", k, "grade_nersc")
            v.update(reward=d["reward"], grader="nersc", leads=len(d["leads"]), seconds={s: x.get("seconds") for s, x in d["per_scenario"].items()})
            if DESIGN == 2:
                v["gpu_hours"] = d.get("gpu_hours")
            if k == gen:
                v.update(status="valid", reason=None, phase="done")
            log(RUN, k, f"reward {d['reward']:.4f} (NERSC A100)", *((f"gpu_hours {d.get('gpu_hours')}",) if DESIGN == 2 else ()), "leads", len(d["leads"]),
                {s: (x["known_found"], x["known_findable"], round(x.get("seconds") or 0)) for s, x in d["per_scenario"].items()})
    save(RUN, st)
    keep_best(RUN, VM_RUN, st)
    try:                                                  # leads -> novelty check; never stops the climb
        with LEADS_LOCK:                                  # two graders: one leads.py at a time (shared ledger)
            p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "harness/leads.py"), os.path.basename(RUN), gen],
                               capture_output=True, text=True, timeout=900)
        log(RUN, gen, "leads:", (p.stdout + p.stderr).strip()[-300:])
    except Exception as e:
        log(RUN, gen, "leads failed:", type(e).__name__, str(e)[:200])


def grader_worker(RUN, VM_RUN, st, sids, q, stop):
    while not (stop.is_set() and q.empty()):
        try:
            gen = q.get(timeout=30)
        except queue.Empty:
            continue
        while True:
            try:
                grade_one(RUN, VM_RUN, st, gen, sids); break
            except Exception as e:
                log(RUN, gen, "grading error, retrying in 10 min:", type(e).__name__, str(e)[:300]); time.sleep(600)
        q.task_done()


def agent_worker(RUN, VM_RUN, st, n, q, stop):
    """One generation's agent, from launch to written; errors retried every 5 min (an infrastructure failure never
    scores 0)."""
    gen = f"gen{n:03d}"; errors = 0
    while not stop.is_set():
        try:
            agent_phase(RUN, VM_RUN, st, n)
            if st["gens"][gen]["phase"] == "grading":
                q.put(gen)
            return
        except SystemExit as e:
            log(RUN, gen, "agent stopped:", str(e)[:300]); return
        except Exception as e:
            errors += 1; log(RUN, "error", errors, gen, type(e).__name__, str(e)[:500])
            if errors >= 10:
                log(RUN, gen, "10 errors in a row; this generation is left for a restart"); return
            time.sleep(300)


def run(run_id, target, goal=None, workers=None):
    """Generations in pairs (2026-11-13): `workers` agents start together from the best graded program; the
    next pair starts when all of them have written, from the best graded program then. Every program is graded, up to
    `workers` NERSC nodes at once; a running agent found at a restart is resumed as part of the current pair."""
    RUN, VM_RUN, VR = paths(run_id)
    os.makedirs(RUN, exist_ok=True)
    configure(run_id)
    goal = goal if goal is not None else (9.0 if DESIGN == 2 else 0.5)      # design 2 has no stopping score
    workers = workers or (1 if DESIGN == 2 else 2)
    if not os.path.exists(os.path.join(RUN, "GO")):
        sys.exit(f"no {RUN}/GO: rounds start only on the user's go (write the date and their words into GO)")
    cfg = json.load(open(os.path.join(RUN, "config.json")))
    wait_grade = DESIGN == 2 and cfg.get("wait_grade", True)   # the next agent starts after the previous grade: it sees every outcome
    lock = open(os.path.join(RUN, "climb.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another climb on this run is going")
    st = load(RUN); st["target"] = target; st["goal"] = goal; st["workers"] = workers; save(RUN, st)
    log(RUN, "run", run_id, "target", target, "goal", goal, "workers", workers, "grading on NERSC, pipelined", "best", st.get("best"))
    q, stop = queue.Queue(), threading.Event()
    graders = [threading.Thread(target=grader_worker, args=(RUN, VM_RUN, st, cfg["sids"], q, stop), daemon=True) for _ in range(workers)]
    for t in graders:
        t.start()
    for k, v in sorted(st["gens"].items()):              # resume: generations written but not graded
        if v.get("phase") == "grading":
            q.put(k)
    keep_best(RUN, VM_RUN, st)                            # a best chosen just before a stop may not have been saved
    agents, n, starting = {}, 1, False
    while n <= st["target"]:
        b = st.get("best") or {}
        if b.get("grader") == "nersc" and (b.get("reward") or 0) >= goal:
            log(RUN, f"goal reached: {b['gen']} {b['reward']:.4f} >= {goal}"); break
        gen = f"gen{n:03d}"
        ph = st["gens"].get(gen, {}).get("phase")
        if ph in ("done", "grading"):
            n += 1; continue
        if ph in (None, "new") and os.path.exists(os.path.join(RUN, "STOP")):
            log(RUN, "STOP file found, stopping before", n); break
        agents = {k: t for k, t in agents.items() if t.is_alive()}
        with LOCK:
            waiting = sum(1 for v in st["gens"].values() if v.get("phase") == "grading")
        pair2 = DESIGN == 2 and workers >= 2
        busy = (len(agents) >= workers) if ((NO_BASE and not pair2) or ph not in (None, "new")) else (bool(agents) and not starting)
        if busy or (ph in (None, "new") and (waiting > workers or (wait_grade and waiting))):
            time.sleep(30); continue
        starting = ph in (None, "new") and len(agents) + 1 < workers   # a pair starts together, from the same best
        if pair2 and ph == "running" and len(agents) + 1 < workers and time.time() - (st["gens"][gen].get("started") or 0) < AGENT_S / 2:
            starting = True                               # a lone agent found early in its session at a restart gets its partner
        with LOCK:
            st["gens"].setdefault(gen, dict(attempt=1, phase="new"))
        t = threading.Thread(target=agent_worker, args=(RUN, VM_RUN, st, n, q, stop), daemon=True); t.start()
        agents[gen] = t; n += 1
        time.sleep(20)                                    # let the launch record its seed before the next one
    for t in agents.values():
        t.join()
    stop.set(); q.join()
    for t in graders:
        t.join(timeout=60)
    log(RUN, "loop ended; best", st.get("best"))


def starter(run_id, grade_dir, sub=None):
    """Record the starter as gen000 from a finished grade (grade.json of the starter on the run's sealed regions).
    --sub: the program's folder (default harness/task/starter); design 2 starts from an earlier run's best program."""
    RUN, _, _ = paths(run_id); os.makedirs(RUN, exist_ok=True); st = load(RUN); configure(run_id)
    d = json.load(open(os.path.join(grade_dir, "grade.json")))
    sub = os.path.abspath(sub) if sub else os.path.join(HERE, "harness/task/starter")
    assert os.path.exists(os.path.join(sub, "finder.py")), sub
    st["gens"]["gen000"] = dict(phase="done", status="valid", reward=d["reward"], reason=None, submission=sub, hash=tree_hash(sub),
                                grade_dir=os.path.abspath(grade_dir), gpu_hours=d.get("gpu_hours"),
                                leads=len(d["leads"]), seconds={s: v.get("seconds") for s, v in d["per_scenario"].items()})
    if not st.get("best") or st["best"]["gen"] == "gen000":
        st["best"] = dict(gen="gen000", reward=d["reward"], gpu_hours=d.get("gpu_hours"), hash=tree_hash(sub), submission=sub)
    save(RUN, st); print("gen000", d["reward"], d.get("gpu_hours"))


def status(run_id):
    RUN, _, _ = paths(run_id); st = load(RUN); configure(run_id)
    print(f"target {st['target']}, best {st.get('best') and {k: st['best'][k] for k in ('gen', 'reward')}}, "
          f"GO {os.path.exists(os.path.join(RUN, 'GO'))}, STOP {os.path.exists(os.path.join(RUN, 'STOP'))}")
    for k, v in sorted(st["gens"].items()):
        r = v.get("reward")
        print(f"{k:8}{v['phase']:9}{v.get('status', ''):11}{(f'{r:.4f}' if isinstance(r, float) else ''):>8}{str(v.get('leads', '')):>6}  "
              f"{note_head(v.get('submission')).get('Method', '')[:70]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run", "status", "starter"))
    ap.add_argument("--run", default="r1"); ap.add_argument("--target", type=int, default=20); ap.add_argument("--grade")
    ap.add_argument("--goal", type=float); ap.add_argument("--workers", type=int); ap.add_argument("--sub")
    a = ap.parse_args()
    {"run": lambda: run(a.run, a.target, a.goal, a.workers), "status": lambda: status(a.run), "starter": lambda: starter(a.run, a.grade, a.sub)}[a.command]()
