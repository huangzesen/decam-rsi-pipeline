#!/usr/bin/env python3
"""Port the decam-big-tno best program to one GPU device for maximum efficiency (2026-11-17).

The flow is one command per device:

    python3 -I tailor/tailor.py port --target gb200-vista --agent astra [--dry]

= ONE Harbor trial on the trial host (2 h, CPU only, no web, the r2 task with the all-open regions mounted, exactly as the
r2 climb's agents run), starting from the hill climb's best program (runs/r2/state.json), with the r2 instruction plus
harness/task_r2/instruction_hw.md: the device's facts from tailor/targets.json and the program's own timings and log
tails on the A100. Single goal: least GPU time on that device. The only test is the task's CPU sample check
(tools/check.py). The result goes to tailor/ports/<target>/: program/ (the port, with BUDGET_S.txt) and PORT.md (what was
changed for the device and the expected effect); an earlier port moves to tailor/ports/<target>/old/. --dry writes the
brief to tailor/ports/<target>/brief/ and starts nothing. Agents: tailor/agents.json (astra, fable). The command blocks
for the session (~2 h): run it with nohup.

    python3 -I tailor/tailor.py status                 every port and running session
    python3 -I tailor/tailor.py table                  one row per device -> tailor/efficiency.json

Optional tools, NOT part of the flow (they spend allocation; only on the user's explicit word):
    python3 -I tailor/tailor.py cost [--target T]      SU per GPU grading of a port
    python3 -I tailor/tailor.py grade --target T --go "approvals" [--budget S] [--dry]
        runs tailor/ports/<target>/program on the device (the r2 four graded regions, its declared BUDGET_S, no network),
        scores it on the Mac; pass = recall >= the base's A100 recall; GPU-seconds, GPU-h and SU per 1,000 exposures
        go into the table. Needs the regions on the device (tailor/stage.py) and, on Lonestar6, the env (tailor/ls6_env.sh).
"""
import argparse, glob, json, math, os, re, shlex, shutil, subprocess, sys, time

T = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(T)
sys.path.insert(0, os.path.join(HERE, "harness"))
import climb as C           # noqa: E402  (trial host helpers; its globals are set below, its own runs are untouched)
import nersc_grade          # noqa: E402

TARGETS = json.load(open(os.path.join(T, "targets.json")))
AGENTS = json.load(open(os.path.join(T, "agents.json")))
REGIONS = {k: v for k, v in json.load(open(os.path.join(T, "regions.json"))).items() if not k.startswith("_")}
R2 = os.path.join(HERE, "runs", "r2")
PORTS = os.path.join(T, "ports")
NPY = os.path.expanduser("~/anaconda3/bin/python")
ADDENDUM = os.path.join(HERE, "harness/task_r2/instruction_hw.md")
AGENT_S, LOG_LINES, MAX_BUDGET = 7200, 300, 1800


def log(where, *a):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    os.makedirs(where, exist_ok=True)
    with open(os.path.join(where, "tailor.log"), "a") as f:
        f.write(line + "\n")


C.log = log                                                 # climb.wait_end logs through this


def r2():
    cfg = json.load(open(os.path.join(R2, "config.json"))); st = json.load(open(os.path.join(R2, "state.json")))
    b = st["best"]; g = st["gens"][b["gen"]]
    return cfg, dict(gen=b["gen"], recall=b["reward"], gpu_hours=b.get("gpu_hours"), seconds=g.get("seconds"), submission=b["submission"],
                     runs=os.path.join(R2, "gens", b["gen"], "grade_runs"), grade=os.path.join(R2, "gens", b["gen"], "grade"))


def target(key):
    t = TARGETS.get(key)
    if not t or key.startswith("_"):
        sys.exit(f"unknown target {key}; known: {', '.join(k for k in TARGETS if not k.startswith('_'))}")
    if not t.get("enabled"):
        sys.exit(f"{key} is disabled: {t.get('note')}")
    return dict(t, key=key)


def tail(path, n):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except OSError:
        return ""


def declared_budget(prog):
    p = os.path.join(prog, "BUDGET_S.txt")
    if not os.path.exists(p):
        return MAX_BUDGET, None
    try:
        b = int(open(p).read().split()[0])
    except Exception:
        return None, "BUDGET_S.txt does not hold an integer"
    return (b, None) if 60 <= b <= MAX_BUDGET else (None, f"BUDGET_S.txt {b} is outside 60..{MAX_BUDGET}")


# ---------------------------------------------------------------- the brief and the session
def hardware_text(t):
    return "\n".join([f"- GPU: {t['gpu']}", f"- CPU: {t['cpu']}" + (f"; host memory {t['memory_host_gb']} GB per node" if t.get("memory_host_gb") else ""),
                      f"- Software: {t['python']}; CUDA {t['cuda']}" + (f"; modules {t['modules']}" if t.get("modules") else ""),
                      f"- Layout: {t['gpus_per_node']} GPU(s) per node; one region per GPU; charge {t['charge']}"])


def brief(t, ag, base, out):
    """/app/history (the base's A100 timings and log tails, the device) and the filled instruction addendum."""
    shutil.rmtree(out, ignore_errors=True)
    hist = os.path.join(out, "history"); os.makedirs(os.path.join(hist, "a100", "log"))
    json.dump(dict(generation=base["gen"], gpu="A100 40 GB (NERSC Perlmutter), one per region", recall=base["recall"], gpu_hours=base["gpu_hours"],
                   seconds_per_region=base["seconds"], budget_s=MAX_BUDGET, regions=REGIONS), open(os.path.join(hist, "a100", "reward.json"), "w"), indent=1)
    for sid in REGIONS:
        lg = tail(os.path.join(base["runs"], "log", sid + ".log"), LOG_LINES)
        if lg:
            open(os.path.join(hist, "a100", "log", sid + ".log"), "w").write(lg)
    pub = {k: v for k, v in t.items() if k not in ("sock", "host", "account", "root", "key")}
    open(os.path.join(hist, "HARDWARE.md"), "w").write(f"# {t['name']}\n\n```json\n{json.dumps(pub, indent=1)}\n```\n")
    note = open(ADDENDUM).read()
    for a, b in (("TARGET_NAME", t["name"]), ("BASE_GEN", base["gen"]), ("BASE_RECALL", f"{base['recall']:.4f}"),
                 ("BASE_GPUH", f"{base['gpu_hours']}"), ("LOG_LINES", str(LOG_LINES)), ("MAX_BUDGET", str(MAX_BUDGET)),
                 ("AGENT_H", f"{AGENT_S / 3600:.0f}"), ("HARDWARE_TEXT", hardware_text(t))):
        note = note.replace(a, b)
    open(os.path.join(out, "note.md"), "w").write(note)
    return out


def launch(where, t, ag, cfg2, label, seed, out):
    """Stage the r2 task on the trial host with this brief and the base program, then start the Harbor trial."""
    VM_RUN = f"{C.W}/tasks-ale/runs/dbt-port-{t['key']}"
    C.ssh(f"mkdir -p {VM_RUN}/incoming {VM_RUN}/seeds {VM_RUN}/launched")
    C.rsync(out + "/", f"{C.VM}:{VM_RUN}/incoming/{label}/")
    C.rsync(seed.rstrip("/") + "/", f"{C.VM}:{VM_RUN}/seeds/{label}/")
    S = f"{VM_RUN}/stage/{label}/decam-big-tno"
    rc, o = C.ssh(f"""set -e
rm -rf {VM_RUN}/stage/{label}; mkdir -p {S}
rsync -a {cfg2['vm_base']}/ {S}/
mkdir -p {S}/environment/seed
rsync -a {VM_RUN}/seeds/{label}/ {S}/environment/seed/
rsync -a {VM_RUN}/incoming/{label}/history/ {S}/environment/history/
printf 'COPY history /app/history\\nCOPY seed/ /app/submission/\\nRUN chmod -R a+rwX /app/submission && chmod -R a+rX /app\\n' >> {S}/environment/Dockerfile
cat {VM_RUN}/incoming/{label}/note.md >> {S}/instruction.md
""")
    if rc != 0:
        raise RuntimeError(f"staging failed: {o[-300:]}")
    y = f"{VM_RUN}/open_mount.yaml"
    rc, o = C.ssh(f"test -d {cfg2['open']}/scen && test -d {cfg2['open']}/labels && printf '%s\\n' 'services:' '  main:' '    volumes:' "
                  f"'      - \"{cfg2['open']}:/app/open:ro\"' > {y}.tmp && mv {y}.tmp {y} && echo ok")
    if "ok" not in o:
        raise RuntimeError(f"open data folder {cfg2['open']} is not complete on the host")
    mount = f"--extra-docker-compose {y}"
    if ag["agent"] == "codex":
        how = f"codex {ag['model']} {label} --ak reasoning_effort={ag['effort']} --ak web_search=disabled --disable-verification {mount}"
    else:
        ae = " ".join(f"--ae {k}={ag['model']}" for k in ("ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"))
        how = f"claude-code {ag['model']} {label} --ak reasoning_effort={ag['effort']} --ak disallowed_tools=WebSearch,WebFetch --disable-verification {mount} {ae}"
    rc, o = C.ssh(f"mkdir {VM_RUN}/launched/{label} 2>/dev/null || {{ echo already; exit 0; }}; cd {C.W}; "
                  f"(HARNESS=plain MAX_AGENT_SEC={AGENT_S} setsid nohup {C.W}/run_trial.sh {S} {label} {how} > {C.W}/logs/{label}.out 2>&1 < /dev/null &); echo started")
    if rc != 0:
        raise RuntimeError(f"launch of {label} failed: {o[-300:]}")
    log(where, label, "started", ag["name"], "on", t["name"])
    return VM_RUN


def port(key, agent, dry=False):
    t = target(key); ag = AGENTS.get(agent)
    if not ag or agent.startswith("_"):
        sys.exit(f"unknown agent {agent}; known: {', '.join(k for k in AGENTS if not k.startswith('_'))}")
    cfg2, base = r2()
    C.VM_BASE, C.OPEN, C.DESIGN, C.AGENT_S = cfg2["vm_base"], cfg2["open"], 2, AGENT_S
    where = os.path.join(PORTS, key); os.makedirs(where, exist_ok=True)
    stp = os.path.join(where, "session.json")
    s = json.load(open(stp)) if os.path.exists(stp) else {}
    if s.get("phase") in (None, "done"):
        if dry:
            out = brief(t, ag, base, os.path.join(where, "brief"))
            print(f"[dry] brief for {t['name']} x {ag['name']} in {out} (note.md, history/); base {base['gen']} "
                  f"(recall {base['recall']:.4f}, {base['gpu_hours']} GPU-h on the A100); nothing staged or started"); return
        label = f"dbt-port-{key}-{agent}-{time.strftime('%m%d%H%M')}"
        s = dict(label=label, target=key, agent=ag, base=base, phase="new", started=None); json.dump(s, open(stp, "w"), indent=1)
    label = s["label"]
    while s["phase"] != "done":
        if s["phase"] == "new":
            out = brief(t, ag, base, os.path.join(where, "brief"))
            s["vm_run"] = launch(where, t, ag, cfg2, label, base["submission"], out); s["phase"] = "running"; s["started"] = time.time()
        elif s["phase"] == "running":
            s["job"], s["harbor_exit"] = C.wait_end(where, label, dict(label=label)); s["phase"] = "ended"; s["ended"] = time.time()
        elif s["phase"] == "ended":
            jobdir = os.path.join(where, "sessions", label)
            if s["job"]:
                C.ssh(f"sudo -n chmod -R a+rX {C.W}/jobs/{s['job']} 2>/dev/null; true")
                os.makedirs(jobdir, exist_ok=True); C.rsync(f"{C.VM}:{C.W}/jobs/{s['job']}/", jobdir + "/")
            C.ssh(f"docker images --format '{{{{.Repository}}}}:{{{{.Tag}}}}' | grep -F '{label}' | xargs -r docker rmi >/dev/null 2>&1; "
                  f"rm -rf {s['vm_run']}/stage/{label} {s['vm_run']}/incoming/{label} {s['vm_run']}/seeds/{label}; true")
            s["phase"] = "synced"
        elif s["phase"] == "synced":
            arts = glob.glob(os.path.join(where, "sessions", label, "*", "artifacts", "submission"))
            sub = arts[0] if arts else None
            if not sub or not os.path.isfile(os.path.join(sub, "finder.py")):
                s.update(phase="done", result="no program left (harbor exit %s)" % s.get("harbor_exit")); log(where, label, s["result"])
            else:
                hits = C.scan(sub); bud, err = declared_budget(sub)
                if os.path.isdir(os.path.join(where, "program")):          # keep the earlier port
                    old = os.path.join(where, "old", time.strftime("%Y%m%d-%H%M%S")); os.makedirs(old)
                    for f in ("program", "PORT.md", "port.json"):
                        if os.path.exists(os.path.join(where, f)):
                            shutil.move(os.path.join(where, f), os.path.join(old, f))
                shutil.copytree(sub, os.path.join(where, "program"), ignore=shutil.ignore_patterns("__pycache__"))
                pm = os.path.join(sub, "PORT.md")
                shutil.copy(pm if os.path.exists(pm) else os.path.join(sub, "NOTES.md") if os.path.exists(os.path.join(sub, "NOTES.md")) else os.devnull,
                            os.path.join(where, "PORT.md"))
                meta = dict(target=key, device=t["name"], agent=ag, base=dict(gen=base["gen"], recall_a100=base["recall"], gpu_hours_a100=base["gpu_hours"]),
                            label=label, session_min=round((s["ended"] - s["started"]) / 60) if s.get("started") and s.get("ended") else None,
                            budget_s=bud, budget_problem=err, scan=hits, hash=C.tree_hash(sub), made=time.strftime("%Y-%m-%d %H:%M"),
                            tested="CPU sample check only (tools/check.py); not run on the device")
                json.dump(meta, open(os.path.join(where, "port.json"), "w"), indent=1)
                s.update(phase="done", result="port written" + (f"; scan: {hits[:3]}" if hits else "") + (f"; {err}" if err else ""))
                log(where, label, s["result"], "->", os.path.join(where, "program"))
        json.dump(s, open(stp, "w"), indent=1)


# ---------------------------------------------------------------- table
def table():
    _, base = r2()
    exp = sum(v["exposures"] for v in REGIONS.values()); det = sum(v["detections"] for v in REGIONS.values())
    rows = []
    for key, t in TARGETS.items():
        if key.startswith("_"):
            continue
        where = os.path.join(PORTS, key); m = json.load(open(os.path.join(where, "port.json"))) if os.path.exists(os.path.join(where, "port.json")) else None
        g = json.load(open(os.path.join(where, "grade", "efficiency.json"))) if os.path.exists(os.path.join(where, "grade", "efficiency.json")) else None
        rows.append(dict(target=key, device=t["name"], enabled=bool(t.get("enabled")), port=bool(m), agent=m and m["agent"]["name"], made=m and m["made"],
                         base=m and m["base"]["gen"], budget_s=m and m["budget_s"], measured=bool(g),
                         gpu_s_4_regions=g and g["gpu_seconds"], s_per_region=g and round(g["gpu_seconds"] / len(REGIONS)),
                         gpu_h_per_1000_exp=g and round(g["gpu_seconds"] / 3600 / (exp / 1000), 4),
                         su_per_1000_exp=g and t.get("su_per_gpu_hour") and round(g["gpu_seconds"] / 3600 / (exp / 1000) * t["su_per_gpu_hour"], 4),
                         gpu_h_per_million_rows=g and round(g["gpu_seconds"] / 3600 / (det / 1e6), 5),
                         recall=g and g["recall"], passes=g and g["passes"]))
    a100 = sum(x for x in base["seconds"].values() if x)
    ref = dict(device="A100 40 GB (NERSC), base unchanged", base=base["gen"], budget_s=MAX_BUDGET, gpu_s_4_regions=round(a100),
               s_per_region=round(a100 / len(REGIONS)), gpu_h_per_1000_exp=round(a100 / 3600 / (exp / 1000), 4),
               su_per_1000_exp=round(a100 / 3600 / (exp / 1000) * TARGETS["a100-nersc"]["su_per_gpu_hour"], 4),
               gpu_h_per_million_rows=round(a100 / 3600 / (det / 1e6), 5), recall=base["recall"])
    out = dict(made=time.strftime("%Y-%m-%d %H:%M"), regions=dict(n=len(REGIONS), exposures=exp, detection_rows=det),
               reference=ref, pass_mark=f"recall >= {base['recall']:.4f} (base on the A100 at 1,800 s)", devices=rows)
    json.dump(out, open(os.path.join(T, "efficiency.json"), "w"), indent=1)
    print(f"reference: {ref['device']} {ref['base']}: {ref['s_per_region']} s/region, {ref['gpu_h_per_1000_exp']} GPU-h and {ref['su_per_1000_exp']} SU "
          f"(NERSC GPU node-h) per 1,000 exposures, recall {ref['recall']}")
    print(f"{'device':14}{'port':>6}{'agent':>17}{'budget':>8}{'s/region':>10}{'GPU-h/1kexp':>13}{'SU/1kexp':>10}{'recall':>8}{'pass':>6}")
    for r in rows:
        f = lambda v: "-" if v in (None, False) else str(v)
        print(f"{r['target']:14}{('yes' if r['port'] else 'no'):>6}{f(r['agent']):>17}{f(r['budget_s']):>8}{f(r['s_per_region']):>10}"
              f"{f(r['gpu_h_per_1000_exp']):>13}{f(r['su_per_1000_exp']):>10}{f(r['recall']):>8}{f(r['passes']):>6}"
              + ("" if r["enabled"] else "   (disabled)") + ("" if r["measured"] or not r["port"] else "   (not run on the device)"))
    print(f"written {os.path.join(T, 'efficiency.json')}")


def status():
    for key in sorted(k for k in TARGETS if not k.startswith("_")):
        where = os.path.join(PORTS, key)
        s = json.load(open(os.path.join(where, "session.json"))) if os.path.exists(os.path.join(where, "session.json")) else None
        m = json.load(open(os.path.join(where, "port.json"))) if os.path.exists(os.path.join(where, "port.json")) else None
        print(f"{key:14} session: {s and (s['label'] + ' ' + s['phase'] + (' ' + s.get('result', '') if s.get('result') else '')) or 'none'}; "
              f"port: {m and (m['agent']['name'] + ' ' + m['made'] + ', budget ' + str(m['budget_s'])) or 'none'}")


# ---------------------------------------------------------------- optional: GPU grading of a port (not part of the flow)
def cost(key=None):
    _, base = r2(); secs = [s for s in base["seconds"].values() if s]
    print(f"optional GPU grading of a port: the four r2 graded regions at the port's budget; at 1,800 s (base on the A100: "
          f"{', '.join(f'{s:.0f}' for s in secs)} s per region):")
    for k, t in TARGETS.items():
        if k.startswith("_") or (key and k != key):
            continue
        if not t.get("enabled"):
            print(f"  {k}: disabled ({t.get('note', '')})"); continue
        lanes = t["nodes"] * t["gpus_per_node"]; waves = math.ceil(len(secs) / lanes)
        wall = sum(sorted(secs, reverse=True)[i * lanes] for i in range(waves)) / 3600 + 0.1
        nh = max(t.get("min_hours", 0), wall) * t["nodes"]
        print(f"  {k}: {t['nodes']} node(s) x {t['gpus_per_node']} GPU, {waves} wave(s), ~{wall:.2f} h, {nh:.2f} node-h, ~{nh * t['su_per_node_hour']:.1f} SU")


def tssh(t, cmd, tries=30):
    for i in range(tries):
        p = subprocess.run(["ssh", "-S", os.path.expanduser(t["sock"]), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                            t["host"], "export LC_ALL=C LANG=C; " + cmd], capture_output=True, text=True, timeout=600)
        if p.returncode != 255:
            return p.returncode, p.stdout + p.stderr
        time.sleep(300)
    raise RuntimeError(f"{t['machine']} cannot be reached (no SSH master)")


def trsync(t, src, dst):
    e = f"ssh -S {os.path.expanduser(t['sock'])} -o ControlMaster=no -o BatchMode=yes"
    for i in range(10):
        p = subprocess.run(["rsync", "-a", "--partial", "-e", e, src, dst], capture_output=True, text=True)
        if p.returncode == 0:
            return
        time.sleep(60)
    raise RuntimeError(f"rsync {src} {dst}: {p.stderr[-300:]}")


def grade(key, go, budget=None, dry=False):
    """Run tailor/ports/<target>/program on the device and score it. Spends allocation: only on the user's explicit word."""
    t = target(key); where = os.path.join(PORTS, key); prog = os.path.join(where, "program")
    if not os.path.isdir(prog):
        sys.exit(f"no port for {key} yet ({prog})")
    if not dry and not go:
        sys.exit("grade spends allocation: pass --go with approvals")
    cfg2, base = r2(); sids = list(REGIONS)
    bud, err = declared_budget(prog)
    if err:
        sys.exit(err)
    bud = budget or bud
    gd = os.path.join(where, "grade"); runs = gd + "_runs"; tag = time.strftime("%m%d%H%M")
    if t["site"] == "nersc":
        job = f"port-{key}-{tag}"
        if dry:
            print(f"[dry] {key}: harness/nersc_grade.grade({job!r}, [('port', {prog!r}, {gd!r})], {sids}, {bud}) on one interactive node; nothing submitted"); return
        nersc_grade.TRUTH = os.path.join(HERE, cfg2["truth"])
        d = nersc_grade.grade(job, [("port", prog, gd)], sids, bud, lambda m: log(where, m))["port"]
        timing = json.load(open(os.path.join(gd + "_runs", "timing.json")))
    else:
        R = f"{t['root']}/port-{key}/{tag}"
        lanes = t["nodes"] * t["gpus_per_node"]; wall = math.ceil(len(sids) / lanes) * (bud + 150) + 900
        cmd = (f"cd {R} && sbatch -A {t['account']} -p {t['partition']} -N {t['nodes']} -t {wall // 3600:02d}:{wall % 3600 // 60:02d}:00 -o {R}/job-%j.out "
               f"{R}/tacc_grade.sbatch {R} {bud} {t['gpus_per_node']} {t['scen']} {t['env']} {shlex.quote(t.get('modules', ''))}")
        if dry:
            print(f"[dry] {key}: upload {prog} -> {t['host']}:{R}/sub/, tailor/sites/tacc_grade.sbatch, list.txt ({' '.join(sids)}); submit: {cmd}; nothing submitted"); return
        rc, o = tssh(t, "for s in " + " ".join(sids) + f"; do test -f {t['scen']}/$s/exposures.json || echo missing $s; done")
        if "missing" in o:
            sys.exit(f"regions not on {t['machine']} ({o.strip()}): tailor/stage.py --target {key} --go first")
        tssh(t, f"mkdir -p {R}/sub"); trsync(t, prog + "/", f"{t['host']}:{R}/sub/")
        trsync(t, os.path.join(T, "sites", "tacc_grade.sbatch"), f"{t['host']}:{R}/tacc_grade.sbatch")
        tssh(t, f"printf '%s\\n' {' '.join(sids)} > {R}/list.txt")
        rc, o = tssh(t, cmd + " 2>&1 | tail -2"); m = re.search(r"Submitted batch job (\d+)", o)
        if not m:
            sys.exit(f"sbatch refused: {o.strip()[-200:]}")
        log(where, f"grading the port on {key}, job {m.group(1)}, budget {bud} s ({go})")
        while True:
            rc, o = tssh(t, f"test -f {R}/DONE && echo done; test -f {R}/NO_ISOLATION && echo noiso; squeue -h -j {m.group(1)} -o %T")
            if "noiso" in o:
                sys.exit(f"a {key} node could not run without network ({R}/probe/): not graded")
            if "done" in o:
                break
            if not o.strip():
                sys.exit(f"job {m.group(1)} ended without DONE: see {R}/job-{m.group(1)}.out")
            time.sleep(300)
        shutil.rmtree(runs, ignore_errors=True); os.makedirs(runs)
        for d_ in ("out", "timing", "log", "probe"):
            trsync(t, f"{t['host']}:{R}/{d_}/", os.path.join(runs, d_) + "/")
        timing = {s: float(open(os.path.join(runs, "timing", s)).read().split()[0]) for s in sids if os.path.exists(os.path.join(runs, "timing", s))}
        json.dump(timing, open(os.path.join(runs, "timing.json"), "w"))
        p = subprocess.run([NPY, "-I", os.path.join(HERE, "harness/grade.py"), os.path.join(runs, "out"), os.path.join(HERE, cfg2["truth"]), gd,
                            "--sids", ",".join(sids), "--timing", os.path.join(runs, "timing.json")], capture_output=True, text=True)
        if p.returncode != 0:
            sys.exit(f"grade.py failed: {p.stderr[-400:]}")
        d = json.load(open(os.path.join(gd, "grade.json")))
        tssh(t, f"rm -rf {R}/out {R}/work")
    gs = sum(timing.values())
    eff = dict(target=key, budget_s=bud, gpu_seconds=round(gs), seconds=timing, recall=d["reward"], passes=d["reward"] >= base["recall"] - 1e-9,
               pass_mark=base["recall"], graded=time.strftime("%Y-%m-%d %H:%M"), go=go)
    json.dump(eff, open(os.path.join(gd, "efficiency.json"), "w"), indent=1)
    log(where, f"port on {key}: {gs:.0f} GPU-s over 4 regions at {bud} s, recall {d['reward']:.4f} ({'passes' if eff['passes'] else 'below'} {base['recall']:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="port the decam-big-tno best program to one GPU device for maximum efficiency")
    ap.add_argument("command", choices=("port", "status", "table", "cost", "grade"))
    ap.add_argument("--target"); ap.add_argument("--agent"); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--go"); ap.add_argument("--budget", type=int)
    a = ap.parse_args()
    if a.command == "port":
        if not (a.target and a.agent):
            sys.exit("port needs --target and --agent")
        port(a.target, a.agent, a.dry)
    elif a.command == "status":
        status()
    elif a.command == "table":
        table()
    elif a.command == "cost":
        cost(a.target)
    else:
        if not a.target:
            sys.exit("grade needs --target")
        grade(a.target, a.go, a.budget, a.dry)
