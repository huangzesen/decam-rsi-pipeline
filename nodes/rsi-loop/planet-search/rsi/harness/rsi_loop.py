"""Outer loop of the DECam TNO-finder RSI task (proof of concept on the Mac).

Each generation is one agent session that cannot score anything: it gets the instruction, the code, notes and scalar
reward of every earlier generation, and unlabeled development scenarios; it leaves a submission; the loop grades it on
a fixed sealed set (harness/grade.py) and the next generation sees its reward.

  init   RUN --eval s1,s2,... [--cap-s 600] [--k-per-scen 25]   create the run (private record + agent area)
  starter RUN                                                    grade the starter as gen000
  gen    RUN [--n 1] [--agent claude] [--model claude-opus-5-5] [--effort xhigh] [--hours 3]
  status RUN

Private record (grades, transcripts): rsi/runs/RUN/ (never visible to agents).
Agent area: $RSI_SANDBOX/RUN/genNNN/{workspace,home,tmp} (no CLAUDE.md / AGENTS.md above it).
The agent runs inside a macOS sandbox: network only to the local relay ($RELAY_ADDR); it can read its gen folder,
the development scenarios, the Python install and the Claude Code binary, and nothing else in the home folder or the
data tree (sealed scenarios, truth, raw downloads); it writes only its gen folder.
With "agent_on": "gcp" in a run's config.json (r2 from gen014, 2026-11-10) the session runs instead in a container on
the GCP VM $VM_HOST (gcp_agent.py: same paths, network only to the relay, CPU only); grading stays here.
"""
import argparse, glob, json, os, re, shutil, signal, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
RSI = os.path.dirname(HERE)
ROOT = os.path.realpath(os.path.join(RSI, "data", ".."))           # the Crucial rsi folder (data, scen_*, private)
AGENT_BASE = "$RSI_SANDBOX"
HOME = os.path.expanduser("~")
PY = os.path.join(HOME, "anaconda3", "bin", "python")
RELAY_SETTINGS = "$RELAY_SETTINGS_FILE"
MAX_SUB_BYTES = 500e6       # submission folder limit (2026-11-05; was 5 MB)


def clone_copy(src, dst, *, follow_symlinks=True):
    """copy2, but files over 1 MB as APFS clones (copy-on-write): every round's history/ holds every earlier
    submission, and with weights of up to 500 MB plain copies would fill the disk."""
    if os.path.isfile(src) and not os.path.islink(src) and os.path.getsize(src) > 1e6:
        if os.path.lexists(dst):
            os.remove(dst)
        if subprocess.run(["/bin/cp", "-c", "-p", src, dst], capture_output=True).returncode == 0:
            return dst
    return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)


def priv(run, *p):
    return os.path.join(RSI, "runs", run, *p)


def gens(run):
    return sorted(d for d in os.listdir(priv(run, "gens")) if d.startswith("gen"))


def cap_of(run, gname):
    """The per-scenario budget this generation was told (cap.txt, written by prepare); the run's config otherwise."""
    f = priv(run, "gens", gname, "cap.txt")
    return float(open(f).read()) if os.path.exists(f) else float(json.load(open(priv(run, "config.json")))["cap_s"])


def grade(run, gname, runs_dir=None):
    cfg = json.load(open(priv(run, "config.json")))
    g = priv(run, "gens", gname)
    cmd = [PY, "-I", os.path.join(HERE, "grade.py"), ROOT, os.path.join(g, "submission"), os.path.join(g, "grade"),
           "--scenarios", ",".join(cfg["eval"]), "--cap-s", str(cap_of(run, gname)), "--k-per-scen", str(cfg["k_per_scen"])]
    cmd += ["--runs-dir", runs_dir] if runs_dir else []
    with open(os.path.join(g, "grade.log"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    pub = os.path.join(g, "grade", "public.json")
    r = json.load(open(pub)) if os.path.exists(pub) else dict(reward=0.0, status="grader failed")
    json.dump(r, open(os.path.join(g, "reward.json"), "w"), indent=1)
    return r


def best(run):
    rs = [(json.load(open(priv(run, "gens", g, "reward.json")))["reward"], g) for g in gens(run)
          if os.path.exists(priv(run, "gens", g, "reward.json"))]
    rs = [(r if r is not None else -1.0, g) for r, g in rs]
    return max(rs, key=lambda x: (x[0], x[1]))[1] if rs else None


def sandbox_profile(gdir):
    home = os.path.realpath(HOME)
    allow_read = [os.path.join(home, "anaconda3"), os.path.join(home, ".local", "bin"),
                  os.path.join(home, ".local", "share", "claude"), os.path.join(ROOT, "scen_dev"), gdir]
    return "\n".join(["(version 1)", "(allow default)",
                      "(deny network*)",
                      '(allow network* (remote ip "$RELAY_ADDR"))',
                      "(deny signal)", "(allow signal (target same-sandbox))",     # parallel agents share a Unix user
                      f'(deny file-read* (subpath "{home}"))',
                      f'(deny file-read* (subpath "{ROOT}"))',
                      f'(deny file-read* (subpath "{os.path.realpath(AGENT_BASE)}"))',
                      *[f'(allow file-read* (subpath "{os.path.realpath(p)}"))' for p in allow_read],
                      f'(allow file-read* (literal "{home}"))',
                      f'(deny file-write* (subpath "{home}"))',
                      f'(deny file-write* (subpath "{ROOT}"))',
                      f'(deny file-write* (subpath "{os.path.realpath(AGENT_BASE)}"))',
                      f'(allow file-write* (subpath "{os.path.realpath(gdir)}"))', ""])


def notes_head(path, limit=500):
    """Family / Base / Change lines of a NOTES.md, or its first prose when they are missing."""
    if not os.path.exists(path):
        return "", "", "", ""
    txt = open(path, errors="replace").read()
    f = {k: (re.search(rf"^\W*{k}\W*:\s*(.+)$", txt, re.M | re.I) or [None, ""])[1].strip() for k in ("Family", "Base", "Change")}
    prose = " ".join(l.strip().lstrip("-*0123456789. ") for l in txt.splitlines() if l.strip() and not l.lstrip().startswith("#"))
    return f["Family"], f["Base"], f["Change"], ("" if f["Change"] else prose[:limit] + ("…" if len(prose) > limit else ""))


def summary_md(run):
    rows = []
    kf = priv(run, "families.json")
    known_fam = json.load(open(kf)) if os.path.exists(kf) else {}
    for g in gens(run):
        r = priv(run, "gens", g, "reward.json")
        if not os.path.exists(r) or os.path.exists(priv(run, "gens", g, "skipped.txt")):
            continue
        rw = json.load(open(r))
        fam, base, chg, prose = notes_head(priv(run, "gens", g, "submission", "NOTES.md"))
        fam = fam or known_fam.get(g, "")
        rows.append((rw.get("reward") if rw.get("reward") is not None else -1, g, rw, fam, base, chg or prose))
    rows.sort(key=lambda x: (-x[0], x[1]))
    out = ["# Every earlier program, best first", "",
           "| program | reward | graded on | family | base | main change / method |", "|---|---|---|---|---|---|"]
    cell = lambda x: str(x).replace("|", "/").replace("\n", " ")
    for rv, g, rw, fam, base, chg in rows:
        out.append(f"| {g} | {rv:.4f} | {cell(rw.get('graded_on', ''))} | {cell(fam)} | {cell(base)} | {cell(chg)} |")
    return "\n".join(out) + "\n"


def trace_lines(src):
    """One transcript as markdown: messages, tool calls and shortened tool results, in order."""
    cut = lambda t, n: t if len(t) <= n else t[:n] + f"\n... [{len(t) - n} more characters]"
    out = []
    for line in open(src, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        ts = (d.get("timestamp") or "")[11:19]
        if d.get("type") == "assistant":
            for c in d["message"].get("content", []):
                if c.get("type") == "text" and c["text"].strip():
                    out += [f"## [{ts}] agent", "", c["text"].strip(), ""]
                elif c.get("type") == "tool_use":
                    inp = c.get("input", {})
                    body = inp.get("command") or json.dumps(inp, indent=1)
                    out += [f"### [{ts}] tool: {c.get('name')}", "", "```", cut(str(body), 3000), "```", ""]
        elif d.get("type") == "user":
            content = d["message"].get("content") or []
            if d.get("isSidechain") and isinstance(content, str):      # a subagent's task
                out += [f"## [{ts}] task", "", cut(content.strip(), 3000), ""]
            for c in content if isinstance(content, list) else []:
                if d.get("isSidechain") and c.get("type") == "text":
                    out += [f"## [{ts}] task", "", cut(c["text"].strip(), 3000), ""]
                if c.get("type") == "tool_result":
                    r = c.get("content")
                    r = "\n".join(x.get("text", "") for x in r if isinstance(x, dict)) if isinstance(r, list) else str(r)
                    out += ["result:", "```", cut(r.strip(), 2000), "```", ""]
        elif d.get("type") == "result":
            out += [f"## session ended: {d.get('subtype')}, {d.get('num_turns')} turns, {round((d.get('duration_ms') or 0) / 60000)} min", ""]
    return out


def trace_md(run, g):
    """A readable copy of an agent's session (its messages, tool calls and shortened results) for later agents;
    cached as gens/<g>/trace.md, rebuilt when the transcript is newer. The relay key is redacted in any case."""
    src, dst = priv(run, "gens", g, "transcript.jsonl"), priv(run, "gens", g, "trace.md")
    if not os.path.exists(src):
        return None
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    tok = json.load(open(RELAY_SETTINGS))["env"]["ANTHROPIC_AUTH_TOKEN"]
    out = [f"# Session trace of {g}", "", "Messages, tool calls and (shortened) tool results, in order.", ""] + trace_lines(src)
    for f in sorted(glob.glob(priv(run, "gens", g, "subagents", "*.jsonl")), key=os.path.getmtime):
        out += [f"# Subagent {os.path.basename(f)[6:-6]}", ""] + trace_lines(f)
    txt = "\n".join(out).replace(tok, "[REDACTED]")
    open(dst, "w").write(txt)
    return dst


def prepare(run, gname, hours, start="best", where="mac"):
    """where: the machine the agent works on, "mac" or "gcp" (gcp_agent.py); it picks the instruction's machine text."""
    cfg = json.load(open(priv(run, "config.json")))
    gdir = os.path.join(AGENT_BASE, run, gname)
    ws = os.path.join(gdir, "workspace")
    for d in ("home", "tmp", "claude_config"):
        os.makedirs(os.path.join(gdir, d), exist_ok=True)
    os.makedirs(os.path.join(ws, "history"), exist_ok=True)
    for g in gens(run):
        src = priv(run, "gens", g)
        if not os.path.exists(os.path.join(src, "reward.json")):
            continue
        dst = os.path.join(ws, "history", g)
        shutil.copytree(os.path.join(src, "submission"), os.path.join(dst, "submission"), dirs_exist_ok=True,
                        copy_function=clone_copy)
        if os.path.exists(os.path.join(src, "submission", "NOTES.md")):
            shutil.copy(os.path.join(src, "submission", "NOTES.md"), os.path.join(dst, "NOTES.md"))
        shutil.copy(os.path.join(src, "reward.json"), os.path.join(dst, "reward.json"))
        # The time each hidden scenario took and how it ended, labelled A, B, ... in the config's fixed eval order
        # (2026-11-10). Times and end status only: no
        # per-scenario score, candidate count or log.
        try:
            per = json.load(open(os.path.join(src, "grade", "grade.json")))["per_scenario"]
            rw = json.load(open(os.path.join(dst, "reward.json")))
            rw["per_scenario"] = {chr(65 + i): dict(seconds=round(per[s]["seconds"]), status=per[s]["status"])
                                  for i, s in enumerate(cfg["eval"]) if s in per}
            json.dump(rw, open(os.path.join(dst, "reward.json"), "w"), indent=1)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        t = trace_md(run, g)
        if t:
            shutil.copy(t, os.path.join(dst, "trace.md"))
    shutil.copytree(os.path.join(HERE, "task", "reference"), os.path.join(ws, "reference"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(HERE, "task", "tools"), os.path.join(ws, "tools"), dirs_exist_ok=True)
    open(os.path.join(ws, "history", "SUMMARY.md"), "w").write(summary_md(run))
    b = best(run)
    # "start": "empty" in the run's config (r2, 2026-11-09): until a program has
    # been graded, the agent gets an empty submission/ instead of task/starter
    empty = cfg.get("start") == "empty" and not b
    start_note = ("an empty folder: this run starts from nothing, there is no earlier program, and you write the finder "
                  "from scratch" if empty else
                  f"a copy of the best earlier program, `history/{b}`" if start == "best" and b else
                  "the bare starter program" + (f" (the best earlier program is in `history/{b}`)" if b else ""))
    if empty:
        os.makedirs(os.path.join(ws, "submission"), exist_ok=True)
    else:
        shutil.copytree(priv(run, "gens", b, "submission") if start == "best" and b else os.path.join(HERE, "task", "starter"),
                        os.path.join(ws, "submission"), dirs_exist_ok=True, copy_function=clone_copy)
    txt = open(os.path.join(HERE, "task", "instruction.md")).read().replace("START_NOTE", start_note)
    txt = txt.replace("`~/anaconda3/bin/python`", f"`{PY}`").replace("SESSION_LIMIT", f"{hours:g} hours")
    txt = txt.replace("in `DEV_DIR` (path given below)", f"in `{os.path.join(ROOT, 'scen_dev')}`")
    key = json.load(open(priv(run, "config.json"))).get("grade_on", "mac")
    gr = json.load(open(os.path.join(HERE, "task", "graders.json")))[key]
    os.makedirs(priv(run, "gens", gname), exist_ok=True)
    open(priv(run, "gens", gname, "grader.txt"), "w").write(key)     # this round is graded where it was told
    open(priv(run, "gens", gname, "cap.txt"), "w").write(f"{cfg['cap_s']:g}")   # ... and with the budget it was told
    # the agent's own machine: the Mac (M1 GPU) or, from 2026-11-10 for runs with "agent_on": "gcp", the GCP VM (CPU only)
    txt = txt.replace("GRADER_MACHINE", gr.get("machine_gcp", gr["machine"]) if where == "gcp" else gr["machine"])
    txt = txt.replace("GRADER_SHORT", gr["short"])
    if where == "gcp":
        txt = txt.replace("here on this Mac", "here on this machine")
    txt = txt.replace("`~/anaconda3/bin/python`", f"`{PY}`")
    txt = txt.replace("currently 600 s", f"currently {cfg['cap_s']:g} s")
    txt += (f"\n## Paths\n\n- Development scenarios: `{os.path.join(ROOT, 'scen_dev')}`\n- Python: `{PY}`\n"
            f"- Your working folder: `{ws}` (deliverable: `{os.path.join(ws, 'submission')}`)\n"
            f"- Scratch: `{os.path.join(gdir, 'tmp')}`\n")
    open(os.path.join(ws, "instruction.md"), "w").write(txt)
    open(os.path.join(gdir, "agent.sb"), "w").write(sandbox_profile(gdir))
    return gdir, ws


def run_agent(run, gname, agent, model, effort, hours, note="", start="best", ultracode=False):
    """One agent session; leaves the frozen submission in the private record (graded separately). With "agent_on":
    "gcp" in the run's config the session runs on the GCP VM (gcp_agent.py), and on the Mac when the VM is not ready."""
    where = "mac"
    if json.load(open(priv(run, "config.json"))).get("agent_on") == "gcp":
        sys.path.insert(0, HERE) if HERE not in sys.path else None     # python -I leaves the script's folder off sys.path
        import gcp_agent
        ok, why = gcp_agent.ready()
        where = "gcp" if ok else "mac"
        if not ok:
            print(f"{gname}: GCP VM not ready ({why}); this round runs on the Mac", flush=True)
    gdir, ws = prepare(run, gname, hours, start, where)
    cc = os.path.join(gdir, "claude_config")
    json.dump({"effortLevel": effort, "permissions": {"deny": ["WebFetch", "WebSearch"]}, **({"ultracode": True} if ultracode else {})},
              open(os.path.join(cc, "settings.json"), "w"))
    rel = json.load(open(RELAY_SETTINGS))["env"]
    env = dict(PATH=f"{os.path.dirname(PY)}:/usr/bin:/bin:/usr/sbin:/sbin", HOME=os.path.join(gdir, "home"),
               TMPDIR=os.path.join(gdir, "tmp"), CLAUDE_CONFIG_DIR=cc, OMP_NUM_THREADS="4",
               PYTORCH_ENABLE_MPS_FALLBACK="1", ANTHROPIC_BASE_URL=rel["ANTHROPIC_BASE_URL"],
               ANTHROPIC_AUTH_TOKEN=rel["ANTHROPIC_AUTH_TOKEN"], CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
               DISABLE_AUTOUPDATER="1", LANG="en_US.UTF-8", RSI_DEV_DIR=os.path.join(ROOT, "scen_dev"),
               # headless -p stops background work (workflows, shells, monitors) 10 min after the agent's last turn
               # (gen103 lost its three subagents this way); 0 = wait for them, the session timeout above still applies
               CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS="0",
               RSI_BUDGET_S=str(json.load(open(priv(run, "config.json")))["cap_s"]))
    prompt = (f"Read {os.path.join(ws, 'instruction.md')} and do the task it describes. You have at most {hours:g} "
              f"hours; keep a working submission in {os.path.join(ws, 'submission')} at all times, run tools/check.py, "
              f"and finish with NOTES.md updated." + note)
    claude = os.path.realpath(os.path.join(HOME, ".local", "bin", "claude"))
    cmd = ["sandbox-exec", "-f", os.path.join(gdir, "agent.sb"), claude, "-p", prompt, "--model", model,
           "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose"]
    pg = priv(run, "gens", gname)
    os.makedirs(pg, exist_ok=True)
    t0 = time.time()
    rc = None
    if where == "gcp":
        import gcp_agent
        rc = gcp_agent.run(run, gname, gdir, ws, env, cmd[4:], hours, pg,     # claude's own arguments
                           log=lambda m: print(time.strftime("%H:%M:%S ") + m, flush=True))
        if rc is None:                   # the VM failed before the agent started: this round runs on the Mac after all
            where = "mac"
            gdir, ws = prepare(run, gname, hours, start, "mac")
            t0 = time.time()
    if where == "mac":
        rc = _run_on_mac(cmd, ws, env, pg, hours)
    for f in glob.glob(os.path.join(cc, "projects", "*", "*", "subagents", "**", "agent-*.jsonl"), recursive=True):
        os.makedirs(os.path.join(pg, "subagents"), exist_ok=True)
        shutil.copy2(f, os.path.join(pg, "subagents"))
    sub = os.path.join(pg, "submission")
    shutil.rmtree(sub, ignore_errors=True)
    shutil.copytree(os.path.join(ws, "submission"), sub, copy_function=clone_copy,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.npz", "*.npy", "*.fits*"))
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(sub, "**", "*"), recursive=True) if os.path.isfile(f))
    meta = dict(agent=agent, model=model, effort=effort, hours=hours, exit=rc, seconds=round(time.time() - t0),
                submission_bytes=size, ultracode=ultracode, machine=where)
    json.dump(meta, open(os.path.join(pg, "session.json"), "w"), indent=1)
    return meta


def _run_on_mac(cmd, ws, env, pg, hours):
    with open(os.path.join(pg, "transcript.jsonl"), "w") as out, open(os.path.join(pg, "agent.err"), "w") as err:
        p = subprocess.Popen(cmd, cwd=ws, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            rc = p.wait(timeout=hours * 3600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM); time.sleep(10)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = "timeout"
    return rc


def grade_valid(run, gname):
    sub = priv(run, "gens", gname, "submission")
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(sub, "**", "*"), recursive=True) if os.path.isfile(f))
    if size > MAX_SUB_BYTES or not os.path.exists(os.path.join(sub, "finder.py")):
        r = dict(reward=0.0, status=f"invalid submission (missing finder.py or over {MAX_SUB_BYTES / 1e6:.0f} MB)")
        json.dump(r, open(priv(run, "gens", gname, "reward.json"), "w"))
        return r
    return grade(run, gname)


def grade_all(run, names):
    """Grade a generation's submissions. With "grade_on": "ls6" in the run's config.json the finders run on one
    Lonestar6 gpu-a100-dev job (ls6/ls6.py) and are scored here; any problem falls back to grading on the Mac."""
    cfg = json.load(open(priv(run, "config.json")))
    logf = open(priv(run, "gens", names[0][:6] + "_grading.log"), "a")
    log = lambda m: (logf.write(time.strftime("%H:%M:%S ") + m + "\n"), logf.flush())
    out, remote = {}, []
    for g in names:
        sub = priv(run, "gens", g, "submission")
        size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(sub, "**", "*"), recursive=True) if os.path.isfile(f))
        if size > MAX_SUB_BYTES or not os.path.exists(os.path.join(sub, "finder.py")):
            out[g] = grade_valid(run, g)
    todo = [g for g in names if g not in out]
    told = lambda g: (open(priv(run, "gens", g, "grader.txt")).read().strip()
                      if os.path.exists(priv(run, "gens", g, "grader.txt")) else "mac")
    where = "A100 (Lonestar6)"
    if any(told(g) in ("ls6", "fastest") for g in todo):
        try:
            sys.path.insert(0, os.path.join(HERE, "ls6"))
            import ls6, race
            fastest = any(told(g) == "fastest" for g in todo)
            avail = race.available if fastest else ls6.available
            ok, why = avail(cfg["eval"])
            if not ok:                   # the round was told where it is graded: wait for LS6 rather than grade elsewhere
                log(f"LS6 not available ({why}); waiting up to 16 h for it")
                t_w = time.time()
                while not ok and time.time() - t_w < 16 * 3600:
                    time.sleep(600); ok, why = avail(cfg["eval"])
                if ok:
                    log(f"LS6 available again after {(time.time() - t_w) / 60:.0f} min")
            if not ok:
                raise RuntimeError(why)
            for g in [g for g in todo if told(g) in ("ls6", "fastest")]:
                hits = ls6.lint(priv(run, "gens", g, "submission"))
                if hits:
                    log(f"{g}: network imports {hits[:5]}; graded on the Mac (sandboxed) instead")
                else:
                    remote.append(g)
            if remote and fastest:
                where = race.run(run, priv(run, "gens"), remote, cfg["eval"], max(cap_of(run, g) for g in remote), log)
            elif remote:
                ls6.run_remote(run, priv(run, "gens"), remote, cfg["eval"], max(cap_of(run, g) for g in remote), log)
        except Exception as e:
            log(f"LS6 grading unavailable ({type(e).__name__}: {e}); grading on the Mac although the round was told A100")
            remote = []
    for g in todo:
        rd = priv(run, "gens", g, "ls6_runs")
        infos = [json.load(open(f)) for f in glob.glob(os.path.join(rd, "*", "info.json"))]
        if g in remote and infos and not any(i["status"] == "ok" for i in infos):
            log(f"{g}: no LS6 run finished cleanly ({[i['status'] for i in infos][:3]}...); checking on the Mac")
            remote.remove(g)
        if g in remote and os.path.isdir(rd) and os.listdir(rd):
            out[g] = grade(run, g, runs_dir=rd); out[g]["graded_on"] = where
        else:
            out[g] = grade(run, g); out[g]["graded_on"] = "Mac M1 Ultra"
        out[g]["budget_s"] = cap_of(run, g)
        json.dump(out[g], open(priv(run, "gens", g, "reward.json"), "w"), indent=1)
        log(f"{g}: {out[g]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("run"); p.add_argument("--eval", required=True)
    p.add_argument("--cap-s", type=float, default=600); p.add_argument("--k-per-scen", type=int, default=25)
    p = sub.add_parser("starter"); p.add_argument("run"); p.add_argument("--placeholder", action="store_true")
    p = sub.add_parser("gen"); p.add_argument("run"); p.add_argument("--n", type=int, default=1)
    p.add_argument("--agent", default="claude"); p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="xhigh"); p.add_argument("--hours", type=float, default=1)
    p.add_argument("--cands", type=int, default=5, help="parallel agents per generation (each chooses to improve or pivot)")
    p.add_argument("--until", type=float, help="start no new generation after this Unix time")
    p.add_argument("--ultracode", action="store_true", help="agents run with Claude Code's ultracode setting (workflows of subagents)")
    p = sub.add_parser("status"); p.add_argument("run")
    p = sub.add_parser("grade"); p.add_argument("run"); p.add_argument("names", nargs="+")
    a = ap.parse_args()
    if a.cmd == "init":
        os.makedirs(priv(a.run, "gens"), exist_ok=True)
        json.dump(dict(eval=a.eval.split(","), cap_s=a.cap_s, k_per_scen=a.k_per_scen, made=time.strftime("%Y-%m-%d %H:%M")),
                  open(priv(a.run, "config.json"), "w"), indent=1)
        os.makedirs(os.path.join(AGENT_BASE, a.run), exist_ok=True)
    elif a.cmd == "starter":
        g = priv(a.run, "gens", "gen000")
        shutil.copytree(os.path.join(HERE, "task", "starter"), os.path.join(g, "submission"), dirs_exist_ok=True)
        if a.placeholder:      # let generation 1 start while the starter is graded
            json.dump(dict(reward=None, status="being graded"), open(os.path.join(g, "reward.json"), "w"))
        else:
            print(grade(a.run, "gen000"))
    elif a.cmd == "gen":
        from concurrent.futures import ThreadPoolExecutor
        for _ in range(a.n):
            if a.until and time.time() > a.until:
                print("deadline reached; no new generation", flush=True)
                break
            k = 1 + max((int(g[3:6]) for g in gens(a.run)), default=0)    # an empty-start run has no gen000
            last = json.load(open(priv(a.run, "config.json"))).get("last_gen")   # read every round, so it can be set while running
            if last is not None and k > last:     # r2, 2026-11-09
                print(f"gen{last:03d} was this run's last generation (config last_gen); stopping", flush=True)
                break
            names = [f"gen{k:03d}" + ("abcdefgh"[j] if a.cands > 1 else "") for j in range(a.cands)]
            def note(j):                 # same prompt for every agent: each one chooses whether to improve or pivot
                t = (f" You are agent {'abcdefgh'[j]} of {a.cands} working in parallel in this generation; the others work "
                     f"independently and you cannot see them. This Mac is shared by all {a.cands}: keep your test runs under "
                     f"about 10 GB of memory and 4 CPU threads.") if a.cands > 1 else ""
                if a.ultracode:
                    t += (" This Mac is shared by all your subagents: keep at most 3 test runs going at a time, each under "
                          "about 10 GB of memory and 4 CPU threads.")
                return t + (" Decide for yourself whether to improve the best earlier program (a copy is in submission/) or "
                            "pivot to a different method family (see history/SUMMARY.md and \"Pivot or improve\"); both are "
                            "useful, and the run keeps its best program whatever you submit.")
            with ThreadPoolExecutor(a.cands) as ex:
                metas = list(ex.map(lambda j: run_agent(a.run, names[j], a.agent, a.model, a.effort, a.hours, note(j), ultracode=a.ultracode),
                                    range(a.cands)))
            rs = grade_all(a.run, names)
            for g, m in zip(names, metas):
                print(g, m, rs[g], flush=True)
    elif a.cmd == "grade":
        print(grade_all(a.run, a.names), flush=True)
    else:
        for g in gens(a.run):
            r = priv(a.run, "gens", g, "reward.json")
            s = priv(a.run, "gens", g, "session.json")
            print(g, json.load(open(r)) if os.path.exists(r) else "-", json.load(open(s)) if os.path.exists(s) else "")


if __name__ == "__main__":
    main()
