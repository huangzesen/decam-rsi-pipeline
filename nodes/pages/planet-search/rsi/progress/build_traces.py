"""Build the trace reader on the tracker site (2026-11-04).

Writes $PAGES_OUT/traces/: the page (traces.html) and data/<round>.json for
each finished round, plus data/index.json and data/task.json. Only rounds whose sources changed are rebuilt, and a file
is written only when its content changed, so $PAGE_PUBLISHER publishes only real changes.

The tracker is a shared page, so everything is cleaned before it is written: keys, sealed body names, field ids, local
paths, user names and email addresses never reach the output, and a final check refuses to write a round that still
holds any. Sealed grades are read only for counts. Raw transcripts are in the private planet-campaign repository.

Usage: python3 -I build_traces.py [--force]
"""
import difflib, glob, importlib.util, json, os, re, shutil, sys, time
from datetime import datetime

VERSION = 1
HERE = os.path.dirname(os.path.abspath(__file__))
RSI = os.path.dirname(HERE)
RUN = os.path.join(RSI, "runs", "r1")
GENS = os.path.join(RUN, "gens")
OUT = os.path.expanduser("$PAGES_OUT/traces")
DATA = os.path.join(OUT, "data")
CACHE = os.path.join(HERE, "out", "trace_meta.json")
FIELDS = os.path.join(HERE, "out", "trace_fields.json")
BODY_CAP, TEXT_CAP, RES_HEAD, RES_TAIL = 6000, 20000, 3000, 1000
DIFF_FILE_CAP, DIFF_ROUND_CAP, NOTES_CAP = 60000, 200000, 14000
SKIP_DIFF = ("NOTES.md", "experiments/", "__pycache__/", ".DS_Store")

spec = importlib.util.spec_from_file_location("programs_git", os.path.join(RSI, "harness", "programs_git.py"))
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

KEY_PATTERNS = re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}|\bsk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|\bhf_[A-Za-z0-9]{20,}|"
                          r"\bAKIA[0-9A-Z]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY|Bearer [A-Za-z0-9._-]{20,}|\bgt-[A-Za-z0-9_.-]{20,}")
PATHS = re.compile(r"(?<![\w.:/-])(?:/Users|/Volumes|/private|/tmp|/opt|/home|/pscratch|/scratch|/global|/work|/var)/[^\s'\"`),:;\]<>]*")
PATH_MARKS = (("/workspace/", "workspace/"), ("/scen_dev/", "dev/"), ("/submission/", "submission/"), ("/history/", "history/"),
              ("/reference/", "reference/"), ("/tools/", "tools/"), ("/tmp/", "tmp/"), ("/anaconda3/", "python/"))
HASH_ID = re.compile(r"\bs[0-9a-f]{8}\b")
USERS = re.compile(r"\b(?:TACC_USER|GITHUB_USER|EMAIL_USER|GMAIL_USER|NERSC_USER)\b", re.I)  # the real login names; placeholders here (CONFIG.md)
EMAIL = re.compile(r"[\w.+-]+@(?!anthropic\.com)[\w-]+\.[\w.-]+")
CLIENT = re.compile(r"client_name|CLIENT_NAME_ZH|client_id\d", re.I)


class Cleaner:
    def __init__(self):
        home = os.path.expanduser("~")
        self.exact = []
        for path, read in ((home + "/<relay-settings-file>", lambda t: json.loads(t)["env"]["ANTHROPIC_AUTH_TOKEN"]),
                           (home + "/<pages-password-file>", str.strip), (home + "/<rubin-token-file>", str.strip)):
            try:
                self.exact.append(read(open(path).read()))
            except (OSError, ValueError, KeyError):
                pass
        self.exact = [s for s in self.exact if s]
        names = set()
        for f in glob.glob(os.path.join(RSI, "private", "truth", "*.json")):
            if os.path.basename(f)[:-5] in set(json.load(open(os.path.join(RUN, "config.json")))["eval"]):
                names |= {b["pdes"] for b in json.load(open(f))["bodies"]}
        if not names:
            raise SystemExit("sealed names unavailable (is Crucial-8TB mounted?); nothing built")
        self.names = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")(?![\w-])")
        ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RSI, "scen_dev", "s*")))
        if ids:
            json.dump(ids, open(FIELDS, "w"))
        else:
            ids = json.load(open(FIELDS)) if os.path.exists(FIELDS) else []
        self.dev = {s: f"practice-field-{i + 1}" for i, s in enumerate(ids)}

    @staticmethod
    def _path(m):
        p = m.group(0)
        for mark, name in PATH_MARKS:
            if mark in p:
                return name + p.split(mark, 1)[1]
        if p.rstrip("/").endswith("/workspace"):
            return "workspace"
        tail = [x for x in p.split("/") if x]
        return ".../" + tail[-1] if tail else "..."

    def __call__(self, s):
        if not s:
            return s
        for e in self.exact:
            s = s.replace(e, "[key]")
        s = KEY_PATTERNS.sub("[key]", s)
        s = self.names.sub("[body]", s)
        s = HASH_ID.sub(lambda m: self.dev.get(m.group(0), "field-x"), s)
        s = PATHS.sub(self._path, s)
        s = USERS.sub("user", s)
        s = EMAIL.sub("[email]", s)
        return CLIENT.sub("[client]", s)

    def check(self, text):
        for e in self.exact:
            if e in text:
                return "a key"
        for name, rx in (("a key", KEY_PATTERNS), ("a sealed name", self.names), ("a field id", HASH_ID), ("a path", PATHS), ("a user name", USERS)):
            if rx.search(text):
                return name
        return None


def order(g):
    return int(re.match(r"gen(\d+)", g).group(1)), g


def read_json(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def stamp(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def cap(s, n):
    return (s, 0) if len(s) <= n else (s[:n], len(s) - n)


def tool_event(c, clean):
    name, inp = c.get("name") or "tool", c.get("input") or {}
    ev = dict(k="tool", n=name, h="")
    body = ""
    if name == "Bash":
        body = str(inp.get("command") or "")
        ev["h"] = str(inp.get("description") or body.strip().split("\n")[0])[:160]
    elif name == "Edit":
        ev["h"] = str(inp.get("file_path") or "")
        o, cut1 = cap(str(inp.get("old_string") or ""), BODY_CAP)
        w, cut2 = cap(str(inp.get("new_string") or ""), BODY_CAP)
        ev.update(o=clean(o), w=clean(w))
        if cut1 + cut2:
            ev["bc"] = cut1 + cut2
    elif name == "Write":
        ev["h"] = str(inp.get("file_path") or "")
        body = str(inp.get("content") or "")
    elif name == "Read":
        extra = f" (from line {inp['offset']})" if inp.get("offset") else ""
        ev["h"] = str(inp.get("file_path") or "") + extra
    elif name in ("Grep", "Glob"):
        ev["h"] = str(inp.get("pattern") or "")
    elif name in ("Task", "Agent"):
        ev["h"] = str(inp.get("description") or "")
        body = str(inp.get("prompt") or "")
    elif name == "Workflow":
        script = str(inp.get("script") or "")
        m = re.search(r"name:\s*['\"]([^'\"]+)", script)
        ev["h"] = (m.group(1) if m else str(inp.get("name") or inp.get("scriptPath") or "workflow"))
        body = script
    elif name == "StructuredOutput":
        ev["h"] = "structured result"
        body = json.dumps(inp, indent=1, ensure_ascii=False)
    else:
        body = json.dumps(inp, indent=1, ensure_ascii=False) if inp else ""
        ev["h"] = body.replace("\n", " ")[:120]
    ev["h"] = clean(ev["h"])
    if body:
        b, cut = cap(body, BODY_CAP)
        ev["b"] = clean(b)
        if cut:
            ev["bc"] = cut
    return ev


def set_result(ev, c, clean):
    r = c.get("content")
    r = "\n".join(x.get("text", "") for x in r if isinstance(x, dict)) if isinstance(r, list) else str(r if r is not None else "")
    r = r.strip()
    ev["rl"] = r.count("\n") + 1 if r else 0
    if len(r) > RES_HEAD + RES_TAIL:
        ev["rc"] = len(r) - RES_HEAD - RES_TAIL
        ev["r"] = clean(r[:RES_HEAD])
        ev["rt"] = clean(r[-RES_TAIL:])
    else:
        ev["r"] = clean(r)
    if c.get("is_error"):
        ev["e"] = True


def parse_thread(path, clean, t0=None):
    """A transcript as events. A tool call and its result are one event. Returns events, the start time, ended info."""
    events, tools, end = [], {}, None
    with open(path, errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ts = stamp(d.get("timestamp"))
            if ts is not None and t0 is None:
                t0 = ts
            t = round(ts - t0) if ts is not None and t0 is not None else None
            kind = d.get("type")
            if kind == "assistant":
                for c in (d.get("message") or {}).get("content") or []:
                    if c.get("type") == "text" and (c.get("text") or "").strip():
                        x, _ = cap(c["text"].strip(), TEXT_CAP)
                        events.append(dict(k="say", t=t, x=clean(x)))
                    elif c.get("type") == "tool_use":
                        ev = tool_event(c, clean)
                        ev["t"] = t
                        events.append(ev)
                        tools[c.get("id")] = ev
            elif kind == "user":
                content = (d.get("message") or {}).get("content") or []
                # only a subagent's first message is its task; the main session is launched with a prompt that isn't
                # recorded (the agent's first step reads instruction.md), and its later text is system notices
                is_task = d.get("isSidechain") and not any(e["k"] == "task" for e in events)
                if isinstance(content, str):
                    if is_task:
                        x, _ = cap(content.strip(), TEXT_CAP)
                        events.append(dict(k="task", t=t, x=clean(x)))
                    continue
                for c in content:
                    if c.get("type") == "tool_result" and c.get("tool_use_id") in tools:
                        set_result(tools[c["tool_use_id"]], c, clean)
                    elif c.get("type") == "text" and (c.get("text") or "").strip() and is_task:
                        x, _ = cap(c["text"].strip(), TEXT_CAP)
                        events.append(dict(k="task", t=t, x=clean(x)))
                        is_task = False
            elif kind == "result":
                end = dict(k="end", t=t, x=clean(str(d.get("result") or ""))[:TEXT_CAP], turns=d.get("num_turns"), how=d.get("subtype"))
    if end:
        events.append(end)
    return events, t0


def sub_title(events):
    task = next((e["x"] for e in events if e["k"] == "task"), "")
    task = re.sub(r"^\[Workflow harness[^\n]*\n+", "", task)
    lines = [x.strip() for x in task.split("\n") if x.strip()]
    # workflow agents share an opening "Context:" line; the line that names their own job says more
    line = next((x for x in lines if re.match(r"^[\w /-]{0,30}\b(lens|angle|focus|role|question)\s*:", x, re.I)), lines[0] if lines else "subagent")
    return re.sub(r"^[#*\s]+", "", line)[:140]


def read_text(path):
    try:
        with open(path, "rb") as f:
            raw = f.read(2_000_000)
        return None if b"\0" in raw else raw.decode("utf-8", "replace").split("\n")
    except OSError:
        return None


def code_diff(base, g, clean):
    """What this round changed against the program it started from: program files only, not notes or experiments."""
    if not base:
        return [], 0, 0
    a, b = os.path.join(GENS, base, "submission"), os.path.join(GENS, g, "submission")
    if not (os.path.isdir(a) and os.path.isdir(b)):
        return [], 0, 0
    def files(root):
        out = set()
        for r, _, fs in os.walk(root):
            for f in fs:
                rel = os.path.relpath(os.path.join(r, f), root)
                if not any(rel == s or rel.startswith(s) or f"/{s}" in f"/{rel}" for s in SKIP_DIFF):
                    out.add(rel)
        return out
    out, add, rem, room = [], 0, 0, DIFF_ROUND_CAP
    for rel in sorted(files(a) | files(b)):
        ta, tb = read_text(os.path.join(a, rel)) if os.path.exists(os.path.join(a, rel)) else [], read_text(os.path.join(b, rel)) if os.path.exists(os.path.join(b, rel)) else []
        if ta is None or tb is None or ta == tb:
            continue
        lines = list(difflib.unified_diff(ta, tb, lineterm="", n=3))[2:]
        fa = sum(1 for x in lines if x.startswith("+"))
        fr = sum(1 for x in lines if x.startswith("-"))
        add, rem = add + fa, rem + fr
        text = "\n".join(lines)
        limit = min(DIFF_FILE_CAP, max(0, room))
        d, cut = cap(text, limit)
        room -= len(d)
        out.append(dict(f=clean(rel), a=fa, r=fr, d=clean(d), c=cut))
    return out, add, rem


def round_notes(g, clean):
    """The round's own part of NOTES.md: the header and its first section."""
    try:
        t = open(os.path.join(GENS, g, "submission", "NOTES.md"), errors="replace").read()
    except OSError:
        return "", None
    heads = [m.start() for m in re.finditer(r"^# ", t, re.M)]
    part = t[:heads[1]] if len(heads) > 1 else t
    change = re.search(r"^Change:\s*(.*)$", part, re.M)
    family = re.search(r"^Family:\s*(.*)$", part, re.M)
    x, cut = cap(part.strip(), NOTES_CAP)
    if cut:
        x += f"\n\n... {cut} more characters"
    return clean(x), dict(change=clean(change.group(1).strip()) if change else "", family=clean(family.group(1).strip()) if family else "")


def build_round(g, rewards, clean):
    d = os.path.join(GENS, g)
    rw = read_json(os.path.join(d, "reward.json")) or {}
    ss = read_json(os.path.join(d, "session.json")) or {}
    grade = read_json(os.path.join(d, "grade", "grade.json")) or {}
    base = pg.base_from_prompt(g)
    inferred = base is None and g != "gen000"
    if inferred:
        base = pg.best_before(g, rewards)
    events, t0, subs = [], None, []
    tr = os.path.join(d, "transcript.jsonl")
    if os.path.exists(tr):
        events, t0 = parse_thread(tr, clean)
    for f in sorted(glob.glob(os.path.join(d, "subagents", "*.jsonl"))):
        ev, _ = parse_thread(f, clean, t0)
        if ev:
            subs.append(dict(k="sub", t=next((e["t"] for e in ev if e.get("t") is not None), None), title=clean(sub_title(ev)), ev=ev))
    merged = sorted(events + subs, key=lambda e: (e["t"] is None, e["t"] if e["t"] is not None else 0))
    tools = sum(e["k"] == "tool" for e in events) + sum(x["k"] == "tool" for s in subs for x in s["ev"])
    says = sum(e["k"] == "say" for e in events) + sum(x["k"] == "say" for s in subs for x in s["ev"])
    errs = sum(1 for e in events if e.get("e")) + sum(1 for s in subs for x in s["ev"] if x.get("e"))
    diff, add, rem = code_diff(base, g, clean)
    notes, head = round_notes(g, clean)
    skipped = os.path.join(d, "skipped.txt")
    meta = dict(g=g, n=order(g)[0], reward=rw.get("reward"), status=rw.get("status") or "",
                points=grade.get("hit_weight"), total=grade.get("truth_weight"),
                found=[sum(h["tier"] == t for h in grade.get("hits") or []) for t in (1, 2, 3)] if grade else None,
                base=base, inferred=inferred, change=(head or {}).get("change", ""), family=(head or {}).get("family", ""),
                minutes=round(ss["seconds"] / 60) if ss.get("seconds") else None, tools=tools, says=says, subs=len(subs), errors=errs,
                add=add, rem=rem, files=len(diff), graded_on=clean(rw.get("graded_on") or ""),
                at=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(os.path.join(d, "reward.json")))) if rw else None,
                skipped=clean(open(skipped).read().strip()) if os.path.exists(skipped) else None,
                starter=g == "gen000")
    return meta, dict(g=g, events=merged, diff=diff, notes=notes)


def write_if_changed(path, text):
    try:
        if open(path, encoding="utf-8").read() == text:
            return False
    except OSError:
        pass
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return True


def task_page(clean):
    t = open(os.path.join(RSI, "harness", "task", "instruction.md")).read()
    gr = json.load(open(os.path.join(RSI, "harness", "task", "graders.json")))["fastest"]
    cfg = json.load(open(os.path.join(RUN, "config.json")))
    for old, new in (("START_NOTE", "a copy of the best earlier program"), ("SESSION_LIMIT", "1 hour"), ("GRADER_MACHINE", gr["machine"]),
                     ("GRADER_SHORT", gr["short"]), ("currently 600 s", f"currently {cfg['cap_s']:g} s")):
        t = t.replace(old, new)
    c = json.load(open(os.path.join(RSI, "task-card", "card.json")))
    keep = dict(one_line=c.get("one_line"), why_care=c.get("why_care"), agent_writes=(c.get("task") or {}).get("agent_writes"),
                truth=(c.get("truth") or {}).get("source"), sealed=(c.get("scenarios") or {}).get("sealed"), metric=(c.get("metric") or {}).get("formula"),
                anti_gaming=c.get("anti_gaming"), budget=c.get("budget"), levers=c.get("levers"), risks=c.get("risks"))
    return dict(instruction=clean(t), card=json.loads(clean(json.dumps(keep, ensure_ascii=False))))


def main():
    force = "--force" in sys.argv
    clean = Cleaner()
    os.makedirs(DATA, exist_ok=True)
    cache = read_json(CACHE) or {}
    if cache.get("version") != VERSION:
        cache, force = {"version": VERSION, "rounds": {}}, True
    names = sorted((g for g in os.listdir(GENS) if os.path.isdir(os.path.join(GENS, g)) and re.match(r"gen\d", g)), key=order)
    rewards = {}
    for g in names:
        rw = read_json(os.path.join(GENS, g, "reward.json"))
        if rw is not None:
            rewards[g] = rw.get("reward") if rw.get("reward") is not None else -1.0
    built, running = 0, []
    for g in names:
        d = os.path.join(GENS, g)
        finished = g in rewards or os.path.exists(os.path.join(d, "skipped.txt")) or g == "gen000"
        if not finished:
            running.append(g)
            continue
        srcs = [os.path.join(d, x) for x in ("transcript.jsonl", "reward.json", "session.json", "skipped.txt") if os.path.exists(os.path.join(d, x))] + glob.glob(os.path.join(d, "subagents", "*.jsonl"))
        newest = max((os.path.getmtime(p) for p in srcs), default=0)
        have = cache["rounds"].get(g)
        if have and not force and have["src"] >= newest and os.path.exists(os.path.join(DATA, g + ".json")):
            continue
        meta, body = build_round(g, rewards, clean)
        text = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        problem = clean.check(text) or clean.check(json.dumps(meta, ensure_ascii=False))
        if problem:
            raise SystemExit(f"stopped at {g}: {problem} survived cleaning; nothing written for it")
        write_if_changed(os.path.join(DATA, g + ".json"), text)
        cache["rounds"][g] = dict(src=newest, meta=meta)
        built += 1
    rounds = [cache["rounds"][g]["meta"] for g in names if g in cache["rounds"]]
    top = -1.0
    for m in sorted(rounds, key=lambda m: order(m["g"])):
        r = m["reward"]
        m["record"] = r is not None and r > top + 1e-9 and m["g"] != "gen000"
        if r is not None and r > top:
            top = r
    index = dict(rounds=rounds, running=running, best=top)
    text = json.dumps(index, separators=(",", ":"), ensure_ascii=False)
    task = json.dumps(task_page(clean), separators=(",", ":"), ensure_ascii=False)
    for t in (text, task):
        problem = clean.check(t)
        if problem:
            raise SystemExit(f"stopped: {problem} survived cleaning in the index or task")
    changed = write_if_changed(os.path.join(DATA, "index.json"), text)
    changed |= write_if_changed(os.path.join(DATA, "task.json"), task)
    changed |= write_if_changed(os.path.join(OUT, "index.html"), open(os.path.join(HERE, "traces.html"), encoding="utf-8").read())
    tmp = CACHE + ".tmp"
    json.dump(cache, open(tmp, "w"))
    os.replace(tmp, CACHE)
    print(f"traces: {built} rounds rebuilt, {len(rounds)} in the reader, running {running}, page {'updated' if changed or built else 'unchanged'}")


if __name__ == "__main__":
    main()
