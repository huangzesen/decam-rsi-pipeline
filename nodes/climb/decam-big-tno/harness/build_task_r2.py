#!/usr/bin/env python3
"""Build the design-2 agent task (run r2): the local mirror harbor/decam-big-tno-r2/ and, with --host, the task base on the
agent host ($VM_WORK/tasks-ale/decam-big-tno-r2). Design 2 (2026-11-14; the same day: "we DO NOT separate open set and hidden set. Agent knows EVERYTHING", "we provide the
agent maximal information ... tell the agent full purpose", so /app/open holds all 23 regions, the graded four included, with
the grader's own labels): the r1 task with
  instruction.md        harness/task_r2/instruction.md, numbers filled from runs/r2/config.json and the open regions
  tools/score.py        harness/grade.py with a neutral docstring (the same scoring code, on the open regions' labels)
  tools/subset.py       harness/task_r2/tools/subset.py (cut an open region for a fast CPU test)
  tools/check.py        as in r1
Everything else (Dockerfile, sample, solution, tests, task.toml) is the r1 task. The open regions are not part of the
task folder: climb.py mounts the host folder read-only at /app/open.
On the host the sample (1.6 GB) is hard-linked from the r1 base; nothing of the r1 base is changed, and the new base is
written under a temporary name and moved into place.
Usage: python3 -I harness/build_task_r2.py [--host]
"""
import json, os, re, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "harness"))
SRC = os.path.join(HERE, "harbor", "decam-big-tno"); DST = os.path.join(HERE, "harbor", "decam-big-tno-r2")
CFG = json.load(open(os.path.join(HERE, "runs", "r2", "config.json")))
HOST_R1, HOST_R2 = "$VM_WORK/tasks-ale/decam-big-tno", CFG["vm_base"]
HSSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "$VM_HOST"]
SCORE_DOC = '''#!/usr/bin/env python3
"""The grader's scoring code, for the labelled regions in /app/open (or cuts of them made by tools/subset.py).

    python3 /app/tools/score.py OUT LABELS GRADE --sids sid1[,sid2] [--report GRADE/report.md] [--timing OUT/timing.json]

OUT/<sid>/candidates.json = [{"det": [row numbers], "p": number}, ...] is your program's output for region <sid>
(finder.py --scen .../scen/<sid> --out OUT/<sid>); LABELS/<sid>.npz is that region's labels (/app/open/labels, or the
labels folder subset.py wrote). At most 2,000 candidates per region are read (highest p first). The candidates of all
regions given are ranked by p together and the top K = 2 x (findable labelled bodies in those regions) are scored.
Found: >= 80 % of a candidate's detections belong to one body and cover >= 5 of its nights; each body counts once.
Findable: a labelled body with a detection on >= 5 nights in the region. reward (recall) = found weight / findable weight.
Per body (GRADE/grade.json "feedback", and the --report table), over all valid candidates of its region:
  found       a candidate in the scored top K is >= 80 % this body and covers >= 5 of its nights
  below_cut   such a candidate exists but is outside the scored top K (or beyond the 2,000 read per region)
  few_nights  its purest link (>= 80 % this body) covers only 2-4 of its nights
  mixed       a candidate holds it on >= 2 nights but under 80 % of that candidate's detections are this body
  none        no candidate holds it on 2 nights
If OUT/<sid>/trace.json = {"stages": [{"name": str, "groups": [[row numbers], ...]}, ...]} exists (at most 6 stages,
500,000 groups and 5 M row numbers per stage, 100 MB), the feedback also gives, per stage, the most of the body's nights in
one group that is >= 80 % this body (pure) and in any group (any). Never scored.
"leads" in grade.json: candidates in the scored top K whose detections are >= 80 % unlabelled, on >= 5 nights (false
linkages, known bodies that are missing from the labels, or bodies nobody knows).
gpu_hours is filled only with --timing (a JSON {sid: seconds} you write yourself); the grade measures it.
Writes GRADE/grade.json and GRADE/public.json and prints one summary line.
"""
'''


def score_py():
    s = open(os.path.join(HERE, "harness", "grade.py")).read()
    m = re.match(r'#!/usr/bin/env python3\n""".*?"""\n', s, re.S)
    assert m, "grade.py: docstring not found"
    body = s[m.end():]
    for bad in ("user:", "user,", "private", "sealed", "PROJECT_LEAD_NAME"):
        assert bad not in body, f"grade.py code holds '{bad}': not for agents"
    return SCORE_DOC + body


def open_card():
    """(regions, detections, findable labelled bodies, findable bodies of the graded regions) of the agent's data, from
    `open_data.py list` (local truth, no ssh)."""
    p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "harness", "open_data.py"), "list"], capture_output=True, text=True)
    m = re.search(r"total (\d+) M detections, (\d+) labelled findable bodies, (\d+) regions", p.stdout)
    assert m, p.stdout[-300:] + p.stderr[-300:]
    g = [int(x) for x in re.findall(r"findable (\d+), graded", p.stdout)]
    assert len(g) == len(CFG["sids"]), p.stdout[-300:]
    return int(m.group(3)), int(m.group(1)) * 1e6, int(m.group(2)), sum(g)


def main():
    n, det, bodies, gbodies = open_card()
    gs = [f"`{s}`" for s in CFG["sids"]]
    shutil.rmtree(DST, ignore_errors=True)
    shutil.copytree(SRC, DST, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.bak*", "data"))
    ins = open(os.path.join(HERE, "harness", "task_r2", "instruction.md")).read()
    for k, v in (("BUDGET_S", f"{CFG['budget_s']:,}"), ("SESSION_LIMIT", "2 hours"), ("OPEN_N", str(n)),
                 ("OPEN_DET", f"{det / 1e6:,.0f} million"), ("OPEN_BODIES", str(bodies)),
                 ("GRADED_SIDS", ", ".join(gs[:-1]) + " and " + gs[-1]), ("GRADED_BODIES", str(gbodies))):
        assert k in ins, k
        ins = ins.replace(k, v)
    assert not re.search(r"\b[A-Z]{3,}_[A-Z_]+\b", ins), "unfilled placeholder in instruction.md"
    open(os.path.join(DST, "instruction.md"), "w").write(ins)
    T = os.path.join(DST, "environment", "tools")
    open(os.path.join(T, "score.py"), "w").write(score_py())
    shutil.copy(os.path.join(HERE, "harness", "task_r2", "tools", "subset.py"), T)
    chk = open(os.path.join(HERE, "harness", "task", "tools", "check.py")).read().replace('"SAMPLE_DIR"', '"/app/data/sample"')
    assert "SAMPLE_DIR" not in chk
    open(os.path.join(T, "check.py"), "w").write(chk)
    for f in ("check.py", "score.py", "subset.py"):
        compile(open(os.path.join(T, f)).read(), f, "exec")
    print(f"local mirror {DST}: {n} regions, {det / 1e6:.0f} M detections, {bodies} findable labelled bodies, {gbodies} in the graded four")
    if "--host" not in sys.argv:
        return
    tmp = HOST_R2 + ".new"
    r = subprocess.run(HSSH + [f"set -e; test -d {HOST_R1}/environment/data; rm -rf {tmp}; mkdir -p {tmp}/environment; "
                               f"cp -al {HOST_R1}/environment/data {tmp}/environment/data; echo linked"], capture_output=True, text=True)
    assert r.returncode == 0 and "linked" in r.stdout, r.stderr[-300:]
    r = subprocess.run(["rsync", "-a", "--exclude", "__pycache__", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",
                        DST + "/", f"$VM_HOST:{tmp}/"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    r = subprocess.run(HSSH + [f"set -e; if [ -e {HOST_R2} ]; then mv {HOST_R2} {HOST_R2}.old.$$; fi; mv {tmp} {HOST_R2}; "
                               f"rm -rf {HOST_R2}.old.*; ls {HOST_R2} {HOST_R2}/environment/tools; du -sh {HOST_R2}/environment/data"],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr[-300:])


main()
