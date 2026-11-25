#!/usr/bin/env python3
"""Copy the GPU run logs of every graded program of a run into its history_extra/gpu_logs/, so the next agents get them
(the earlier run's logs are already in prior_r1/; 2026-11-14).
Agents have no GPU, so these logs are the only place they can see what their CUDA code did in grading (compile failures,
device errors, time per stage). Run after every grading: python3 -I harness/gpu_logs.py r2"""
import glob, json, os, shutil, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(HERE, "runs", sys.argv[1]); out = os.path.join(RUN, "history_extra", "gpu_logs")
rows = []
for g in sorted(glob.glob(os.path.join(RUN, "gens", "gen*"))):
    logs = sorted(glob.glob(os.path.join(g, "grade_runs", "log", "*.log")))
    if not logs or not os.path.exists(os.path.join(g, "grade", "grade.json")):
        continue
    gen = os.path.basename(g); d = os.path.join(out, gen); os.makedirs(d, exist_ok=True)
    for f in logs:
        t = os.path.join(d, os.path.basename(f))
        if not os.path.exists(t) or os.path.getsize(t) != os.path.getsize(f):
            shutil.copyfile(f, t + ".new"); os.replace(t + ".new", t)
        sid = os.path.basename(f)[:-4]; tm = os.path.join(g, "grade_runs", "timing", sid)
        sec, rc = (open(tm).read().split() + ["", ""])[:2] if os.path.exists(tm) else ("", "")
        txt = open(f, errors="replace").read()
        rows.append((gen, sid, sec, rc, "yes" if "Traceback" in txt else "no", "yes" if "CompilationError" in txt else "no"))
L = ["# GPU run logs of this run's graded programs", "",
     "`gpu_logs/genNNN/<region>.log` is the program's own output in the graded run on a NERSC A100 node (the grader's environment).",
     "Exit code 1 with a saved answer means the program stopped on an error after its last checkpoint; the checkpoint was scored.", "",
     "| program | region | seconds | exit code | traceback in log | kernel compile failure in log |", "|---|---|---|---|---|---|"]
L += [f"| {a} | {b} | {c} | {d} | {e} | {f} |" for a, b, c, d, e, f in rows]
open(os.path.join(out, "README.md.new"), "w").write("\n".join(L) + "\n"); os.replace(os.path.join(out, "README.md.new"), os.path.join(out, "README.md"))
print(len(rows), "logs of", len({r[0] for r in rows}), "programs ->", out)
