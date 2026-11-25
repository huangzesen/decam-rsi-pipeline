#!/usr/bin/env python3
"""ldt5 detection feeder (Mac, 2026-11-18): keeps NERSC's CPU debug queue full (5 submitted, 2 run) with 30-min detection
jobs (l5_detect_dbg.sbatch), one job per share at a time, download parts p0-p6 first, then the mirror shares l5_4-l5_7.
The shared-GPU jobs l5d (61519795-98 = l5_4..l5_7) stay queued as a backup; a pending one is cancelled when the feeder
takes its share, a running one keeps its share. When nothing is left it submits the scenario build (rsi/decam-big-tno/harness/searchldt5/buildldt5.sbatch, CPU debug)
until dbt/searchldt5/BUILT exists; the queued search jobs (runldt5.sbatch) wait for that file. Stops then, or when
ldt5/log/FEED_STOP exists here.
Usage: python3 -I ldt5/debug_feed.py [MINUTES]"""
import os, subprocess, sys, time
H = os.path.dirname(os.path.abspath(__file__)); R = "$NERSC_SCRATCH/ldt5"
SSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "$NERSC_USER@perlmutter.nersc.gov"]
GPUJOB = {"l5_4": "61519795", "l5_5": "61519796", "l5_6": "61519797", "l5_7": "61519798"}
def sh(c): return subprocess.run(SSH + [c], capture_output=True, text=True, timeout=600).stdout
def log(*x):
    line = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, x)); print(line, flush=True)
    open(os.path.join(H, "log", "debug_feed.log"), "a").write(line + "\n")
m = float(sys.argv[1]) if len(sys.argv) > 1 else 4
while not os.path.exists(os.path.join(H, "log", "FEED_STOP")):
    try:
        left = {l.split()[0]: int(l.split()[1]) for l in sh(f"$NERSC_SCRATCH/dbt/env/bin/python -I {R}/remaining.py").splitlines() if l.strip()}
        q = [l.split() for l in sh("squeue -u $NERSC_USER -h -o '%i %j %T'").splitlines() if l.strip()]
        dbg = {j[1][4:]: j for j in q if j[1].startswith("l5c-")}
        gpu = {j[0]: j[2] for j in q if j[1] == "l5d"}
        if not left:                              # NERSC unreachable (11-18 14:47-15:53 it read as "nothing left"): skip
            log("no answer from NERSC; skipped"); time.sleep(m * 60); continue
        todo = [s for s, n in left.items() if n > 0]
        if not todo:
            S = "$NERSC_SCRATCH/dbt/searchldt5"
            if sh(f"[ -f {S}/BUILT ] && echo yes").strip() == "yes":
                log("scenarios built; search jobs can run"); break
            if not any(j[1] == "dbt-buildl5" for j in q):
                log("detection done:", sum(left.values()), "left; build:", sh(f"cd {S} && sbatch --export=ALL,LIST={S}/list.txt buildldt5.sbatch").strip())
            time.sleep(m * 60); continue
        free = 5 - len(dbg)
        for s in todo:
            if free <= 0: break
            if s in dbg: continue
            g = GPUJOB.get(s)
            if g and gpu.get(g) == "RUNNING": continue
            if g and gpu.get(g) == "PENDING":
                sh(f"scancel {g}"); log("cancelled pending GPU job", g, "for", s)
            out = sh(f"cd {R} && sbatch -J l5c-{s} --export=ALL,DIR={R}/{s} l5_detect_dbg.sbatch").strip()
            log(s, left[s], "left:", out); free -= 1
        log("left", sum(left.values()), {s: n for s, n in left.items() if n}, "| debug jobs", {s: j[2] for s, j in dbg.items()})
    except Exception as e:
        log("cycle error:", type(e).__name__, str(e)[:200])
    time.sleep(m * 60)
