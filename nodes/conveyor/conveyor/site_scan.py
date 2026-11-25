#!/usr/bin/env python3
"""conveyor: one look at a site for the Mac coordinator (any python3 >= 3.6, standard library only).
Reads the sids to check on stdin (one per line) and prints one JSON object:
  scen     sids whose scenario (SCEN/<sid>/exposures.json) is on this site
  claims   {unit_sid: "LOCAL <site> <job> <lane> <host> <t>" | "REMOTE <site>"}
  timing   {unit_sid: "<seconds> <rc> <gen> <site> <device> <isolation>"}
  cands    {unit_sid: size in bytes of out/<unit>/<sid>/candidates.json} (the program writes "[]" first)
  queue    task files in ROOT/queue
  current  the program in ROOT/best/CURRENT
  workers  this user's conveyor jobs (name cvw): [job, state, partition, elapsed, reason]
Usage: python3 site_scan.py ROOT SCEN USER < sids.txt"""
import glob, json, os, subprocess, sys
root, scen, user = sys.argv[1:4]
sids = [l.strip() for l in sys.stdin if l.strip()]
out = dict(scen=[s for s in sids if os.path.exists(os.path.join(scen, s, "exposures.json"))], claims={}, timing={}, cands={}, queue=[], current=None, workers=[])
for d in glob.glob(os.path.join(root, "claims", "*")):
    k = os.path.basename(d); txt = ""
    for f in ("LOCAL", "REMOTE"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            txt = f + " " + open(p).read().strip(); break
    out["claims"][k] = txt or "LOCAL ?"
for p in glob.glob(os.path.join(root, "timing", "*", "*")):
    unit, sid = p.split(os.sep)[-2:]
    out["timing"][unit + "_" + sid] = open(p).read().strip()
for p in glob.glob(os.path.join(root, "out", "*", "*", "candidates.json")):
    unit, sid = p.split(os.sep)[-3:-1]
    out["cands"][unit + "_" + sid] = os.path.getsize(p)
out["queue"] = sorted(os.listdir(os.path.join(root, "queue"))) if os.path.isdir(os.path.join(root, "queue")) else []
c = os.path.join(root, "best", "CURRENT")
out["current"] = open(c).read().strip() if os.path.exists(c) else None
try:
    q = subprocess.run(["squeue", "-u", user, "-h", "-n", "cvw", "-o", "%i|%T|%P|%M|%r|%q"], stdout=subprocess.PIPE, universal_newlines=True, timeout=120).stdout
    out["workers"] = [l.split("|") for l in q.splitlines() if l.strip()]
except Exception as e:
    out["workers_error"] = repr(e)[:200]
print(json.dumps(out))
