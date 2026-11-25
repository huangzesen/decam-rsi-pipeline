"""Finish a generation whose LS6 job was submitted by a loop process that was then stopped (to load new loop code):
wait for the job, fetch its outputs, score each submission as grade_all does (Mac fallback when no LS6 run finished
cleanly or the submission was not sent), write reward.json.

Usage: python -I resume_ls6.py RUN JOBID REMOTE_DIR NAME...
"""
import glob, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "ls6"))
import rsi_loop as L
import ls6

run, jid, R, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
logf = open(L.priv(run, "gens", names[0][:6] + "_grading.log"), "a")
log = lambda m: (logf.write(time.strftime("%H:%M:%S ") + m + "\n"), logf.flush())
sent = ls6.ssh(f"ls {R}/subs").stdout.split()
log(f"resumed by a new loop process: waiting for LS6 job {jid} ({', '.join(sent)})")
ls6.wait_fetch(jid, R, L.priv(run, "gens"), sent, log)
for g in names:
    rd = L.priv(run, "gens", g, "ls6_runs")
    infos = [json.load(open(f)) for f in glob.glob(os.path.join(rd, "*", "info.json"))]
    if g in sent and any(i["status"] == "ok" for i in infos):
        r = L.grade(run, g, runs_dir=rd); r["graded_on"] = "A100 (Lonestar6)"
    else:
        log(f"{g}: not sent or no LS6 run finished cleanly; grading on the Mac")
        r = L.grade(run, g); r["graded_on"] = "Mac M1 Ultra"
    r["budget_s"] = L.cap_of(run, g)
    json.dump(r, open(L.priv(run, "gens", g, "reward.json"), "w"), indent=1)
    log(f"{g}: {r}")
