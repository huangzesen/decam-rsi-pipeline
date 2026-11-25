"""Watcher for the lead confirmation queue (2026-11-09). Every 60 s, each
confirm_queue/inbox/*.json (a novelty_check spec; novelty_watch.py puts a lead here when its verdict is NOT IN MPC FILES and the spec has
"confirm": true) is run through lead_confirm.py into confirm_queue/runs/<name>/ (one at a time, 4 h limit), the spec moved into that folder, and one
line appended to confirm_queue/RESULTS.md. Heartbeat confirm_queue/WATCHER.json; stop: touch confirm_queue/STOP.
Usage, from candidates/: nohup python -I confirm_watch.py > confirm_queue/watcher.log 2>&1 &"""
import glob, json, os, subprocess, sys, time
C = os.path.dirname(os.path.abspath(__file__)); Q = C + "/confirm_queue"; PY = sys.executable
for d in ("inbox", "runs"):
    os.makedirs(f"{Q}/{d}", exist_ok=True)
if not os.path.exists(Q + "/RESULTS.md"):
    open(Q + "/RESULTS.md", "w").write("# Lead confirmations (lead_confirm.py, run by confirm_watch.py)\n\n| finished (PST) | id | verdict | folder |\n|---|---|---|---|\n")
while not os.path.exists(Q + "/STOP"):
    json.dump(dict(pid=os.getpid(), last=time.strftime("%F %T %Z"), pending=len(glob.glob(Q + "/inbox/*.json"))), open(Q + "/WATCHER.json", "w"))
    for spec in sorted(glob.glob(Q + "/inbox/*.json"), key=os.path.getmtime):
        if time.time() - os.path.getmtime(spec) < 5:
            continue
        name = os.path.basename(spec)[:-5]; D = f"{Q}/runs/{name}"; os.makedirs(D, exist_ok=True); t0 = time.time()
        try:
            p = subprocess.run([PY, "-I", C + "/lead_confirm.py", spec, D], capture_output=True, text=True, timeout=4 * 3600, cwd=C)
            log = p.stdout[-6000:] + p.stderr[-3000:]
        except subprocess.TimeoutExpired:
            log = "TIMEOUT after 4 h"
        open(D + "/pipeline.log", "a").write(log)
        os.replace(spec, D + "/spec_submitted.json")
        v = open(D + "/verdict.txt").read().strip() if os.path.exists(D + "/verdict.txt") else "ERROR (see pipeline.log)"
        sid = json.load(open(D + "/spec_submitted.json")).get("id", name)
        with open(Q + "/RESULTS.md", "a") as fh:
            fh.write(f"| {time.strftime('%F %H:%M')} | {sid} | {v} | runs/{name}/ |\n")
        print(time.strftime("%F %T"), sid, "->", v, f"({round(time.time() - t0)} s)", flush=True)
        if os.path.exists(Q + "/STOP"):
            break
    time.sleep(60)
print(time.strftime("%F %T"), "STOP file found; watcher exits", flush=True)
