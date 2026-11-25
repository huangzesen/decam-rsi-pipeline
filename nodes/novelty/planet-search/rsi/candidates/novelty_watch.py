"""Watcher for the candidate novelty queue (2026-11-09). Every 60 s: if the newest MPC files (rsi/data/mpc/<date>/) are over 24 h old, download MPCORB and the ITF to rsi/data/mpc/<today>/
(gzip-checked; old dates are kept, prune by hand); then each spec in novelty_queue/inbox/*.json is run through novelty_check.py (one at a time,
40 min limit, single-threaded), its result written to novelty_queue/done/<name>.result.json, the spec moved to done/, and one line appended to
novelty_queue/RESULTS.md; a lead whose spec has "confirm": true and whose verdict is NOT IN MPC FILES is copied to
confirm_queue/inbox/ for lead_confirm.py (confirm_watch.py). Heartbeat: novelty_queue/WATCHER.json. Stop: create novelty_queue/STOP (or kill the pid in WATCHER.json).
Usage, from candidates/: nohup python -I novelty_watch.py > novelty_queue/watcher.log 2>&1 &"""
import datetime, glob, json, math, os, shutil, subprocess, sys, time
C = os.path.dirname(os.path.abspath(__file__)); Q = C + "/novelty_queue"; H = os.path.dirname(C); PY = sys.executable
for d in ("inbox", "done"):
    os.makedirs(f"{Q}/{d}", exist_ok=True)


def refresh_mpc():
    have = sorted(glob.glob(H + "/data/mpc/20*-*-*/"))
    newest = os.path.getmtime(have[-1] + "MPCORB.DAT.gz") if have and os.path.exists(have[-1] + "MPCORB.DAT.gz") else 0
    if time.time() - newest < 86400:
        return have[-1].rstrip("/").split("/")[-1]
    d = H + "/data/mpc/" + datetime.date.today().isoformat(); tmp = d + ".part"; os.makedirs(tmp, exist_ok=True)
    try:
        for f, url in (("MPCORB.DAT.gz", "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"), ("itf.txt.gz", "https://www.minorplanetcenter.net/iau/ITF/itf.txt.gz")):
            subprocess.run(["curl", "-sS", "-L", "-m", "1800", "-o", f"{tmp}/{f}", url], check=True); subprocess.run(["gzip", "-t", f"{tmp}/{f}"], check=True)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.replace(tmp, d); print(time.strftime("%F %T"), "MPC files refreshed:", d, flush=True)
    except Exception as e:
        print(time.strftime("%F %T"), "MPC refresh failed, keeping the old files:", repr(e)[:150], flush=True)
    return sorted(glob.glob(H + "/data/mpc/20*-*-*/"))[-1].rstrip("/").split("/")[-1]


if not os.path.exists(Q + "/RESULTS.md"):
    open(Q + "/RESULTS.md", "w").write("# Novelty checks (novelty_check.py, run by novelty_watch.py)\n\n| checked (PST) | id | verdict | flagged | MPC files | result |\n|---|---|---|---|---|---|\n")
while not os.path.exists(Q + "/STOP"):
    mpc = refresh_mpc()
    json.dump(dict(pid=os.getpid(), last=time.strftime("%F %T %Z"), mpc_files=mpc, pending=len(glob.glob(Q + "/inbox/*.json"))), open(Q + "/WATCHER.json", "w"))
    def mtime(f):  # an item moved out of the inbox meanwhile (2026-11-11 22:50 crash) sorts last and is skipped below
        try:
            return os.path.getmtime(f)
        except OSError:
            return float("inf")
    for spec in sorted(glob.glob(Q + "/inbox/*.json"), key=mtime):
        if not os.path.exists(spec) or time.time() - mtime(spec) < 5:  # gone, or still being written
            continue
        name = os.path.basename(spec)[:-5]; out = f"{Q}/done/{name}.result.json"; t0 = time.time()
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            p = subprocess.run([PY, "-I", C + "/novelty_check.py", spec, out], capture_output=True, text=True, timeout=2400, env=env, cwd=C)
            log = (p.stdout or "")[-3000:] + (p.stderr or "")[-1500:]
        except subprocess.TimeoutExpired:
            log = "TIMEOUT after 40 min"
        open(f"{Q}/done/{name}.log", "w").write(log)
        try:
            R = json.load(open(out)); verdict, nf, sid = R["verdict"], len(R.get("flagged", [])), R.get("id") or name
        except Exception:
            verdict, nf, sid = "ERROR (see done/" + name + ".log)", "-", name
        try:  # hand a new-looking lead to the confirmation pipeline (lead_confirm.py via confirm_watch.py) when the submitter asked for it
            S_ = json.load(open(spec))
            if verdict.startswith("NOT IN MPC FILES") and S_.get("confirm") and S_.get("sid"):
                # the same body found again in a later round: queue it once (same field, places within 2" on every shared night)
                reg_p = C + "/confirm_queue/queued.json"; reg = json.load(open(reg_p)) if os.path.exists(reg_p) else []; pts = R.get("points") or []; dup = None
                for q in reg:
                    if q["sid"] != S_["sid"]:
                        continue
                    common = [(a, b) for a in pts for b in q["points"] if abs(a[0] - b[0]) < .3]
                    if common and all(math.hypot(((a[1] - b[1] + 180) % 360 - 180) * math.cos(math.radians(a[2])), a[2] - b[2]) * 3600 < 2.0 for a, b in common):
                        dup = q["id"]; break
                if dup:
                    verdict += f" (same body as {dup}; not sent to confirmation again, its result is under that id in confirm_queue/RESULTS.md)"
                else:
                    os.makedirs(C + "/confirm_queue/inbox", exist_ok=True); shutil.copy(spec, C + "/confirm_queue/inbox/" + os.path.basename(spec))
                    reg.append(dict(id=S_.get("id"), sid=S_["sid"], points=pts)); json.dump(reg, open(reg_p, "w"))
                    verdict += " -> queued for confirmation"
        except Exception as e:
            print("could not queue for confirmation:", repr(e)[:100], flush=True)
        os.replace(spec, f"{Q}/done/{name}.json")
        with open(Q + "/RESULTS.md", "a") as fh:
            fh.write(f"| {time.strftime('%F %H:%M')} | {sid} | {verdict} | {nf} | {mpc} | done/{name}.result.json |\n")
        print(time.strftime("%F %T"), sid, "->", verdict, f"({round(time.time() - t0)} s)", flush=True)
        if os.path.exists(Q + "/STOP"):
            break
    time.sleep(60)
print(time.strftime("%F %T"), "STOP file found; watcher exits", flush=True)
