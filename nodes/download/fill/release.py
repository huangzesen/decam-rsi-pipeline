#!/usr/bin/env python3
"""Release NERSC scratch as old image folders land on Vista (2026-11-17). Globus lists a task's copied files only after the task ends, so copies are cut into
small tasks (~0.5 TB) and each task's folders are deleted at NERSC as soon as it has SUCCEEDED with every file transferred
or skipped (checksums verified). Files the third 5 % links to stay (NERSC helper rm_dirs.py).
Tracked tasks: fill/release/tasks.json [{task, root, dirs, label, released}]. Subcommands:
  submit-archive5     cancel the single archive5 task (if still queued) and submit the folders in ~0.5 TB tasks
  add TASK ROOT DIR.. track an existing task (e.g. archive10's a10n_k tasks: ROOT $NERSC_SCRATCH/archive10, DIR a10n_0/img)
  once | loop [MIN]   release every SUCCEEDED task (loop: every MIN minutes, default 10, until nothing is left)"""
import json, os, subprocess, sys, tempfile, time
H = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(H, "release"); os.makedirs(S, exist_ok=True)
TF = os.path.join(S, "tasks.json")
NERSC = "$GLOBUS_NERSC_COLLECTION"; VISTA = "$GLOBUS_VISTA_COLLECTION"
SSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "$NERSC_USER@perlmutter.nersc.gov"]
ENV = dict(os.environ, LC_ALL="C", LANG="C")
def log(*x):
    line = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, x)); print(line, flush=True)
    open(os.path.join(S, "release.log"), "a").write(line + "\n")
def gl(*a):
    p = subprocess.run(["globus", *a], capture_output=True, text=True, env=ENV, timeout=600); return p.returncode, p.stdout.strip(), p.stderr.strip()
def load(): return json.load(open(TF)) if os.path.exists(TF) else []
def save(T): json.dump(T, open(TF + ".new", "w"), indent=1); os.replace(TF + ".new", TF)

def submit_archive5():
    rc, st, _ = gl("task", "show", "$GLOBUS_TASK_ID", "--jmespath", "[status, bytes_transferred]", "--format", "unix")
    if st.startswith("ACTIVE"):
        gl("task", "cancel", "$GLOBUS_TASK_ID"); log("cancelled the single archive5 task <globus-task> (", st, ")")
    rows = [l.split() for l in open(os.path.join(H, "archive5_dirs.txt")) if l.strip()]   # dir nfiles bytes (relative to archive5)
    groups, cur, cb = [], [], 0
    for d, n, b in rows:
        cur.append(d); cb += int(b)
        if cb >= 0.5e12: groups.append(cur); cur, cb = [], 0
    if cur: groups.append(cur)
    T = load()
    for g in groups:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            for d in g: f.write(f"{d}/ {d}/ --recursive\n")
        rc, tid, err = gl("transfer", f"{NERSC}:$NERSC_SCRATCH/archive5/", f"{VISTA}:$TACC_SCRATCH/archive5_nersc/",
                          "--batch", f.name, "--sync-level", "size", "--verify-checksum", "--preserve-timestamp",
                          "--label", f"archive5 {g[0].replace('/', '-')} +{len(g) - 1}", "--notify", "failed", "--jmespath", "task_id", "--format", "unix")
        os.remove(f.name)
        if rc != 0 or not tid: log("submit failed:", err[-200:]); continue
        T.append(dict(task=tid, root="$NERSC_SCRATCH/archive5", dirs=g, label=f"archive5 {g[0]} +{len(g) - 1}", released=None))
        save(T)
    log(f"archive5: {len(groups)} tasks submitted for {len(rows)} folders")

def cycle():
    T = load(); left = 0
    for t in T:
        if t.get("released"): continue
        rc, out, _ = gl("task", "show", t["task"], "--jmespath", "[status, files, files_transferred, files_skipped]", "--format", "unix")
        f = out.split()
        if not f or f[0] in ("ACTIVE", "INACTIVE") or rc != 0: left += 1; continue
        if f[0] != "SUCCEEDED" or int(f[2]) + int(f[3]) != int(f[1]):
            t["released"] = f"not released: {out}"; log(t["label"], "NOT released:", out); save(T); continue
        dirs = [os.path.join(t["root"], d) for d in t["dirs"]]
        p = subprocess.run(SSH + ["python3 -I $NERSC_SCRATCH/decam-fill/rm_dirs.py"], input="\n".join(dirs) + "\n",
                           capture_output=True, text=True, timeout=3600)
        if p.returncode != 0: log(t["label"], "delete failed:", p.stderr[-200:]); left += 1; continue
        nd, n, tb, k = p.stdout.split()
        t["released"] = dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), folders=int(nd), files=int(n), tb=float(tb), kept=int(k)); save(T)
        log(f"{t['label']}: copy SUCCEEDED ({f[2]} files), released {n} files {tb} TB at NERSC, kept {k}")
    return left

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "submit-archive5": submit_archive5()
    elif c == "add":
        T = load(); T.append(dict(task=sys.argv[2], root=sys.argv[3], dirs=sys.argv[4:], label=" ".join(sys.argv[4:]), released=None)); save(T)
    elif c == "once": print("left", cycle())
    else:
        m = float(sys.argv[2]) if len(sys.argv) > 2 else 10
        while True:
            try:
                if cycle() == 0: log("all tracked tasks released"); break
            except Exception as e:
                log("cycle error:", type(e).__name__, str(e)[:200])
            time.sleep(m * 60)
