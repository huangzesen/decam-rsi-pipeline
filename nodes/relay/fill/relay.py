#!/usr/bin/env python3
"""DECam fill relay, NERSC -> Vista (2026-11-17).

One cycle: (1) for every Globus task this relay submitted that has ended, SUCCEEDED -> its files are marked relayed and
deleted from NERSC scratch (except files tagged a15 in blank.tsv: the third 5 % needs them at NERSC until its detection
is done; they are listed in relay/keep_a15.txt), FAILED -> its files go back to the queue; (2) new files that fetch.py
has finished (each part's log/done_*.txt) and that are not relayed or in flight go into ONE Globus task (sync size,
checksum verified after transfer) from NERSC $NERSC_SCRATCH/decam-fill/ to Vista
$TACC_SCRATCH/decam-fill/img/<YYMMDD>/. Deletes touch only files under decam-fill/p*/img/ that a
SUCCEEDED task carried. State in fill/relay/ (tasks/*.json, relayed.txt, keep_a15.txt, relay.log).
Usage: python3 -I fill/relay.py once   |   python3 -I fill/relay.py loop [MINUTES=30]   (stop: touch fill/relay/STOP)"""
import json, os, subprocess, sys, tempfile, time
H = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(H, "relay"); os.makedirs(os.path.join(S, "tasks"), exist_ok=True)
NERSC = "$GLOBUS_NERSC_COLLECTION"; VISTA = "$GLOBUS_VISTA_COLLECTION"
SRC = "$NERSC_SCRATCH/decam-fill"; DST = "$TACC_SCRATCH/decam-fill/img"
SSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "$NERSC_USER@perlmutter.nersc.gov"]
ENV = dict(os.environ, LC_ALL="C", LANG="C")
MAX_FILES = 60000                      # per Globus task (~20 TB at most; a cycle normally carries far less)

def log(*x):
    line = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, x)); print(line, flush=True)
    open(os.path.join(S, "relay.log"), "a").write(line + "\n")

def a15_names():
    p = os.path.join(S, "a15_names.txt")
    if not os.path.exists(p):
        names = [l.split("\t")[2] for l in open(os.path.join(H, "blank.tsv")) if l.rstrip("\n").endswith("\ta15")]
        open(p, "w").write("\n".join(names) + "\n")
    return set(l.strip() for l in open(p) if l.strip())

def remote_done():
    """{name: part dir} of every file fetch.py has finished in any part (done logs), read in one ssh call."""
    cmd = f"cd {SRC} && for d in p*/; do d=${{d%/}}; cat $d/log/done_*.txt 2>/dev/null | sed \"s|^|$d |\"; done"
    p = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError("ssh failed: " + p.stderr[-300:])
    out = {}
    for line in p.stdout.splitlines():
        d, _, n = line.partition(" ")
        if n: out[n.strip()] = d
    return out

def gl(*args):
    p = subprocess.run(["globus", *args], capture_output=True, text=True, env=ENV, timeout=600)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def delete_remote(paths):
    """rm the given files (relative to SRC, each p*/img/YYMMDD/name) on NERSC; only such paths are accepted."""
    ok = [x for x in paths if x.startswith("p") and "/img/" in x and ".." not in x and not x.startswith("/")]
    if not ok: return 0
    p = subprocess.run(SSH + [f"python3 -I {SRC}/fill_rm.py"], input="\n".join(ok) + "\n", capture_output=True, text=True, timeout=1800)
    if p.returncode != 0: raise RuntimeError("remote delete failed: " + p.stderr[-300:])
    return int(p.stdout.strip() or 0)

def cycle():
    if not os.path.exists(os.path.join(S, "a15_names.txt")): a15_names()
    keep = a15_names()
    relayed_p = os.path.join(S, "relayed.txt"); relayed = set(l.strip() for l in open(relayed_p)) if os.path.exists(relayed_p) else set()
    inflight = set()
    # 1. finished tasks
    for f in sorted(os.listdir(os.path.join(S, "tasks"))):
        t = json.load(open(os.path.join(S, "tasks", f)))
        if t.get("closed"): continue
        rc, st, err = gl("task", "show", t["task"], "--jmespath", "status", "--format", "unix")
        if st in ("ACTIVE", "INACTIVE") or rc != 0:
            inflight.update(n for _, n in t["files"]); continue
        if st == "SUCCEEDED":
            names = [n for _, n in t["files"]]
            dele = [f"{d}/img/{n[4:10]}/{n}" for d, n in t["files"] if n not in keep]
            k = [n for n in names if n in keep]
            nd = delete_remote(dele)
            open(relayed_p, "a").write("".join(n + "\n" for n in names)); relayed.update(names)
            if k: open(os.path.join(S, "keep_a15.txt"), "a").write("".join(f"{d}/img/{n[4:10]}/{n}\n" for d, n in t["files"] if n in keep))
            t["closed"] = st; t["deleted"] = nd; t["kept_a15"] = len(k); t["bytes"] = gl("task", "show", t["task"], "--jmespath", "bytes_transferred", "--format", "unix")[1]
            log(f"task {t['task'][:8]} SUCCEEDED: {len(names)} files on Vista, {nd} deleted at NERSC, {len(k)} kept (a15)")
        else:
            t["closed"] = st; log(f"task {t['task'][:8]} {st}: {len(t['files'])} files back to the queue")
        json.dump(t, open(os.path.join(S, "tasks", f), "w"))
    # 2. new files
    done = remote_done()
    new = [(d, n) for n, d in done.items() if n not in relayed and n not in inflight][:MAX_FILES]
    if not new:
        log(f"no new files ({len(done)} done at NERSC, {len(relayed)} relayed, {len(inflight)} in flight)"); return
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as b:
        for d, n in new: b.write(f"{d}/img/{n[4:10]}/{n} {n[4:10]}/{n}\n")
    rc, tid, err = gl("transfer", f"{NERSC}:{SRC}/", f"{VISTA}:{DST}/", "--batch", b.name, "--sync-level", "size", "--verify-checksum",
                      "--preserve-timestamp", "--label", f"decam fill relay {time.strftime('%m%d-%H%M')}", "--notify", "failed",
                      "--jmespath", "task_id", "--format", "unix")
    os.remove(b.name)
    if rc != 0 or not tid:
        log("submit failed:", err[-300:]); return
    json.dump(dict(task=tid, submitted=time.strftime("%Y-%m-%d %H:%M:%S"), files=new), open(os.path.join(S, "tasks", tid + ".json"), "w"))
    log(f"task {tid[:8]} submitted: {len(new)} files ({len(done)} done at NERSC, {len(relayed)} relayed before)")

if __name__ == "__main__":
    if sys.argv[1:2] == ["once"]:
        cycle()
    else:
        m = float(sys.argv[2]) if len(sys.argv) > 2 else 30
        while not os.path.exists(os.path.join(S, "STOP")):
            try:
                cycle()
            except Exception as e:
                log("cycle error:", type(e).__name__, str(e)[:300])
            time.sleep(m * 60)
        log("STOP")
