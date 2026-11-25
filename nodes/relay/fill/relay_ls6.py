#!/usr/bin/env python3
"""DECam fill relay, Lonestar6 -> Vista (2026-11-20).

Lonestar6 downloads its own share of the blank (blank_ls6.tsv: the last 8 TB of what was left at 03:30 PST; those names
were appended to NERSC's blank.skip.txt so the NERSC parts leave them) with fetch.py, 4 streams, niced, on <login-node>:
$TACC_SCRATCH/decam-fill-ls6/img/<YYMMDD>/, done log log/done_*.txt.

One cycle, as relay.py: (1) every Globus task this relay submitted that has ended: SUCCEEDED -> files marked relayed and
deleted on Lonestar6; FAILED -> back to the queue; (2) files fetch.py finished on Lonestar6 that are not relayed or in
flight go into ONE Globus task (sync size, checksum verified) to Vista $TACC_SCRATCH/decam-fill/img/<YYMMDD>/,
the same place the NERSC relay fills. Deletes touch only decam-fill-ls6/img/ files a SUCCEEDED task carried.
State in fill/relay_ls6/ (tasks/*.json, relayed.txt, relay.log).
Usage: python3 -I fill/relay_ls6.py once | loop [MINUTES=30]   (stop: touch fill/relay_ls6/STOP)"""
import json, os, subprocess, sys, tempfile, time
H = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(H, "relay_ls6"); os.makedirs(os.path.join(S, "tasks"), exist_ok=True)
LS6 = "$GLOBUS_LS6_COLLECTION"; VISTA = "$GLOBUS_VISTA_COLLECTION"
SRC = "$TACC_SCRATCH/decam-fill-ls6"; DST = "$TACC_SCRATCH/decam-fill/img"
SSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/ls6-$TACC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "x"]
ENV = dict(os.environ, LC_ALL="C", LANG="C")
MAX_FILES = 20000


def log(*x):
    line = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, x)); print(line, flush=True)
    open(os.path.join(S, "relay.log"), "a").write(line + "\n")


def remote(cmd, inp=None, timeout=600):
    p = subprocess.run(SSH + ["LC_ALL=C; " + cmd], input=inp, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError("ssh failed: " + p.stderr[-300:])
    return p.stdout


def gl(*args):
    p = subprocess.run(["globus", *args], capture_output=True, text=True, env=ENV, timeout=600)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def cycle():
    relayed_p = os.path.join(S, "relayed.txt"); relayed = set(l.strip() for l in open(relayed_p)) if os.path.exists(relayed_p) else set()
    inflight = set()
    for f in sorted(os.listdir(os.path.join(S, "tasks"))):
        t = json.load(open(os.path.join(S, "tasks", f)))
        if t.get("closed"):
            continue
        rc, st, err = gl("task", "show", t["task"], "--jmespath", "status", "--format", "unix")
        if st in ("ACTIVE", "INACTIVE") or rc != 0:
            inflight.update(t["files"]); continue
        if st == "SUCCEEDED":
            paths = [f"img/{n[4:10]}/{n}" for n in t["files"] if n.startswith("c4d_") and "/" not in n]
            out = remote(f"cd {SRC} && n=0; while read -r p; do case $p in img/*) [ -f \"$p\" ] && rm -f \"$p\" && n=$((n+1));; esac; done; echo $n",
                         inp="\n".join(paths) + "\n", timeout=1800)
            open(relayed_p, "a").write("".join(n + "\n" for n in t["files"])); relayed.update(t["files"])
            t["closed"] = st; t["deleted"] = int((out.strip().splitlines() or ["0"])[-1])
            log(f"task {t['task'][:8]} SUCCEEDED: {len(t['files'])} files on Vista, {t['deleted']} deleted on Lonestar6")
        else:
            t["closed"] = st; log(f"task {t['task'][:8]} {st}: {len(t['files'])} files back to the queue")
        json.dump(t, open(os.path.join(S, "tasks", f), "w"))
    done = [l.strip() for l in remote(f"cat {SRC}/log/done_*.txt 2>/dev/null").splitlines() if l.strip().startswith("c4d_")]
    new = [n for n in dict.fromkeys(done) if n not in relayed and n not in inflight][:MAX_FILES]
    if not new:
        log(f"no new files ({len(done)} done on Lonestar6, {len(relayed)} relayed, {len(inflight)} in flight)"); return
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as b:
        for n in new:
            b.write(f"img/{n[4:10]}/{n} {n[4:10]}/{n}\n")
    rc, tid, err = gl("transfer", f"{LS6}:{SRC}/", f"{VISTA}:{DST}/", "--batch", b.name, "--sync-level", "size", "--verify-checksum",
                      "--preserve-timestamp", "--label", f"decam fill ls6 relay {time.strftime('%m%d-%H%M')}", "--notify", "failed",
                      "--jmespath", "task_id", "--format", "unix")
    os.remove(b.name)
    if rc != 0 or not tid:
        log("submit failed:", err[-300:]); return
    json.dump(dict(task=tid, submitted=time.strftime("%Y-%m-%d %H:%M:%S"), files=new), open(os.path.join(S, "tasks", tid + ".json"), "w"))
    log(f"task {tid[:8]} submitted: {len(new)} files ({len(done)} done on Lonestar6, {len(relayed)} relayed before)")


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
