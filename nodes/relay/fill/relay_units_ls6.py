#!/usr/bin/env python3
"""Conveyor units' missing NOIRLab files, Lonestar6 -> NERSC (2026-11-20).

NOIRLab refuses most NERSC connections since 11-19 ~19:00, so u06 and u16 waited on 11,236 files (2.2 TB) their NERSC
fetchers could not get. Lonestar6 fetches them ($TACC_SCRATCH/units-ls6, fetch.py on units_ls6.tsv, the
list reversed so the NERSC fetchers, which work from the front, rarely take the same file; job <slurm-job-id>). This relay puts
each finished file into the NERSC share folder its own fetcher writes to (relay_units_ls6/dest.tsv: name -> share dir,
e.g. $NERSC_SCRATCH/units/u06/p0), at <share>/img/<YYMMDD>/<name>. There the NERSC fetcher finds it on disk with
the right size, marks it done and skips it, so its share can finish (SHARE_DONE), and remaining.py / the conveyor
detect it like any other download. Globus checks sizes and checksums; a SUCCEEDED task's files are deleted on Lonestar6.
State in fill/relay_units_ls6/ (tasks/*.json, relayed.txt, relay.log).
Usage: python3 -I fill/relay_units_ls6.py once | loop [MINUTES=10]   (stop: touch fill/relay_units_ls6/STOP)"""
import json, os, subprocess, sys, tempfile, time
H = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(H, "relay_units_ls6"); os.makedirs(os.path.join(S, "tasks"), exist_ok=True)
LS6 = "$GLOBUS_LS6_COLLECTION"; NERSC = "$GLOBUS_NERSC_COLLECTION"
SRC = "$TACC_SCRATCH/units-ls6"
SSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/ls6-$TACC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "x"]
ENV = dict(os.environ, LC_ALL="C", LANG="C")
NBASE = "$NERSC_SCRATCH/units/"


def load_dest():
    """dest.tsv (name -> NERSC share dir) is re-read every cycle: the conveyor appends each new unit's gap (2026-11-20)."""
    d = dict(l.rstrip("\n").split("\t") for l in open(os.path.join(S, "dest.tsv")) if l.strip())
    assert all(v.startswith(NBASE) for v in d.values())
    return d


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
    DEST = load_dest()
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
            paths = t.get("paths") or [f"img/{n[4:10]}/{n}" for n in t["files"] if n.startswith("c4d_") and "/" not in n]
            out = remote(f"cd {SRC} && n=0; while read -r p; do case $p in img/*|u[0-9]*/img/*) [ -f \"$p\" ] && rm -f \"$p\" && n=$((n+1));; esac; done; echo $n",
                         inp="\n".join(paths) + "\n", timeout=1800)
            open(relayed_p, "a").write("".join(n + "\n" for n in t["files"])); relayed.update(t["files"])
            t["closed"] = st; t["deleted"] = int((out.strip().splitlines() or ["0"])[-1])
            log(f"task {t['task'][:8]} SUCCEEDED: {len(t['files'])} files at NERSC, {t['deleted']} deleted on Lonestar6")
        else:
            t["closed"] = st; log(f"task {t['task'][:8]} {st}: {len(t['files'])} files back to the queue")
        json.dump(t, open(os.path.join(S, "tasks", f), "w"))
    where = {}
    for l in remote(f"cd {SRC} && for f in log/done_*.txt u[0-9]*/log/done_*.txt; do [ -f \"$f\" ] || continue; "
                    f"p=${{f%%/log/*}}; [ \"$f\" = \"${{f#log/}}\" ] || p=.; sed \"s|^|$p\t|\" \"$f\"; done").splitlines():
        d, _, n = l.strip().partition("\t")
        if n.startswith("c4d_"):
            where.setdefault(n, "" if d == "." else d + "/")
    done = list(where)
    new = [n for n in done if n not in relayed and n not in inflight and n in DEST]
    if not new:
        log(f"no new files ({len(done)} done on Lonestar6, {len(relayed)} relayed, {len(inflight)} in flight)"); return
    # base paths are the real folders: "/" as a base was refused on 2026-11-20 ("does not map to a valid username")
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as b:
        for n in new:
            b.write(f"{where[n]}img/{n[4:10]}/{n} {DEST[n][len(NBASE):]}/img/{n[4:10]}/{n}\n")
    rc, tid, err = gl("transfer", f"{LS6}:{SRC}/", f"{NERSC}:{NBASE}", "--batch", b.name, "--sync-level", "size", "--verify-checksum",
                      "--preserve-timestamp", "--label", f"units ls6 to nersc {time.strftime('%m%d-%H%M')}", "--notify", "failed",
                      "--jmespath", "task_id", "--format", "unix")
    os.remove(b.name)
    if rc != 0 or not tid:
        log("submit failed:", err[-300:]); return
    json.dump(dict(task=tid, submitted=time.strftime("%Y-%m-%d %H:%M:%S"), files=new, paths=[f"{where[n]}img/{n[4:10]}/{n}" for n in new]),
              open(os.path.join(S, "tasks", tid + ".json"), "w"))
    log(f"task {tid[:8]} submitted: {len(new)} files ({len(done)} done on Lonestar6, {len(relayed)} relayed before)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["once"]:
        cycle()
    else:
        m = float(sys.argv[2]) if len(sys.argv) > 2 else 10
        while not os.path.exists(os.path.join(S, "STOP")):
            try:
                cycle()
            except Exception as e:
                log("cycle error:", type(e).__name__, str(e)[:300])
            time.sleep(m * 60)
        log("STOP")
