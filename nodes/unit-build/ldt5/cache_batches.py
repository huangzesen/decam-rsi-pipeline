#!/usr/bin/env python3
"""Copy a NERSC folder's downloaded images to the Vista cache in batches of ~400 GB (up to 3 Globus tasks at once) and delete
each batch's files at NERSC as soon as its task SUCCEEDED with every file copied; the names go to fill/relay/relayed.txt.
(2026-11-19, : one 4.7 TB task frees nothing
until it ends, because Globus lists a task's copied files only then.) --sync-level checksum: a file the earlier single
task left complete but unverified is checked, not skipped. Usage: python3 -I ldt5/cache_batches.py ROOT 'p*/img' LABEL
Log ldt5/log/cache.log; state ROOT_cache_batches.json beside this script. Stop: touch ldt5/STOP_CACHE."""
import json, os, subprocess, sys, tempfile, time
H = os.path.dirname(os.path.abspath(__file__)); PS = os.path.dirname(H)
NERSC = "$GLOBUS_NERSC_COLLECTION"; VISTA = "$GLOBUS_VISTA_COLLECTION"; VC = "$TACC_SCRATCH/decam-fill/img/"
SSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "$NERSC_USER@perlmutter.nersc.gov"]
ENV = dict(os.environ, LC_ALL="C", LANG="C"); GB, N = 400e9, 3
root, pat, label = sys.argv[1:4]
SF = os.path.join(H, os.path.basename(root) + "_cache_batches.json")
def sh(c): return subprocess.run(SSH + [c], capture_output=True, text=True, timeout=3600).stdout
def gl(*a): return subprocess.run(["globus", *a], capture_output=True, text=True, env=ENV, timeout=600)
def log(*x):
    l = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, x)); print(l, flush=True); open(os.path.join(H, "log", "cache.log"), "a").write(l + "\n")
st = json.load(open(SF)) if os.path.exists(SF) else dict(batches=[], done=[], failed=[])
files = [l.split(" ", 1) for l in sh(f"cd {root} && find {pat} -type f -name 'c4d_*.fits.fz' -printf '%s %p\\n'").splitlines() if l.strip()]
inflight = {r for b in st["batches"] for r in b["files"]}
todo = [(int(s), p) for s, p in files if p not in inflight]
log(f"{label}: {len(files)} files at NERSC, {len(todo)} not in a batch yet")
while True:
    if os.path.exists(os.path.join(H, "STOP_CACHE")):
        log("STOP_CACHE"); break
    keep = []
    for b in st["batches"]:
        s = gl("task", "show", b["task"], "--jmespath", "[status, files, files_transferred, files_skipped]", "--format", "unix").stdout.split()
        if not s or s[0] in ("ACTIVE", "INACTIVE"):
            keep.append(b); continue
        if s[0] == "SUCCEEDED" and int(s[2]) + int(s[3]) == int(s[1]):
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
                f.write("".join(p + "\n" for p in b["files"]))
            subprocess.run(["scp", "-q", "-o", f"ControlPath={os.path.expanduser('$SSH_CM_DIR/nersc-$NERSC_USER.sock')}", f.name,
                            f"$NERSC_USER@perlmutter.nersc.gov:{root}/.cache_del.txt"], check=True); os.remove(f.name)
            sh(f"cd {root} && xargs -a .cache_del.txt -d '\\n' rm -f && rm -f .cache_del.txt")
            open(os.path.join(PS, "fill", "relay", "relayed.txt"), "a").write("".join(p.rsplit("/", 1)[-1] + "\n" for p in b["files"]))
            st["done"].append(b["id"]); log(f"{label} {b['id']} on Vista: {len(b['files'])} files {b['gb']} GB deleted at NERSC")
        else:
            st["failed"].append(b["id"]); todo += [(0, p) for p in b["files"]]; log(f"{label} {b['id']} NOT copied ({' '.join(s)}); its files go to a later batch")
    st["batches"] = keep
    while len(st["batches"]) < N and todo:
        size, take = 0, []
        while todo and size < GB:
            s, p = todo.pop(0); take.append(p); size += s
        bid = f"c{len(st['done']) + len(st['failed']) + len(st['batches']) + 1:03d}"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            for p in take:
                f.write(f"{p} {p.split('/img/', 1)[1]}\n")
        r = gl("transfer", f"{NERSC}:{root}/", f"{VISTA}:{VC}", "--batch", f.name, "--sync-level", "checksum", "--verify-checksum",
               "--label", f"{label} {bid} to the Vista cache", "--notify", "failed", "--jmespath", "task_id", "--format", "unix"); os.remove(f.name)
        if r.returncode or not r.stdout.strip():
            todo = [(0, p) for p in take] + todo; log(f"{label} submit failed: {r.stderr.strip()[-200:]}"); break
        st["batches"].append(dict(id=bid, task=r.stdout.strip(), files=take, gb=round(size / 1e9, 1)))
        log(f"{label} {bid}: {len(take)} files {size / 1e9:.0f} GB, task {r.stdout.strip()}")
    json.dump(st, open(SF, "w"))
    if not st["batches"] and not todo:
        log(f"{label}: all on Vista"); break
    time.sleep(120)
