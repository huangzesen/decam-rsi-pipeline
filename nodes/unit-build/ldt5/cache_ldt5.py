#!/usr/bin/env python3
"""ldt5 (2026-11-19): its 8,385 downloaded exposures (ldt5/p0-p6/img, 4.9 TB) to the Vista cache
($TACC_SCRATCH/decam-fill/img/<YYMMDD>/) with Globus; when the task SUCCEEDED, their names go to
fill/relay/relayed.txt (later units pull them from Vista) and the NERSC copies are deleted (detection lists stay).
NERSC scratch was at 19.1 of 20 TiB, which pauses the fill. Usage: python3 -I ldt5/cache_ldt5.py submit | wait TASK"""
import os, subprocess, sys, tempfile, time
H = os.path.dirname(os.path.abspath(__file__)); PS = os.path.dirname(H)
R = "$NERSC_SCRATCH/ldt5"; NERSC = "$GLOBUS_NERSC_COLLECTION"; VISTA = "$GLOBUS_VISTA_COLLECTION"
SSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "$NERSC_USER@perlmutter.nersc.gov"]
ENV = dict(os.environ, LC_ALL="C", LANG="C")
def sh(c): return subprocess.run(SSH + [c], capture_output=True, text=True, timeout=1800).stdout
def log(*x):
    l = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, x)); print(l, flush=True); open(os.path.join(H, "log", "cache.log"), "a").write(l + "\n")
if sys.argv[1] == "submit":
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        for k in range(7):
            f.write(f"p{k}/img/ ./ --recursive\n")
    p = subprocess.run(["globus", "transfer", f"{NERSC}:{R}/", f"{VISTA}:$TACC_SCRATCH/decam-fill/img/", "--batch", f.name,
                        "--sync-level", "size", "--verify-checksum", "--label", "ldt5 downloads to the Vista cache", "--notify", "failed",
                        "--jmespath", "task_id", "--format", "unix"], capture_output=True, text=True, env=ENV)
    log("submitted", p.stdout.strip(), p.stderr.strip()[-200:])
else:
    t = sys.argv[2]
    while True:
        s = subprocess.run(["globus", "task", "show", t, "--jmespath", "[status, files, files_transferred, files_skipped]", "--format", "unix"],
                           capture_output=True, text=True, env=ENV).stdout.split()
        if s and s[0] not in ("ACTIVE", "INACTIVE"):
            break
        time.sleep(600)
    if s[0] == "SUCCEEDED" and int(s[2]) + int(s[3]) == int(s[1]):
        names = sh(f"cd {R} && ls p[0-9]*/img/*/ | grep '^c4d_'").split()
        open(os.path.join(PS, "fill", "relay", "relayed.txt"), "a").write("".join(n + "\n" for n in names))
        sh(f"rm -rf {R}/p[0-9]*/img")
        log("SUCCEEDED", s, "; names to relayed.txt:", len(names), "; NERSC copies removed")
    else:
        log("not removed:", s)
