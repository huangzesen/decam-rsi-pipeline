#!/usr/bin/env python3
"""Grade decam-big-tno submissions on NERSC Perlmutter (2026-11-13, after Vista's gb queue held a
round for 10 h). One `interactive` node (4 x A100 40 GB, up to 4 h, granted at once or refused: retried every 10 min),
`harness/nersc/grade_node.sh` runs every (submission, sealed region) pair, four at a time, one A100 each, no network,
BUDGET_S per region; only outputs and timings come back; `harness/grade.py` scores them here against private/truth.
Uses only the SSH master the user opened ($SSH_CM_DIR/nersc-$NERSC_USER.sock); never logs in itself.

    grade(job, items, sids, budget, log)   items = [(tag, local_submission_dir, local_grade_dir)] -> {tag: grade dict}
CLI:  python3 -I harness/nersc_grade.py JOB TAG=SUBDIR=GRADEDIR [...] [--budget 3600]   (sealed regions from runs/r1/config.json)
"""
import argparse, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH = os.path.join(HERE, "private/truth")
NPY = os.path.expanduser("~/anaconda3/bin/python")
SOCK = os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock")
HOST = "$NERSC_USER@perlmutter.nersc.gov"
OPTS = ["-S", SOCK, "-o", "ControlMaster=no", "-o", "BatchMode=yes", "-o", "ProxyCommand=false", "-o", "ConnectTimeout=20"]
D = "$NERSC_SCRATCH/dbt"
ACCOUNT = "$NERSC_GPU_ACCOUNT"


def ssh(cmd, timeout=300):
    for i in range(30):
        p = subprocess.run(["ssh", *OPTS, HOST, "export LC_ALL=C LANG=C; " + cmd], capture_output=True, text=True, timeout=timeout)
        if p.returncode != 255:
            return p.returncode, p.stdout + p.stderr
        time.sleep(300)                                  # master gone: wait for the user's login (sshproxy + MFA)
    raise RuntimeError("NERSC cannot be reached (no master)")


def rsync(src, dst):
    e = "ssh " + " ".join(OPTS)
    for i in range(10):
        p = subprocess.run(["rsync", "-a", "--partial", "-e", e, src, dst], capture_output=True, text=True)
        if p.returncode == 0:
            return
        time.sleep(60)
    raise RuntimeError(f"rsync {src} {dst}: {p.stderr[-300:]}")


def grade(job, items, sids, budget, log=print, wall=None):
    R = f"{D}/runs/{job}"
    rc, o = ssh(f"test -f {R}/DONE && echo done; squeue -h -u $NERSC_USER -n dbt-{job} -o %T")
    if "done" not in o and o.strip():
        log(f"nersc {job}: allocation already running, waiting for it")   # resume: never rewrite the list a node is reading
    if "done" not in o:
        ssh(f"mkdir -p {R}")
        for tag, sub, _ in items:
            rc, o = ssh(f"test -d {R}/{tag}/sub && echo yes")
            if "yes" not in o:
                ssh(f"mkdir -p {R}/{tag}"); rsync(sub.rstrip("/") + "/", f"{HOST}:{R}/{tag}/sub/")
        lines = "".join(f"{tag} {sid}\\n" for sid in sids for tag, _, _ in items)       # region-major: every GPU busy early
        ssh(f"test -f {R}/list.txt || {{ printf '{lines}' > {R}/list.txt.tmp && mv {R}/list.txt.tmp {R}/list.txt; }}; "
            f"test -f {R}/grade_node.sh || cp {D}/bin/grade_node.sh {R}/grade_node.sh")
        n = len(items) * len(sids)
        w = wall or min(240, int((n + 3) // 4 * (budget + 120) / 60) + 20)
        t = f"{w // 60:02d}:{w % 60:02d}:00"
        tries = 0
        while True:
            rc, o = ssh(f"test -f {R}/DONE && echo done; squeue -h -u $NERSC_USER -n dbt-{job} -o %T")
            if "done" in o:
                break
            if o.strip():                                # allocation alive (RUNNING / PENDING / COMPLETING)
                time.sleep(120); continue
            tries += 1
            if tries > 1:
                log(f"nersc {job}: no allocation running (try {tries}); last salloc log: "
                    + ssh(f"tail -2 {R}/salloc.log 2>/dev/null")[1].strip().replace(chr(10), ' | ')[-200:])
            ssh(f"cd {R} && rm -rf locks && (setsid nohup salloc -J dbt-{job} -A {ACCOUNT} -C gpu -q interactive -N 1 -t {t} "
                f"--gpus-per-node=4 --immediate=90 srun -N 1 -n 1 --gpus-per-node=4 bash {R}/grade_node.sh {R} {budget} "
                f"> {R}/salloc.log 2>&1 < /dev/null &); sleep 100; tail -3 {R}/salloc.log")
            rc, o = ssh(f"test -f {R}/DONE && echo done; squeue -h -u $NERSC_USER -n dbt-{job} -o %T")
            if "done" in o or o.strip():
                log(f"nersc {job}: node granted ({n} runs, wall {t})")
                continue
            time.sleep(600)                              # refused (no node at once, or the 2-job interactive limit)
    out = {}
    for tag, sub, gd in items:
        local = os.path.join(os.path.dirname(gd.rstrip("/")), os.path.basename(gd.rstrip("/")) + "_runs")
        shutil.rmtree(local, ignore_errors=True); os.makedirs(local)
        for d in ("out", "timing", "log"):
            rsync(f"{HOST}:{R}/{tag}/{d}/", os.path.join(local, d) + "/")
        timing = {}
        for sid in sids:
            p = os.path.join(local, "timing", sid)
            if os.path.exists(p):
                timing[sid] = float(open(p).read().split()[0])
        json.dump(timing, open(os.path.join(local, "timing.json"), "w"))
        p = subprocess.run([NPY, "-I", os.path.join(HERE, "harness/grade.py"), os.path.join(local, "out"), TRUTH, gd,
                            "--sids", ",".join(sids), "--timing", os.path.join(local, "timing.json")], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"grade.py failed for {tag}: {p.stderr[-400:]}")
        out[tag] = json.load(open(os.path.join(gd, "grade.json")))
    ssh(f"rm -rf {R}/*/out {R}/*/work")                  # candidates stay only on the Mac
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("job"); ap.add_argument("items", nargs="+"); ap.add_argument("--budget", type=int, default=3600)
    a = ap.parse_args()
    sids = json.load(open(os.path.join(HERE, "runs/r1/config.json")))["sids"]
    items = [tuple(x.split("=", 2)) for x in a.items]
    say = lambda m: print(time.strftime("%Y-%m-%d %H:%M:%S"), m, flush=True)
    for tag, d in grade(a.job, items, sids, a.budget, say).items():
        say(f"{tag}: reward {d['reward']:.4f} " + json.dumps({s: (v['known_found'], v['known_findable'], round(v['seconds'] or 0)) for s, v in d['per_scenario'].items()}))
