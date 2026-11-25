"""Run a generation's finder submissions on a Lonestar6 gpu-a100-dev node (one submission per A100), bring the
outputs back; rsi_loop.py then scores them here with grade.py --runs-dir (truth never leaves the Mac).

Uses only the existing SSH master opened by the user ($SSH_CM_DIR/ls6-$TACC_USER.sock); never logs in itself.
Remote layout ($SCRATCH/rsi-planet): scen/<sid>/ (the sealed eval scenarios, uploaded once), runs/<run>/<tag>/
{subs/<gen>/, finder_node.py, job.sh, slurm-<id>.out, out/<gen>/<sid>/}. Python: $WORK/rsi-planet/env.
"""
import json, os, re, subprocess, time

SOCK = os.path.expanduser("$SSH_CM_DIR/ls6-$TACC_USER.sock")
HOST = "$TACC_USER@ls6.tacc.utexas.edu"
SSH_OPTS = ["-S", SOCK, "-o", "ControlMaster=no", "-o", "BatchMode=yes", "-o", "ProxyCommand=false"]
SCR = "$TACC_SCRATCH/rsi-planet"
ENVPY = "$TACC_WORK/ls6/rsi-planet/env/bin/python"
ACCOUNT, PARTITION, MAX_WALL_MIN = "$LS6_ACCOUNT", "gpu-a100-dev", 120
RSYNC = "/opt/homebrew/bin/rsync"
NET = re.compile(r"^\s*(?:import|from)\s+(socket|urllib|requests|http|httpx|aiohttp|ftplib|smtplib|paramiko|astroquery|"
                 r"webbrowser|pyvo|urllib3)\b", re.M)
HERE = os.path.dirname(os.path.abspath(__file__))


def ssh(cmd, inp=None, timeout=120):
    return subprocess.run(["ssh", *SSH_OPTS, HOST, "export LC_ALL=C LANG=C; " + cmd], input=inp, capture_output=True,
                          text=True, timeout=timeout)


def rsync(src, dst, timeout=3600):
    e = "ssh " + " ".join(SSH_OPTS)
    r = subprocess.run([RSYNC, "-a", "--delete", "-e", e, src, dst], capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"rsync {src} -> {dst}: {r.stderr[-500:]}")


def available(sids):
    """(True, '') when the master is alive, the Python env exists and every eval scenario is uploaded."""
    if not os.path.exists(SOCK):
        return False, "no LS6 master socket"
    if subprocess.run(["ssh", "-S", SOCK, "-O", "check", HOST], capture_output=True).returncode:
        return False, "LS6 master not alive"
    try:
        r = ssh(f"test -x {ENVPY} && echo env; for s in {' '.join(sids)}; do test -f {SCR}/scen/$s/exposures.csv && echo $s; done")
    except subprocess.TimeoutExpired:
        return False, "LS6 ssh timed out"
    got = r.stdout.split()
    if "env" not in got:
        return False, "no Python env on LS6"
    miss = [s for s in sids if s not in got]
    return (False, f"scenarios not uploaded: {miss}") if miss else (True, "")


def lint(sub):
    """Network imports in a submission (LS6 cannot sandbox the network if namespaces are off)."""
    hits = []
    for dp, _, fs in os.walk(sub):
        for f in fs:
            if f.endswith(".py"):
                for m in NET.finditer(open(os.path.join(dp, f), errors="replace").read()):
                    hits.append(f"{os.path.relpath(os.path.join(dp, f), sub)}: {m.group(1)}")
    return hits


def run_remote(run, gens_dir, names, sids, cap_s, log):
    """Upload submissions, run one Slurm job, wait, fetch outputs to gens_dir/<name>/ls6_runs/. Returns the job id."""
    jid, R = submit(run, gens_dir, names, sids, cap_s, log)
    return wait_fetch(jid, R, gens_dir, names, log)


def submit(run, gens_dir, names, sids, cap_s, log):
    """Upload submissions and submit one Slurm job. Returns (job id, remote folder)."""
    tag = time.strftime("%Y%m%d-%H%M%S") + "-" + names[0]
    R = f"{SCR}/runs/{run}/{tag}"
    ssh(f"mkdir -p {R}/subs {R}/out")
    for g in names:
        rsync(os.path.join(gens_dir, g, "submission") + "/", f"{HOST}:{R}/subs/{g}/")
    rsync(os.path.join(HERE, "finder_node.py"), f"{HOST}:{R}/finder_node.py")
    runs = len(names) * len(sids)
    for nodes in (1, 2):                              # 3 A100 per node; a 2-node job when one node would need > 110 min
        wall = int(-(-runs // (3 * nodes)) * (cap_s + 45) / 60) + 10
        if wall <= 110:
            break
    wall = min(MAX_WALL_MIN, wall)
    job = f"""#!/bin/bash
#SBATCH -J rsi-{run}-{names[0][:6]}
#SBATCH -p {PARTITION}
#SBATCH -A {ACCOUNT}
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node=1
#SBATCH -t {wall // 60:02d}:{wall % 60:02d}:00
#SBATCH -o {R}/slurm-%j.out
nvidia-smi -L
srun -N {nodes} --ntasks-per-node=1 {ENVPY} -I {R}/finder_node.py --subs {','.join(f'{g}={R}/subs/{g}' for g in names)} --scen-root {SCR}/scen \\
  --sids {','.join(sids)} --cap-s {cap_s} --out {R}/out
"""
    ssh(f"cat > {R}/job.sh", inp=job)
    r = ssh(f"cd {R} && sbatch --parsable job.sh")
    ids = [l.strip().split(";")[0] for l in r.stdout.splitlines() if l.strip().split(";")[0].isdigit()]   # TACC prints a banner first
    if not ids:
        raise RuntimeError(f"sbatch failed: {r.stdout[-800:]} {r.stderr[-400:]}")
    jid = ids[-1]
    log(f"LS6 job {jid}: {len(names)} submissions x {len(sids)} scenarios, {nodes} node(s), wall {wall} min, cap {cap_s:g} s, {R}")
    return jid, R


def wait_fetch(jid, R, gens_dir, names, log):
    """Wait for LS6 job JID (remote folder R) to end, then fetch out/<name>/ for each name."""
    t0, started, down = time.time(), None, None
    while True:
        time.sleep(60)
        try:
            r = ssh(f"squeue -h -j {jid} -o %T", timeout=60)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 255:                       # ssh itself failed (master gone): the job runs on; keep asking
            if not down:
                down = time.time(); log(f"LS6 unreachable ({r.stderr.strip()[-200:]}); waiting for the master to come back")
            continue
        if down:
            log(f"LS6 reachable again after {(time.time() - down) / 60:.0f} min"); down = None
        st = r.stdout.strip()
        if st == "RUNNING" and not started:
            started = time.time(); log(f"LS6 job {jid} running after {(started - t0) / 60:.0f} min in the queue")
        if not st:
            break
        if not started and time.time() - t0 > 24 * 3600:
            ssh(f"scancel {jid}")
            raise RuntimeError(f"LS6 job {jid} still pending after 24 h; cancelled")
    return fetch(jid, R, gens_dir, names, log)


def fetch(jid, R, gens_dir, names, log):
    """Log how job JID ended and bring out/<name>/ back to gens_dir/<name>/ls6_runs/."""
    acct = ssh(f"sacct -n -X -j {jid} -o State,Elapsed").stdout.split()
    log(f"LS6 job {jid} finished: {' '.join(acct)}")
    for g in names:
        dst = os.path.join(gens_dir, g, "ls6_runs")
        os.makedirs(dst, exist_ok=True)
        try:
            rsync(f"{HOST}:{R}/out/{g}/", dst + "/")
        except RuntimeError as e:
            log(str(e))
    rsync(f"{HOST}:{R}/slurm-{jid}.out", os.path.join(gens_dir, f"ls6-{jid}.out"))
    return jid
