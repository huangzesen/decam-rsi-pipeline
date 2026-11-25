"""Run a generation's finder submissions on NERSC Perlmutter (A100), the second grading site next to Lonestar6.

Uses only the existing SSH master opened by the user ($SSH_CM_DIR/nersc-$NERSC_USER.sock); never logs in itself. Same remote
layout as ls6.py, with $SCRATCH/rsi-planet. A Site tries the `interactive` QoS (one node, up to 4 h, salloc only: it
grants a node within seconds or fails) every 10 min and also queues in `debug` (30 min limit, so as many nodes as keep
every run inside it). When one of its jobs starts, the other is cancelled. No truth is on this machine.
"""
import math, os, subprocess, time

from ls6 import HERE, RSYNC, lint  # noqa: F401  (same RSYNC and node runner as Lonestar6)

SOCK = os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock")
HOST = "$NERSC_USER@perlmutter.nersc.gov"
SSH_OPTS = ["-S", SOCK, "-o", "ControlMaster=no", "-o", "BatchMode=yes", "-o", "ProxyCommand=false"]
SCR = "$NERSC_SCRATCH/rsi-planet"
ENVPY = f"{SCR}/env/bin/python"
ACCOUNT = "$NERSC_GPU_ACCOUNT"
GPUS, DEBUG_MIN, RETRY_S = 4, 30, 600
LABEL = "A100 (Perlmutter)"


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
        return False, "no NERSC master socket"
    if subprocess.run(["ssh", "-S", SOCK, "-O", "check", HOST], capture_output=True).returncode:
        return False, "NERSC master not alive"
    try:
        r = ssh(f"test -x {ENVPY} && echo env; for s in {' '.join(sids)}; do test -f {SCR}/scen/$s/exposures.csv && echo $s; done")
    except subprocess.TimeoutExpired:
        return False, "NERSC ssh timed out"
    got = r.stdout.split()
    if "env" not in got:
        return False, "no Python env on NERSC"
    miss = [s for s in sids if s not in got]
    return (False, f"scenarios not uploaded: {miss}") if miss else (True, "")


class Site:
    label = LABEL

    def __init__(self, run, gens_dir, names, sids, cap_s, log):
        self.gens_dir, self.names, self.log = gens_dir, names, log
        tag = time.strftime("%Y%m%d-%H%M%S") + "-" + names[0]
        self.R = R = f"{SCR}/runs/{run}/{tag}"
        ssh(f"mkdir -p {R}/subs {R}/out")
        for g in names:
            rsync(os.path.join(gens_dir, g, "submission") + "/", f"{HOST}:{R}/subs/{g}/")
        rsync(os.path.join(HERE, "finder_node.py"), f"{HOST}:{R}/finder_node.py")
        self.jname = f"rsi-{run}-{names[0][:6]}"
        self.step = (f"--cpu-bind=none -c 128 --gpus-per-node={GPUS} {ENVPY} -I {R}/finder_node.py "
                     f"--subs {','.join(f'{g}={R}/subs/{g}' for g in names)} --scen-root {SCR}/scen "
                     f"--sids {','.join(sids)} --cap-s {cap_s} --out {R}/out")
        self.runs, self.cap_s = len(names) * len(sids), cap_s
        self.t0, self.t_int, self.k_int, self.dbg, self.started = time.time(), 0, 0, False, None

    def _wall(self, nodes):                       # minutes for every run to finish on this many nodes
        return math.ceil(self.runs / (GPUS * nodes)) * (self.cap_s + 45) / 60

    def _interactive(self):
        self.k_int += 1
        wall = int(self._wall(1)) + 10
        R = self.R
        # started directly by the login shell with its streams redirected, or ssh stays open until salloc ends
        ssh(f"cd {R} || exit 1; setsid nohup salloc -J {self.jname}-int -A {ACCOUNT} -C gpu -q interactive -N 1 -t {wall} "
            f"--gpus-per-node={GPUS} --immediate=90 srun -N 1 -n 1 {self.step} "
            f"> int.{self.k_int}.log 2>&1 < /dev/null & disown; echo ok", timeout=30)
        self.t_int = time.time()

    def _debug(self):
        self.dbg = True
        for nodes in range(1, 9):                 # the fewest nodes that keep every run inside debug's wall limit
            if self._wall(nodes) + 5 <= DEBUG_MIN:
                break
        else:
            self.log("NERSC debug: runs do not fit its 30 min limit; not queueing there")
            return
        wall = min(DEBUG_MIN, int(self._wall(nodes)) + 6)
        job = f"""#!/bin/bash
#SBATCH -J {self.jname}-dbg
#SBATCH -A {ACCOUNT}
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={GPUS}
#SBATCH -t {wall}
#SBATCH -o {self.R}/slurm-%j.out
nvidia-smi -L
srun -N {nodes} --ntasks-per-node=1 {self.step}
"""
        ssh(f"cat > {self.R}/job.sh", inp=job)
        r = ssh(f"cd {self.R} && sbatch job.sh")
        self.log(f"NERSC debug: {nodes} node(s), wall {wall} min: {r.stdout.strip()[-60:]} {r.stderr.strip()[-200:]}")

    def tick(self):
        """Called once a minute while no site has started."""
        now = time.time()
        if now - self.t_int >= RETRY_S:
            self._interactive()
        if not self.dbg and now - self.t0 >= 150:
            self._debug()

    def _rows(self):
        r = ssh(f"squeue --me -h -n {self.jname}-int,{self.jname}-dbg -o '%i %j %T'", timeout=60)
        if r.returncode == 255:
            return None
        return [l.split() for l in r.stdout.splitlines() if len(l.split()) == 3]

    def state(self):
        """PENDING, RUNNING, GONE (this site's job has ended) or ? (cannot tell now)."""
        try:
            rows = self._rows()
        except subprocess.TimeoutExpired:
            return "?"
        if rows is None:
            return "?"
        if self.started:
            return "RUNNING" if any(x[0] == self.started for x in rows) else "GONE"
        run = [x for x in rows if x[2] in ("RUNNING", "COMPLETING")]
        if run:
            self.started = run[0][0]
            self.log(f"NERSC job {self.started} ({run[0][1]}) started")
            for x in rows:
                if x[0] != self.started:
                    ssh(f"scancel {x[0]}")
            return "RUNNING"
        return "PENDING"

    def cancel(self):
        self.t_int = float("inf"); self.dbg = True
        ssh(f"scancel -u $USER -n {self.jname}-int; scancel -u $USER -n {self.jname}-dbg")

    def fetch(self):
        acct = ssh(f"sacct -n -X -j {self.started} -o State,Elapsed,NNodes,QOS").stdout.split()
        self.log(f"NERSC job {self.started} finished: {' '.join(acct)}")
        for g in self.names:
            dst = os.path.join(self.gens_dir, g, "ls6_runs")
            os.makedirs(dst, exist_ok=True)
            try:
                rsync(f"{HOST}:{self.R}/out/{g}/", dst + "/")
            except RuntimeError as e:
                self.log(str(e))
