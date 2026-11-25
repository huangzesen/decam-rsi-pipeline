"""Run one agent session of an RSI round on the GCP VM instead of the Mac (2026-11-10, ; chose "GCP, CPU only").

rsi_loop.run_agent() calls run() when the run's config.json has "agent_on": "gcp". The round is prepared on the Mac
as before (history, instruction, tools); this module copies the round folder to the VM, runs Claude Code there in a
container, keeps the transcript on the Mac current while it runs, and copies the submission and subagent transcripts
back. Grading, the sealed answers and every private file stay on the Mac: the VM only ever holds what the agent may see.

The container (image rsi-agent:r2, /data/rsi-planet/img/Dockerfile on the VM) is the Linux stand-in for the Mac's
sandbox-exec profile:
- network: the internal docker network rsi-agent-net, whose only other member is the proxy container rsi-relay
  (socat to the VM's relay forwarder $DOCKER_HOST_IP:$RELAY_FORWARD_PORT -> SSH tunnel -> the Mac's relay); no internet, no host;
- files: the round folder (read-write) and the development scenarios (read-only), mounted at the Mac's paths, so
  every path in the instruction and in earlier notes stays valid; the grader's packages (CPU torch) live in the image
  at the Mac's interpreter path;
- limits: 16 CPUs, 48 GB memory, its own PID namespace (it cannot signal anything outside), runs as the VM user.
The transcript is the container's stdout (docker logs), which the agent cannot rewrite.
"""
import json, os, shlex, subprocess, time

HOST = "$VM_HOST"                       # ssh alias of the VM $VM_HOST (~/.ssh/config)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", HOST]
VM_BASE = "/data/rsi-planet"            # on the VM: scen_dev/, <run>/<gname>/, secret/, img/
IMAGE, NET = "rsi-agent:r2", "rsi-agent-net"
CPUS, MEM = "16", "48g"
POLL_S = 120


def sh(cmd, timeout=600, **kw):
    return subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout, **kw)


def rsync(args, timeout=1800):
    return subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=30"] + args,
                          capture_output=True, text=True, timeout=timeout)


def ready():
    """(ok, why): the VM answers, the image, network, relay proxy and dev data are there and the relay is reachable."""
    try:
        r = sh(f"docker image inspect {IMAGE} >/dev/null && docker network inspect {NET} >/dev/null && "
               f"[ \"$(docker inspect -f '{{{{.State.Running}}}}' rsi-relay)\" = true ] && test -d {VM_BASE}/scen_dev && "
               f"docker run --rm --network {NET} {IMAGE} curl -s -m 10 -o /dev/null -w '%{{http_code}}' http://rsi-relay:$RELAY_PORT/health",
               timeout=120)
    except subprocess.TimeoutExpired:
        return False, "VM did not answer in 120 s"
    return (r.returncode == 0 and r.stdout.strip().endswith("200")), (r.stdout + r.stderr).strip()[-300:]


def _fetch_logs(name, out_path, err_path, full=False):
    """Append what is new in the container's stdout/stderr to the Mac's transcript (byte offsets), or refetch all."""
    for stream, path in (("1", out_path), ("2", err_path)):
        have = 0 if full or not os.path.exists(path) else os.path.getsize(path)
        redirect = "2>/dev/null" if stream == "1" else "2>&1 >/dev/null"
        try:
            p = subprocess.run(SSH + [f"docker logs {name} {redirect} | tail -c +{have + 1}"], capture_output=True,
                               timeout=900)
        except subprocess.TimeoutExpired:
            continue
        if p.returncode == 0 and not (full and os.path.exists(path) and len(p.stdout) < os.path.getsize(path)):
            with open(path, "wb" if full else "ab") as f:
                f.write(p.stdout)


def run(run_name, gname, gdir, ws, env, cmd, hours, pg, log=print):
    """Run the prepared round on the VM. env: the environment the Mac would have used (Mac paths); cmd: the claude
    argv after the binary. Returns the exit code, "timeout", or None when the VM could not start it (the caller then
    runs the round on the Mac)."""
    ok, why = ready()
    if not ok:
        log(f"{gname}: GCP VM not ready ({why}); running this round on the Mac")
        return None
    vdir = f"{VM_BASE}/{run_name}/{gname}"
    name = f"rsi-{run_name}-{gname}"
    # the round folder (history, instruction, tools, empty home/tmp/claude_config); agent.sb is the Mac's sandbox
    sh(f"mkdir -p {VM_BASE}/{run_name}", timeout=60)
    r = rsync(["--delete", "--exclude", "agent.sb", gdir + "/", f"{HOST}:{vdir}/"])
    if r.returncode != 0:
        log(f"{gname}: copy to the VM failed ({r.stderr.strip()[-200:]}); running this round on the Mac")
        return None
    # the relay key goes into a 0600 env file on the VM through stdin; it is never on a command line or printed
    secret = f"{VM_BASE}/$RUN_SECRETS_DIR/{run_name}.env"
    p = subprocess.run(SSH + [f"umask 077; cat > {secret}"], input=f"ANTHROPIC_AUTH_TOKEN={env['ANTHROPIC_AUTH_TOKEN']}\n",
                       text=True, capture_output=True, timeout=60)
    if p.returncode != 0:
        log(f"{gname}: could not write the env file on the VM; running this round on the Mac")
        return None
    venv = {k: v for k, v in env.items() if k not in ("ANTHROPIC_AUTH_TOKEN", "PYTORCH_ENABLE_MPS_FALLBACK")}
    venv.update(ANTHROPIC_BASE_URL="http://rsi-relay:$RELAY_PORT", OMP_NUM_THREADS=CPUS, MKL_NUM_THREADS=CPUS,
                OPENBLAS_NUM_THREADS=CPUS, LANG="C.UTF-8",
                PATH=f"{env['PATH'].split(':')[0]}:/usr/local/bin:/usr/bin:/bin")     # the Mac's python dir first
    dev_dir = env["RSI_DEV_DIR"]
    dock = ["docker", "run", "-d", "--name", name, "--network", NET, "--user", "1001:1002",
            "--cpus", CPUS, "--memory", MEM, "--memory-swap", MEM, "--pids-limit", "8192", "--shm-size", "8g",
            "--log-driver", "json-file", "--log-opt", "max-size=4g", "--log-opt", "max-file=1",
            "-v", f"{vdir}:{gdir}", "-v", f"{VM_BASE}/scen_dev:{dev_dir}:ro", "-w", ws, "--env-file", secret]
    for k, v in venv.items():
        dock += ["-e", f"{k}={v}"]
    dock += [IMAGE, "claude"] + cmd
    # a container of this name left from an aborted attempt is removed first (only this round's own name)
    p = sh(f"docker rm -f {name} >/dev/null 2>&1; " + shlex.join(dock), timeout=180)
    if p.returncode != 0:
        log(f"{gname}: container did not start ({p.stderr.strip()[-300:]}); running this round on the Mac")
        return None
    log(f"{gname}: agent session started on the GCP VM (container {name})")
    out_path, err_path = os.path.join(pg, "transcript.jsonl"), os.path.join(pg, "agent.err")
    open(out_path, "wb").close(); open(err_path, "wb").close()
    deadline, rc = time.time() + hours * 3600, None
    while True:
        time.sleep(POLL_S)
        _fetch_logs(name, out_path, err_path)
        try:
            st = sh(f"docker inspect -f '{{{{.State.Running}}}} {{{{.State.ExitCode}}}}' {name}", timeout=120)
        except subprocess.TimeoutExpired:
            st = None
        if st is not None and st.returncode == 0 and st.stdout.split()[0] == "false":
            rc = int(st.stdout.split()[1]); break
        if time.time() > deadline:       # as on the Mac: SIGTERM, 10 s, SIGKILL
            try:
                sh(f"docker stop -t 10 {name}", timeout=180)
            except subprocess.TimeoutExpired:
                pass
            rc = "timeout"; break
    _fetch_logs(name, out_path, err_path, full=True)
    # submission back (same files the Mac's copy into the private record keeps), and Claude Code's own folder for the
    # subagent transcripts that run_agent() collects
    for sub in ("workspace/submission", "claude_config"):
        r = rsync(["--delete", "--exclude", "__pycache__", "--exclude", "*.pyc", "--exclude", "*.npz", "--exclude",
                   "*.npy", "--exclude", "*.fits*", f"{HOST}:{vdir}/{sub}/", os.path.join(gdir, sub) + "/"])
        if r.returncode != 0:
            log(f"{gname}: copying {sub} back failed: {r.stderr.strip()[-300:]}")
    try:
        sh(f"docker rm {name} >/dev/null 2>&1; "
           # keep the last three rounds' folders on the VM (scratch can be large); older ones go
           f"ls -d {VM_BASE}/{run_name}/gen* 2>/dev/null | sort | head -n -3 | xargs -r rm -rf", timeout=600)
    except subprocess.TimeoutExpired:
        pass
    log(f"{gname}: agent session on the GCP VM ended ({rc})")
    return rc
