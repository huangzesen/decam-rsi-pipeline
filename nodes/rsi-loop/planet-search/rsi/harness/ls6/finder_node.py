"""Run finder submissions on one Lonestar6 GPU node (inside a Slurm job): every (submission, scenario) run is a\nwork item, taken by the next free GPU (each run has a GPU to itself).

For each submission and each scenario, as grade.py does on the Mac: a fresh copy of the submission in node-local /tmp,
    python -I finder.py --scenario SCEN --out candidates.json --budget-s CAP
with 8 CPU threads, killed CAP + 30 s after start; no network when unprivileged namespaces work (unshare -rn).
Writes OUT/<name>/<sid>/{candidates.json, info.json}. No truth is on this machine; scoring is done on the Mac
(grade.py --runs-dir).

Usage: python -I finder_node.py --subs NAME=DIR,NAME=DIR --scen-root DIR --sids s1,s2 --cap-s 300 --out DIR
"""
import argparse, json, os, shutil, signal, socket, subprocess, sys, tempfile, threading, time


def netns_ok():
    try:
        return subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def run_one(name, sub, scen, sid, cap, out, gpu, netns):
    d = os.path.join(out, name, sid)
    os.makedirs(d, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"rsi_{name}_{sid}_", dir="/tmp")
    subc, home = os.path.join(tmp, "sub"), os.path.join(tmp, "home")
    shutil.copytree(sub, subc); os.makedirs(home)
    py = sys.executable
    env = dict(PATH=os.path.dirname(py) + ":/usr/bin:/bin", HOME=home, TMPDIR=home, OMP_NUM_THREADS="8",
               MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", CUDA_VISIBLE_DEVICES=str(gpu), MPLCONFIGDIR=home,
               PYTHONDONTWRITEBYTECODE="1",
               # the env's own libstdc++ first: torch otherwise loads the system's old one and scipy.spatial fails
               LD_LIBRARY_PATH=os.path.join(os.path.dirname(os.path.dirname(py)), "lib"))
    res = os.path.join(tmp, "candidates.json")
    cmd = (["unshare", "-rn"] if netns else []) + [py, "-I", os.path.join(subc, "finder.py"), "--scenario", scen,
                                                  "--out", res, "--budget-s", str(cap)]
    t = time.time()
    with open(os.path.join(tmp, "finder.log"), "w") as log:
        p = subprocess.Popen(cmd, cwd=subc, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status = "ok"
        try:
            rc = p.wait(timeout=cap + 30)
            if rc != 0:
                status = f"exit {rc}"
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); status = "timeout"
    dt = time.time() - t
    if os.path.exists(res):
        shutil.copy(res, os.path.join(d, "candidates.json"))
    tail = open(os.path.join(tmp, "finder.log"), errors="replace").read()[-3000:]
    json.dump(dict(status=status, seconds=round(dt, 1), log_tail=tail, host=socket.gethostname(), gpu=gpu, netns=netns),
              open(os.path.join(d, "info.json"), "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{time.strftime('%H:%M:%S')} {name} {sid}: {status} {dt:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", required=True); ap.add_argument("--scen-root", required=True)
    ap.add_argument("--sids", required=True); ap.add_argument("--cap-s", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    subs = [x.split("=", 1) for x in a.subs.split(",")]
    sids = a.sids.split(",")
    netns = netns_ok()
    print("network namespace:", netns, flush=True)
    ngpu = max(1, len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")]))
    # every (submission, scenario) run is one work item; one worker per GPU takes the next item when it is free
    import queue
    q = queue.Queue()
    shard, nshards = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))   # one task per node
    items = [(name, sub, sid) for sid in sids for name, sub in subs]
    for k, it in enumerate(items):
        if k % nshards == shard:
            q.put(it)
    print(f"node {socket.gethostname()}: shard {shard} of {nshards}, {q.qsize()} runs on {ngpu} GPUs", flush=True)

    def worker(gpu):
        while True:
            try:
                name, sub, sid = q.get_nowait()
            except queue.Empty:
                return
            run_one(name, sub, os.path.join(a.scen_root, sid), sid, a.cap_s, a.out, gpu, netns)

    th = [threading.Thread(target=worker, args=(g,)) for g in range(ngpu)]
    for t in th: t.start()
    for t in th: t.join()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
