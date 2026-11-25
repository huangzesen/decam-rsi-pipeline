"""Development runner on one GPU node (same rules as rsi/harness/ls6/finder_node.py, which is used for the sealed run):
fresh copy of the submission in node-local /tmp, `python -I finder.py --scenario SCEN --out candidates.json --budget-s
CAP` with a stripped environment (8 CPU threads, one GPU, LD_LIBRARY_PATH = the env's lib), no network when `unshare -rn`
works, killed CAP + 30 s after start. Unlike finder_node.py, each item names its own python and may set FINDER_INJECT.

Usage: python -I run_node.py ITEMS.json OUTDIR      ITEMS: [{"name", "py", "sub", "scen", "cap", "gpu", "inject"?}, ...]
Items with the same "gpu" run one after another; different GPUs run in parallel."""
import json, os, shutil, signal, socket, subprocess, sys, tempfile, threading, time


def netns_ok():
    try:
        return subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def run_one(it, outdir, netns):
    d = os.path.join(outdir, it["name"]); os.makedirs(d, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"kb_{it['name']}_", dir="/tmp")
    subc, home = os.path.join(tmp, "sub"), os.path.join(tmp, "home")
    shutil.copytree(it["sub"], subc, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.npz", "*.npy")); os.makedirs(home)
    py = it["py"]; cap = float(it["cap"])
    env = dict(PATH=os.path.dirname(py) + ":/usr/bin:/bin", HOME=home, TMPDIR=home, OMP_NUM_THREADS="8",
               MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", CUDA_VISIBLE_DEVICES=str(it["gpu"]), MPLCONFIGDIR=home,
               PYTHONDONTWRITEBYTECODE="1", LD_LIBRARY_PATH=os.path.join(os.path.dirname(os.path.dirname(py)), "lib"))
    if it.get("inject"):
        env["FINDER_INJECT"] = it["inject"]
    res = os.path.join(tmp, "candidates.json")
    cmd = (["unshare", "-rn"] if netns else []) + [py, "-I", os.path.join(subc, "finder.py"), "--scenario", it["scen"],
                                                  "--out", res, "--budget-s", str(cap)]
    t = time.time()
    with open(os.path.join(d, "finder.log"), "w") as log:
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
    if os.path.exists(res + ".debug.json"):
        shutil.copy(res + ".debug.json", os.path.join(d, "candidates.debug.json"))
    json.dump(dict(status=status, seconds=round(dt, 1), host=socket.gethostname(), gpu=it["gpu"], netns=netns),
              open(os.path.join(d, "info.json"), "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{time.strftime('%H:%M:%S')} {it['name']}: {status} {dt:.0f} s", flush=True)


def main():
    items = json.load(open(sys.argv[1])); outdir = sys.argv[2]
    netns = netns_ok()
    print("host", socket.gethostname(), "network namespace:", netns, flush=True)
    by_gpu = {}
    for it in items:
        by_gpu.setdefault(it["gpu"], []).append(it)
    th = [threading.Thread(target=lambda L=L: [run_one(it, outdir, netns) for it in L]) for L in by_gpu.values()]
    for t in th: t.start()
    for t in th: t.join()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
