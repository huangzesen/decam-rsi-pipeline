"""One short LS6 test: the starter (gen000) on two sealed scenarios, to check the env, CUDA and speed against the
Mac's grade log. Output in rsi/runs/RUN/ls6_test/. Usage: python -I test_job.py RUN [GEN]"""
import json, os, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ls6

run, GEN = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "gen001")
RUNS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "runs", run)
T = os.path.join(RUNS, "ls6_test")
shutil.copytree(os.path.join(RUNS, "gens", GEN, "submission"), os.path.join(T, GEN, "submission"), dirs_exist_ok=True)
cfg = json.load(open(os.path.join(RUNS, "config.json")))
sids = ["scen11", "scen06"]
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
log(f"available: {ls6.available(sids)}")
ls6.run_remote(run + "-test", T, [GEN], sids, cfg["cap_s"], log)
for sid in sids:
    f = os.path.join(T, GEN, "ls6_runs", sid, "info.json")
    i = json.load(open(f)) if os.path.exists(f) else {}
    c = os.path.join(T, GEN, "ls6_runs", sid, "candidates.json")
    n = len(json.load(open(c))["candidates"]) if os.path.exists(c) else None
    log(f"{sid}: {i.get('status')} {i.get('seconds')} s, {n} candidates, gpu {i.get('gpu')}, netns {i.get('netns')}")
    log("   " + (i.get("log_tail") or "")[-400:].replace("\n", "\n   "))
