#!/usr/bin/env python3
"""search38: run harness/lead_check.py (orbit fit + in-data repeat test) on a search's leads in K processes.

lead_check.py fits one lead at a time (~25 s each) and keeps one cache and one result file per run, so K copies cannot
share a run folder. `start` makes K pseudo-runs runs/<RUN>p<k>/ (config, leads ledger, the fits done so far and the
region exposure lists, all copied), stops a running lead_check.py on RUN GEN, and starts one lead_check.py per part with
--ids for its share of the leads (DBT_SITE=nersc: scenarios on NERSC scratch). `merge` copies every part's fits and
verdicts back into runs/<RUN>/lead_check/<GEN>_plan.json and <GEN>.json (by id).
2026-11-14.
Usage: python3 -I leadcheck_par.py start RUN GEN K  |  merge RUN GEN K  |  status RUN GEN K
"""
import glob, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.path.expanduser("~/anaconda3/bin/python")
cmd, run, gen, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
R = os.path.join(HERE, "runs", run)
parts = [os.path.join(HERE, "runs", f"{run}p{k}") for k in range(K)]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else []


if cmd == "start":
    subprocess.run(["pkill", "-f", f"lead_check.py {run} {gen}"])
    led = [x for x in json.load(open(os.path.join(R, "leads_sent.json"))) if x["gen"] == gen]
    for k, P in enumerate(parts):
        os.makedirs(os.path.join(P, "lead_check"), exist_ok=True)
        for f in ("config.json", "leads_sent.json"):
            shutil.copy(os.path.join(R, f), P)
        for f in glob.glob(os.path.join(R, "lead_check", "*_exposures.json")) + [os.path.join(R, "lead_check", f"{gen}_plan.json")]:
            if os.path.exists(f):
                shutil.copy(f, os.path.join(P, "lead_check"))
        ids = ",".join(x["id"] for x in led[k::K])
        log = os.path.join(P, "lead_check", gen + ".log")
        subprocess.Popen(f"nohup nice -n 5 {PY} -I {os.path.join(HERE, 'harness', 'lead_check.py')} {os.path.basename(P)} {gen} "
                         f"--ids {ids} > {log} 2>&1 &", shell=True, cwd=HERE, stdin=subprocess.DEVNULL, start_new_session=True,
                         env=dict(os.environ, DBT_SITE="nersc", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    print(f"started {K} parts for {len(led)} leads")
elif cmd in ("merge", "status"):
    plan = {d["id"]: d for d in load(os.path.join(R, "lead_check", f"{gen}_plan.json"))}
    res = {d["id"]: d for d in load(os.path.join(R, "lead_check", f"{gen}.json"))}
    for P in parts:
        plan.update({d["id"]: d for d in load(os.path.join(P, "lead_check", f"{gen}_plan.json"))})
        res.update({d["id"]: d for d in load(os.path.join(P, "lead_check", f"{gen}.json"))})
    running = subprocess.run(["pgrep", "-f", f"lead_check.py {run}p"], capture_output=True, text=True).stdout.split()
    print(f"{len(plan)} fitted, {len(res)} verdicts, {len(running)} parts running")
    if cmd == "merge":
        for name, d in ((f"{gen}_plan.json", plan), (f"{gen}.json", res)):
            p = os.path.join(R, "lead_check", name)
            json.dump(list(d.values()), open(p + ".tmp", "w"), indent=1 if name == f"{gen}.json" else None); os.replace(p + ".tmp", p)
        print("merged into", os.path.join(R, "lead_check"))
