#!/usr/bin/env python3
"""search38: run the slow half of harness/lead_check.py (the orbit fit of each lead, ~30 s) on any machine, in many
processes, without NERSC access. User, 2026-11-16, on checking all 25,000 candidates below the confidence cut:
"155 CPU hour is fine no?? we parallelly do this on mac and gcp anyway".

lead_check.py reads the leads' detections from NERSC, fits, then asks NERSC for the detections near each predicted place.
Only the Mac can reach NERSC, so:
  prep TAG              (Mac) read the detections of every lead of pseudo-generation TAG (runs/s38/leads_sent.json) from
                        NERSC and write a bundle runs/s38/fitfarm/TAG/: leads.json, R1.json (the detections),
                        exp/<sid>_exposures.json, letter.json, code/ (lead_check.py + the four linker modules it imports)
  work BUNDLE K k       (anywhere) fit share k of K (leads[k::K]) with lead_check.lead_fit, one JSON line per lead appended
                        to BUNDLE/plan/part_<k>_of_<K>_<host>.jsonl; skips leads already in any plan file, so it can be
                        restarted and shares re-cut; skips the whole share if another host has a plan file for it (two
                        machines can then walk the shares from opposite ends, with plan/ synced between them).
                        FITFARM_MAXLOAD=x: before each lead, wait while the 1-minute load per CPU is above a limit drawn
                        once per process in [x - 0.13, x] (shared hosts: the GCP VM's launch guard trips at 0.85)
  collect TAG           (Mac) merge BUNDLE/plan/*.jsonl into runs/s38/lead_check/TAG_plan.json, lead_check.py's fit cache
Then on the Mac: DBT_SITE=nersc ~/anaconda3/bin/python -I harness/lead_check.py s38 TAG   (no fits left: repeat test only).
The bundle holds no labels: detections' places and fluxes, exposure lists, code.
"""
import glob, json, os, random, shutil, socket, sys, time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
cmd = sys.argv[1]


def lines(B):
    out = {}
    for f in sorted(glob.glob(os.path.join(B, "plan", "*.jsonl"))):
        for ln in open(f):
            try:
                d = json.loads(ln)
            except ValueError:          # a line cut by a stop
                continue
            out[d["id"]] = d
    return out


if cmd == "work":
    B = os.path.abspath(sys.argv[2]); K, k = int(sys.argv[3]), int(sys.argv[4])
    sys.path.insert(0, os.path.join(B, "code"))
    import lead_check as LC
    host = socket.gethostname().split(".")[0]; mine = os.path.join(B, "plan", f"part_{k:03d}_of_{K}_{host}.jsonl")
    if [f for f in glob.glob(os.path.join(B, "plan", f"part_{k:03d}_of_{K}_*.jsonl")) if f != mine]:
        print("share", k, "of", K, "is another host's", flush=True); sys.exit(0)
    mx = float(os.environ.get("FITFARM_MAXLOAD", "0")); lim = mx - 0.13 * random.random(); ncpu = os.cpu_count()
    leads = json.load(open(os.path.join(B, "leads.json")))[k::K]
    letter = json.load(open(os.path.join(B, "letter.json")))
    need = {x["sid"] for x in leads}
    R1 = {s: v for s, v in json.load(open(os.path.join(B, "R1.json"))).items() if s in need}
    done = set(lines(B)); os.makedirs(os.path.join(B, "plan"), exist_ok=True)
    with open(mine, "a") as fo:
        for x in leads:
            if x["id"] in done:
                continue
            while mx and os.getloadavg()[0] / ncpu > lim:
                time.sleep(20 + 40 * random.random())
            try:
                one = LC.lead_fit(x, R1, os.path.join(B, "exp"), {}, [], [], letter)
            except Exception as err:
                one = dict(id=x["id"], fail=f"fit failed: {type(err).__name__}")
            fo.write(json.dumps(one) + "\n"); fo.flush()
    print("share", k, "of", K, "done", flush=True)
    sys.exit(0)

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(HERE, "runs", "s38"); tag = sys.argv[2]; B = os.path.join(RUN, "fitfarm", tag)
planf = os.path.join(RUN, "lead_check", tag + "_plan.json")
if cmd == "prep":
    os.environ["DBT_SITE"] = "nersc"
    sys.path.insert(0, os.path.join(HERE, "harness"))
    import lead_check as LC
    cached = {d["id"] for d in json.load(open(planf))} if os.path.exists(planf) else set()
    led = [x for x in json.load(open(os.path.join(RUN, "leads_sent.json"))) if x["gen"] == tag and x["id"] not in cached]
    cfg = json.load(open(os.path.join(RUN, "config.json")))
    letter = cfg.get("names") or {s: chr(65 + i) for i, s in enumerate(cfg["sids"])}
    for d in ("exp", "code", "plan"):
        os.makedirs(os.path.join(B, d), exist_ok=True)
    q = {}
    for x in led:
        q.setdefault(x["sid"], {"rows": []})["rows"] += x["det"]
    R1 = {}
    for sid in q:                                   # one region per call keeps each reply small
        q[sid]["rows"] = sorted(set(q[sid]["rows"]))
        R1.update(LC.remote({sid: q[sid]}))
        LC.exposures(sid, os.path.join(RUN, "lead_check"))
        shutil.copy(os.path.join(RUN, "lead_check", f"{sid}_exposures.json"), os.path.join(B, "exp"))
        print(sid, len(q[sid]["rows"]), "rows", flush=True)
    for name, obj in (("leads.json", led), ("R1.json", R1), ("letter.json", letter)):
        json.dump(obj, open(os.path.join(B, name + ".new"), "w")); os.replace(os.path.join(B, name + ".new"), os.path.join(B, name))
    lk = os.path.expanduser("$ALE_HOME/planet-search/linker")
    for f in [os.path.join(HERE, "harness", "lead_check.py")] + [os.path.join(lk, n) for n in ("orbit_fit.py", "orbits.py", "link_cpu.py", "ed138.py")]:
        shutil.copy(f, os.path.join(B, "code"))
    shutil.copy(os.path.abspath(__file__), B)
    print(tag, len(led), "leads to fit in", len(q), "regions ->", B)
elif cmd == "collect":
    plan = {d["id"]: d for d in json.load(open(planf))} if os.path.exists(planf) else {}
    new = lines(B); plan.update(new)
    json.dump(list(plan.values()), open(planf + ".tmp", "w")); os.replace(planf + ".tmp", planf)
    n = len(json.load(open(os.path.join(B, "leads.json"))))
    print(tag, len(new), "of", n, "fitted in the bundle;", sum(1 for d in new.values() if d.get("fail")), "failed;", len(plan), "in", planf)
