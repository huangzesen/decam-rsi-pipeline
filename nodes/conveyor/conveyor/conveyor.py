#!/usr/bin/env python3
"""conveyor: continuous discovery, 5 % of the DECam sky at a time, on whichever GPUs start first.

2026-11-18.

The Mac coordinator. Every tick (default 3 min) it
  1. program   copies the hill climb's current best program (rsi/decam-big-tno/runs/r2/state.json) to every site as
               ROOT/best/<gen>/sub and marks it ROOT/best/CURRENT; each worker lane copies CURRENT when it starts a
               region, so a region always runs the newest best program;
  2. tasks     one search task per region of every unit in stage "search" whose scenario is built on NERSC
               (dbt/scen/<sid>); priority = unit order, then the unit's own order (LDT hours first); budget 1,800 s, or
               7,200 s for regions over 1,000 exposures (class big: only lanes with >= ~100 GB host memory);
  3. sites     looks at NERSC and Vista (site_scan.py), mirrors every claim made at one site to the other (so no region
               runs twice, apart from a race inside one tick), releases claims of workers that died, retries a
               region once if it ended without candidates.json, writes each site's queue of open tasks whose scenario
               is there;
  4. staging   copies scenarios NERSC -> Vista with Globus while Vista has a worker queued or running;
  5. workers   keeps worker jobs queued: NERSC A100 shared (1 GPU for normal regions, 2 GPUs with one lane for big
               ones), Vista gh and gb (all regions); cancels pending ones when no work is open or coming;
  6. results   copies out/timing/log of every site to rsi/decam-big-tno/runs/conveyor/<unit>/.
Units and their stages live in conveyor/state.json ("units"). Stage changes of the data side (select, download,
detect, build) are done by the unit tools; a unit enters "search" when its scenarios are being built.
Stop: touch conveyor/STOP (the loop ends; jobs keep running). Stop workers too: python3 -I conveyor.py stop-workers.
Usage: python3 -I conveyor.py tick | loop [MIN] | status | stop-workers | add-unit ID REGIONS_JSON [PRIO]"""
import json, math, os, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H = os.path.dirname(os.path.abspath(__file__)); PS = os.path.dirname(H); ALE = os.path.dirname(PS)
DBT = os.path.join(ALE, "rsi", "decam-big-tno")
STATE = os.path.join(H, "state.json"); LOG = os.path.join(H, "conveyor.log")
LOCAL_OUT = os.path.join(DBT, "runs", "conveyor")
NERSC_EP = "$GLOBUS_NERSC_COLLECTION"; VISTA_EP = "$GLOBUS_VISTA_COLLECTION"
SITES = {
    "nersc": dict(host="$NERSC_USER@perlmutter.nersc.gov", sock="$SSH_CM_DIR/nersc-$NERSC_USER.sock", user="$NERSC_USER",
                  root="$NERSC_SCRATCH/conveyor", scen="$NERSC_SCRATCH/dbt/scen",
                  py="$NERSC_SCRATCH/dbt/env/bin/python"),
    "vista": dict(host="$TACC_USER@vista.tacc.utexas.edu", sock="$SSH_CM_DIR/vista-$TACC_USER.sock", user="$TACC_USER",
                  root="$TACC_SCRATCH/conveyor", scen="$TACC_SCRATCH/dbt/scen",
                  py="python3"),
}
BIG_EXP = 1000; B_NORMAL = 1800; B_BIG = 7200
# worker kinds: sbatch arguments, lanes, big allowed, wall, how many to keep queued at most
KINDS = {
    "nersc-s1": dict(site="nersc", args="-q shared --gpus-per-node=1 -c 32 -t 08:00:00 --export=ALL,NGPU=1,LANES=1,BIGOK=0,WALL_S=28800", lanes=1, big=False, wall=28800, max=8, script="nersc_worker.sbatch"),   # 6 -> 8 (2026-11-20)
    "nersc-s2": dict(site="nersc", args="-q shared --gpus-per-node=2 -c 64 -t 08:00:00 --export=ALL,NGPU=2,LANES=1,BIGOK=1,WALL_S=28800", lanes=1, big=True, wall=28800, max=6, script="nersc_worker.sbatch"),   # 5 -> 6 (2026-11-20)
    # 2026-11-21 12:2x: no Vista GPU
    # workers; max was gh 2, gh-dev 2, gb 1. Vista keeps one gg download node at 10 streams.
    # 12:4x restored to gh 2, gh-dev 2, gb 1
    "vista-gh": dict(site="vista", args="-p gh -t 12:00:00 --export=ALL,WALL_S=43200", lanes=1, big=True, wall=43200, max=2, script="vista_worker.sbatch"),
    "vista-ghdev": dict(site="vista", args="-p gh-dev -t 02:00:00 --export=ALL,WALL_S=7200", lanes=1, big=False, wall=7200, max=2, script="vista_worker.sbatch"),  # 12:4x: back on,  (TACC: 1 running / 3 queued per user)
    "vista-gb": dict(site="vista", args="-p gb -t 12:00:00 --export=ALL,WALL_S=43200", lanes=4, big=True, wall=43200, max=1, script="vista_worker.sbatch"),
}


def log(*x):
    line = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, x)); print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def ssh(site, cmd, inp=None, timeout=600):
    s = SITES[site]
    p = subprocess.run(["ssh", "-S", os.path.expanduser(s["sock"]), "-o", "ControlMaster=no", "-o", "BatchMode=yes", s["host"], cmd],
                       input=inp, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def rsync(site, src, dst, to_site=True, extra=()):
    s = SITES[site]; e = f"ssh -S {os.path.expanduser(s['sock'])} -o ControlMaster=no -o BatchMode=yes"
    a, b = (src, f"{s['host']}:{dst}") if to_site else (f"{s['host']}:{src}", dst)
    return subprocess.run(["rsync", "-a", *extra, "-e", e, a, b], capture_output=True, text=True, timeout=3600).returncode


def gl(*a):
    p = subprocess.run(["globus", *a], capture_output=True, text=True, env=dict(os.environ, LC_ALL="C", LANG="C"), timeout=600)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return dict(units=[], tasks={}, vista_staged=[], staging=None, attempts={}, failed={}, deployed=False)


def save(st):
    json.dump(st, open(STATE + ".new", "w"), indent=1); os.replace(STATE + ".new", STATE)


def deploy(st):
    """copy the worker scripts and site_scan.py to both sites (always; they are small)."""
    for site in SITES:
        r = SITES[site]["root"]
        ssh(site, f"mkdir -p {r}/queue {r}/claims {r}/out {r}/log {r}/timing {r}/best {r}/slurm {r}/workers {SITES[site]['scen']}")
        for f in ("worker.sh", "site_scan.py", "nersc_worker.sbatch" if site == "nersc" else "vista_worker.sbatch"):
            rsync(site, os.path.join(H, f), f"{r}/{f}")
    st["deployed"] = True


def sync_program(st, scans):
    best = json.load(open(os.path.join(DBT, "runs", "r2", "state.json")))["best"]
    gen, sub = best["gen"], best["submission"]
    st["program"] = dict(gen=gen, reward=best.get("reward"))
    for site, sc in scans.items():
        if sc.get("current") == gen:
            continue
        r = SITES[site]["root"]
        ssh(site, f"mkdir -p {r}/best/{gen}")
        if rsync(site, sub.rstrip("/") + "/", f"{r}/best/{gen}/sub/") != 0:
            log(site, "program copy failed for", gen); continue
        ssh(site, f"echo {gen} > {r}/best/CURRENT.tmp && mv {r}/best/CURRENT.tmp {r}/best/CURRENT")
        log(site, "program now", gen, best.get("reward"))


def unit_regions(u):
    R = json.load(open(os.path.join(ALE, u["regions"])))
    order = sorted(range(len(R)), key=lambda i: -(R[i].get("ldt_h") or 0)) if u.get("order") == "ldt" else list(range(len(R)))
    return [dict(R[i], rank=k) for k, i in enumerate(order)]


def scan_all(st, sids):
    scans = {}
    for site in SITES:
        r = SITES[site]
        rc, out, err = ssh(site, f"{r['py']} -I {r['root']}/site_scan.py {r['root']} {r['scen']} {r['user']}" if site == "nersc"
                           else f"python3 {r['root']}/site_scan.py {r['root']} {r['scen']} {r['user']}", inp="\n".join(sids) + "\n")
        if rc != 0:
            log(site, "scan failed:", err.strip()[-200:]); continue
        scans[site] = json.loads(out)
    return scans


def tick():
    st = load()
    if not st.get("deployed"):
        deploy(st)
    # a unit still in prep searches the regions it has built early (units_side.early_build, 2026-11-20)
    units = [u for u in st["units"] if u.get("stage") == "search" or (u.get("stage") == "prep" and u.get("early"))]
    regs = {u["id"]: unit_regions(u) for u in units}
    sids = sorted({r["sid"] for rs in regs.values() for r in rs})
    scans = scan_all(st, sids)
    if "nersc" not in scans:
        log("NERSC not reachable; tick skipped")
        try:
            import watchdog
            watchdog.check(st, scans)
        except Exception as e:
            log("watchdog error:", type(e).__name__, str(e)[:200])
        save(st); return st
    sync_program(st, scans)
    built = set(scans["nersc"]["scen"])
    # ---- tasks
    tasks = {}
    for u in units:
        for r in regs[u["id"]]:
            if r["sid"] not in built:
                continue
            big = r["n_exp"] > BIG_EXP
            k = f"{u['id']}_{r['sid']}"
            tasks[k] = dict(unit=u["id"], sid=r["sid"], rid=r["rid"], n_exp=r["n_exp"], cls="big" if big else "normal",
                            budget=B_BIG if big else B_NORMAL, prio=int(u.get("prio", 1)) * 10000 + r["rank"])
    # ---- per task state across sites
    done, running, open_ = {}, {}, []
    for k, t in tasks.items():
        fin = [(s, sc["timing"][k]) for s, sc in scans.items() if k in sc["timing"]]
        # done: the run ended (timing written) with exit 0, or with any exit but a non-empty candidate list
        ok = [s for s, tm in fin if k in scans[s]["cands"] and (tm.split()[1:2] == ["0"] or scans[s]["cands"][k] > 2)]
        if ok:
            done[k] = dict(site=ok[0], timing=dict(fin)[ok[0]])
            if not st.setdefault("verified", {}).get(ok[0]):
                st["verified"][ok[0]] = time.strftime("%Y-%m-%d %H:%M"); log("SITE", ok[0], "verified: first region finished", k)
            continue
        if fin and k not in st["failed"]:
            # ended without candidates.json: retry once, then give up. A site that has never finished a region
            # (st["verified"]) is held back instead: its failure does not count against the region.
            unver = [s for s, _ in fin if not st.setdefault("verified", {}).get(s)]
            n = st["attempts"].get(k, 0) + (0 if unver else 1); st["attempts"][k] = n
            for s in unver:
                st.setdefault("site_bad", {})[s] = dict(at=time.strftime("%Y-%m-%d %H:%M"), task=k, timing=dict(fin)[s])
                log("SITE", s, "failed its first region", k, "(", dict(fin)[s], "): no more regions there until checked by hand")
            for s, tm in fin:
                r = SITES[s]["root"]; u, sid = t["unit"], t["sid"]
                ssh(s, f"mkdir -p {r}/failed/{u} && mv {r}/timing/{u}/{sid} {r}/failed/{u}/{sid}.{n} && cp {r}/log/{u}/{sid}.log {r}/failed/{u}/{sid}.{n}.log; rm -rf {r}/claims/{k}")
                log("task", k, "ended without candidates at", s, "(", tm, ") attempt", n)
            for s in scans:
                ssh(s, f"rm -rf {SITES[s]['root']}/claims/{k}")
            if n >= 2:
                st["failed"][k] = fin[0][1]; log("task", k, "failed twice; left out")
            continue
        if k in st["failed"]:
            continue
        local = [(s, sc["claims"][k]) for s, sc in scans.items() if sc["claims"].get(k, "").startswith("LOCAL")]
        if local:
            s, c = local[0]
            job = c.split()[2] if len(c.split()) > 2 else "?"
            alive = {w[0] for w in scans[s]["workers"] if w[1] in ("RUNNING", "COMPLETING")}
            if job != "?" and job not in alive:          # "?": claim dir made, LOCAL file not yet written
                log("task", k, "claimed by", s, "job", job, "which is gone; released")
                for s2 in scans:
                    ssh(s2, f"rm -rf {SITES[s2]['root']}/claims/{k}")
                open_.append(k); continue
            running[k] = s
            for s2, sc2 in scans.items():           # mirror the claim to the other sites
                if s2 != s and k not in sc2["claims"]:
                    ssh(s2, f"mkdir -p {SITES[s2]['root']}/claims/{k} && echo {s} > {SITES[s2]['root']}/claims/{k}/REMOTE")
            continue
        open_.append(k)
    # remote claims that no longer have a live local claim anywhere: drop them
    for s, sc in scans.items():
        for k, c in sc["claims"].items():
            if c.startswith("REMOTE") and k not in running and k not in done:
                ssh(s, f"rm -rf {SITES[s]['root']}/claims/{k}")
    # ---- vista staging
    vw = scans.get("vista", {}).get("workers", [])
    staged = set(st["vista_staged"]) | set(scans.get("vista", {}).get("scen", []))
    st["vista_staged"] = sorted(staged)
    if "vista" in scans and vw:
        stage_vista(st, [tasks[k]["sid"] for k in sorted(open_, key=lambda k: tasks[k]["prio"]) if tasks[k]["sid"] not in staged])
    # ---- queues
    for s, sc in scans.items():
        here = set(sc["scen"])
        want = {f"{tasks[k]['prio']:06d}_{k}": f"{tasks[k]['unit']} {tasks[k]['sid']} {tasks[k]['budget']} {tasks[k]['cls']}"
                for k in sorted(open_, key=lambda k: tasks[k]["prio"]) if tasks[k]["sid"] in here}
        if st.get("site_bad", {}).get(s):
            want = {}                              # held back after a failure; clear st["site_bad"][site] to release
        elif not st.get("verified", {}).get(s):
            normal = [f for f, v in want.items() if v.endswith(" normal")]
            want = {f: want[f] for f in (normal or list(want))[:1]}   # one region until the site has finished one
        have = set(sc["queue"])
        rm = [f for f in have if f not in want]
        add = {f: v for f, v in want.items() if f not in have}
        if rm or add:
            r = SITES[s]["root"]
            cmd = "".join(f"rm -f {r}/queue/{f}; " for f in rm) + "".join(f"echo '{v}' > {r}/queue/{f}; " for f, v in add.items())
            ssh(s, cmd)
    # ---- workers (regions still to be built count too, so workers wait in the queue before the scenarios exist)
    unbuilt = [dict(cls="big" if r["n_exp"] > BIG_EXP else "normal", budget=B_BIG if r["n_exp"] > BIG_EXP else B_NORMAL)
               for u in units for r in regs[u["id"]] if r["sid"] not in built]
    manage_workers(st, scans, tasks, open_, running, unbuilt)
    # ---- results
    for u in units:
        dst = os.path.join(LOCAL_OUT, u["id"]); os.makedirs(dst, exist_ok=True)
        for s in scans:
            r = SITES[s]["root"]
            for sub in ("out", "timing", "log"):
                os.makedirs(os.path.join(dst, sub), exist_ok=True)
                rsync(s, f"{r}/{sub}/{u['id']}/", os.path.join(dst, sub) + "/", to_site=False, extra=("--ignore-missing-args",))
    # ---- data side: unit stages, detection, builds, caching, the next unit
    try:
        import importlib, units_side
        importlib.reload(units_side)        # edits take effect next tick, no loop restart (2026-11-21)
        units_side.tick(st, log)
    except Exception as e:
        log("units side error:", type(e).__name__, str(e)[:300])
    # ---- summary
    summ = {}
    for u in units:
        ks = [k for k in tasks if tasks[k]["unit"] == u["id"]]
        summ[u["id"]] = dict(regions=len(regs[u["id"]]), built=len(ks), done=sum(k in done for k in ks), running=sum(k in running for k in ks),
                             open=sum(k in open_ for k in ks), failed=sum(k in st["failed"] for k in ks),
                             gens=sorted({done[k]["timing"].split()[2] for k in ks if k in done and len(done[k]["timing"].split()) > 2}),
                             sites=sorted({done[k]["site"] for k in ks if k in done}))
        if summ[u["id"]]["regions"] and summ[u["id"]]["done"] + summ[u["id"]]["failed"] == summ[u["id"]]["regions"] and u.get("stage") == "search":
            u["stage"] = "searched"; u["searched_at"] = time.strftime("%Y-%m-%d %H:%M"); log("unit", u["id"], "searched:", summ[u["id"]])
    st["summary"] = summ
    try:
        import watchdog
        watchdog.check(st, scans)
    except Exception as e:
        log("watchdog error:", type(e).__name__, str(e)[:200])
    st["workers"] = {s: sc["workers"] for s, sc in scans.items()}
    st["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(st)
    log("tick:", json.dumps(summ), "| program", st.get("program", {}).get("gen"),
        "| workers", {s: [f"{w[2]}:{w[1][:3]}" for w in sc["workers"]] for s, sc in scans.items()})
    return st


def stage_vista(st, sids):
    g = st.get("staging")
    if g:
        rc, out, _ = gl("task", "show", g["task"], "--jmespath", "status", "--format", "unix")
        if out in ("ACTIVE", "INACTIVE") or rc != 0:
            return
        if out == "SUCCEEDED":
            st["vista_staged"] = sorted(set(st["vista_staged"]) | set(g["sids"])); log("vista: staged", len(g["sids"]), "scenarios")
        else:
            log("vista: staging task", g["task"], out)
        st["staging"] = None
    sids = sids[:60]
    if not sids:
        return
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        for s in sids:
            f.write(f"{s}/ {s}/ --recursive\n")
    rc, tid, err = gl("transfer", f"{NERSC_EP}:{SITES['nersc']['scen']}/", f"{VISTA_EP}:{SITES['vista']['scen']}/", "--batch", f.name,
                      "--sync-level", "size", "--verify-checksum", "--label", f"conveyor scen {len(sids)}", "--notify", "failed",
                      "--jmespath", "task_id", "--format", "unix")
    os.remove(f.name)
    if rc == 0 and tid:
        st["staging"] = dict(task=tid, sids=sids, at=time.strftime("%Y-%m-%d %H:%M")); log("vista: staging", len(sids), "scenarios, task", tid)
    else:
        log("vista: staging submit failed:", err[-200:])


def manage_workers(st, scans, tasks, open_, running, unbuilt=()):
    upcoming = any(u.get("stage") in ("selected", "prep") for u in st["units"])
    # NERSC's shared GPU queue starts within the hour: its workers follow the regions ready now. Vista waits days: its
    # workers are queued for the regions still being built and for units on the way.
    hours = dict(normal=sum(tasks[k]["budget"] for k in open_ if tasks[k]["cls"] == "normal") / 3600,
                 big=sum(tasks[k]["budget"] for k in open_ if tasks[k]["cls"] == "big") / 3600)
    hours_ahead = sum(t["budget"] for t in unbuilt) / 3600
    work = hours["normal"] + hours["big"] + hours_ahead > 0
    for kind, K in KINDS.items():
        s = K["site"]
        if s not in scans or st.get("site_bad", {}).get(s):
            continue
        if s == "vista":                          # gh / gb by partition
            jobs = [w for w in scans[s]["workers"] if w[2] == kind.split("-", 1)[1].replace("ghdev", "gh-dev")]
        else:                                     # NERSC kinds differ by GPU count: by the job ids this coordinator queued
            jobs = [w for w in scans[s]["workers"] if st.get("kind_of", {}).get(w[0]) == kind]
        pend = [w for w in jobs if w[1] == "PENDING"]
        if kind == "nersc-s1":
            want = min(K["max"], math.ceil(hours["normal"] / 4)) if hours["normal"] > 0 else 0
        elif kind == "nersc-s2":
            want = min(K["max"], math.ceil(hours["big"] / 4)) if hours["big"] > 0 else 0
        elif kind == "vista-ghdev":                # 2 h, normal regions only: worth queueing while such work is open or near
            want = K["max"] if (hours["normal"] > 0 or hours_ahead > 0) else 0
        else:
            want = K["max"] if (work or upcoming) else 0
        if len(jobs) < want:
            for _ in range(want - len(jobs)):
                r = SITES[s]["root"]
                rc, out, err = ssh(s, f"cd {r} && sbatch {K['args']} {K['script']}")
                jid = out.strip().split()[-1] if rc == 0 and out.strip() else None
                if jid:
                    st.setdefault("kind_of", {})[jid] = kind; log("worker", kind, "queued:", jid)
                else:
                    log("worker", kind, "submit failed:", (err or out).strip()[-200:]); break
        elif len(jobs) > want and pend:
            for w in pend[: len(jobs) - want]:
                ssh(s, f"scancel {w[0]}"); log("worker", kind, w[0], "cancelled while pending (not needed now)")


def stop_workers():
    for s in SITES:
        rc, out, _ = ssh(s, f"squeue -u {SITES[s]['user']} -h -n cvw -o %i")
        ids = out.split()
        if ids:
            ssh(s, "scancel " + " ".join(ids)); log(s, "cancelled workers", ids)


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "tick"
    if c == "tick":
        tick()
    elif c == "loop":
        m = float(sys.argv[2]) if len(sys.argv) > 2 else 3
        bad = 0                                    # whole-tick failures skip the watchdog, so the loop alerts itself (11-19 14:05-15:00 lesson)
        while not os.path.exists(os.path.join(H, "STOP")):
            try:
                tick()
                if bad >= 3:
                    open(os.path.join(H, "alerts.log"), "a").write(time.strftime("%m-%d %H:%M") + f" OK tick_crash: cleared after {bad} failed ticks\n")
                bad = 0
            except Exception as e:
                import traceback
                bad += 1
                log("tick error:", type(e).__name__, str(e)[:300], "|", traceback.format_exc().strip().splitlines()[-3][-200:])
                if bad == 3:
                    open(os.path.join(H, "alerts.log"), "a").write(time.strftime("%m-%d %H:%M") + f" ALERT tick_crash: 3 ticks in a row failed: {type(e).__name__} {str(e)[:120]}\n")
            time.sleep(m * 60)
        log("loop stopped (STOP)")
    elif c == "status":
        print(json.dumps({k: load().get(k) for k in ("at", "program", "summary", "workers", "staging")}, indent=1))
    elif c == "stop-workers":
        stop_workers()
    elif c == "add-unit":
        st = load(); uid, regions = sys.argv[2], sys.argv[3]
        st["units"] = [u for u in st["units"] if u["id"] != uid] + [dict(id=uid, regions=regions, stage="search", prio=int(sys.argv[4]) if len(sys.argv) > 4 else len(st["units"]) + 1, order="ldt")]
        save(st); print("unit", uid, "added")
    elif c == "deploy":
        st = load(); deploy(st); save(st); print("deployed")
