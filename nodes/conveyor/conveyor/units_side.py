#!/usr/bin/env python3
"""conveyor, data side (called by conveyor.py every tick). Units made by unit.py ("kind": "conveyor") move through:
  selected  -> start known-body positions (Mac) and stage at NERSC (mirror links, NOIRLab downloads, Globus pull from
               the Vista cache)                                                                  -> prep
  prep      -> detection jobs per share as images arrive (CPU debug while NERSC's other CPU queues wait; CPU regular
               when Slurm says it starts within 30 min); when every share is final and detected and the known bodies
               are done: build input to dbt/units/<uid>/in, build job                            -> search
  search    -> rebuild until BUILT; then truth labels to the Mac (rsi/decam-big-tno/private/truth_<uid>, deleted at
               NERSC) and the unit's own downloads cached on Vista and removed at NERSC (detection lists stay)
A new unit is selected when no unit is selected or in prep with downloads still running, fewer than MAX_AHEAD units
wait for or are in search, and conveyor/NO_NEW_UNITS does not exist."""
import glob, json, os, shutil, subprocess, sys, tempfile, time
import unit as U

H = os.path.dirname(os.path.abspath(__file__)); PS = os.path.dirname(H); ALE = os.path.dirname(PS)
NR = U.NR; DBT = U.DBT; MAX_AHEAD = 4          # was 3 (2026-11-20: units ready ahead of the GPUs)
DEBUG_MAX = 5                                   # NERSC debug QOS: 2 running, 5 submitted per user
REG_MAX = 8                                     # regular CPU detection jobs at once (2026-11-20)
PREP_MAX = 4                                    # units in selected/prep at once (2 until 2026-11-20 16:00, 3 until 2026-11-22 18:2x)


def sh(cmd, timeout=900):
    return U.sh(cmd, timeout)


def gl(*a):
    p = subprocess.run(["globus", *a], capture_output=True, text=True, env=dict(os.environ, LC_ALL="C", LANG="C"), timeout=600)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def tick(st, log):
    cu = [u for u in st["units"] if u.get("kind") == "conveyor"]
    rc, q, _ = sh("squeue -u $NERSC_USER -h -o '%i|%j|%T|%q|%r'")
    jobs = [l.split("|") for l in q.splitlines() if l.strip()]
    for u in cu:
        try:                      # one unit's error must not stop the others, releases, downloads or detection (11-19:
            unit_step(u, jobs, log)   # a15's build failed every tick 05:08-09:55 and starved everything after it)
        except Exception as e:
            log("unit", u["id"], "step error:", type(e).__name__, str(e)[:240])
    for u in cu:
        if u.get("stage") in ("prep", "search") and not u.get("cached"):
            try:
                release(u, log)
            except Exception as e:
                log("unit", u["id"], "release error:", type(e).__name__, str(e)[:200])
    try:
        ls6_feed(st, cu, log)
    except Exception as e:
        log("ls6_feed error:", type(e).__name__, str(e)[:240])
    for step in (keep_downloads, feed_detection):
        try:
            step(st, cu, jobs, log)
        except Exception as e:
            log(step.__name__, "error:", type(e).__name__, str(e)[:240])
    next_unit(st, cu, log)


def unit_step(u, jobs, log):
    if True:
        uid = u["id"]; R = f"{NR}/{uid}"
        if u["stage"] == "selected":
            if not u.get("known_started"):
                U.known(uid); u["known_started"] = time.strftime("%Y-%m-%d %H:%M")
            info = U.stage(uid)
            u.update(stage="prep", pull_task=info.get("pull_task"), fetch_jobs=info.get("fetch_jobs"), staged=info.get("staged"))
            try:      # region-first download order (user 11-20: "can we search region by region"); done by hand for u17-u20
                n = max(1, len(info.get("fetch_jobs") or []))
                subprocess.run(U.SCP + [os.path.join(U.U0, uid, "regions_exposures.json"), os.path.join(H, "nersc", "prio_missing.py"),
                                        f"$NERSC_USER@perlmutter.nersc.gov:{R}/"], check=True)
                rc, o, e = sh(f"cd {R} && $NERSC_SCRATCH/dbt/env/bin/python -I prio_missing.py . regions_exposures.json {n} | tail -1")
                lm = os.path.join(U.U0, uid, "missing.tsv")
                shutil.copy2(lm, lm + ".bak-" + time.strftime("%Y%m%d-%H%M"))
                subprocess.run(U.SCP + [f"$NERSC_USER@perlmutter.nersc.gov:{R}/missing.tsv", lm], check=True)
                log("unit", uid, "download list region-first:", o.strip()[-160:] or e.strip()[-160:])
            except Exception as e:
                log("unit", uid, "region-first order error:", type(e).__name__, str(e)[:200])
            try:
                ls6_gap(uid, len(info.get("fetch_jobs") or []), log)
            except Exception as e:
                log("unit", uid, "Lonestar6 gap list error:", type(e).__name__, str(e)[:200])
            log("unit", uid, "staged:", json.dumps({k: info.get(k) for k in ("exposures", "tb", "mirror", "vista_files", "noirlab_files", "noirlab_tb")}))
            return
        if u["stage"] == "prep":
            # a pull whose submission failed (u16, 11-19: Globus "operation timed out") is retried like a deferred one (2026-11-20)
            if (u.get("pull_task") == "deferred" or str(u.get("pull_task") or "").startswith("failed")) and U.under_cap():
                was = u.get("pull_task")
                u["pull_task"] = U.start_pull(uid); log("unit", uid, "Vista pull started (scratch under the cap):", u["pull_task"],
                                                        "" if was == "deferred" else "(retry after a failed submission)")
            t = u.get("pull_task")
            if t and not t.startswith("failed") and not u.get("pull_done"):
                _, s, _ = gl("task", "show", t, "--jmespath", "status", "--format", "unix")
                if s == "SUCCEEDED":
                    sh(f"touch {R}/v/log/PULL_DONE"); u["pull_done"] = True; log("unit", uid, "Vista pull done")
                elif s == "FAILED":
                    log("unit", uid, "Vista pull FAILED; the missing files go to NOIRLab next round"); u["pull_done"] = "failed"; sh(f"touch {R}/v/log/PULL_DONE")
            rc, o, _ = sh(f"python3 -I {NR}/remaining.py {R}")
            try:
                rem = json.loads(o)
            except Exception:
                return
            u["detect"] = rem
            full = rem and all(v[0] == 0 and v[2] for v in rem.values())
            if not full and time.time() - u.get("early_t", 0) > 600 and not any(j[1] == f"cvb-{uid}" for j in jobs) and U.known_done(uid):
                u["early_t"] = time.time(); early_build(u, log)
            if full and any(j[1] == f"cvb-{uid}" for j in jobs):
                return                                # an early build still runs: the full build waits for it
            if full and U.known_done(uid):
                B = f"{DBT}/units/{uid}"
                sh(f"mkdir -p {B}/in")
                regs = json.load(open(os.path.join(U.U0, uid, "regions_search.json")))
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
                    f.write("".join(f"{r['rid']} {r['sid']} {r['n_exp']}\n" for r in regs))
                subprocess.run(U.SCP + [f.name, f"$NERSC_USER@perlmutter.nersc.gov:{B}/list.txt"], check=True); os.remove(f.name)
                subprocess.run(["rsync", "-a", "-e", f"ssh -S {os.path.expanduser('$SSH_CM_DIR/nersc-$NERSC_USER.sock')} -o ControlMaster=no",
                                os.path.join(U.U0, uid, "search_in") + "/", f"$NERSC_USER@perlmutter.nersc.gov:{B}/in/"], check=True)
                sh(f"cp -n {DBT}/searchldt5/build_scen.py {DBT}/units/build_scen.py")
                rc, o, e = sh("squeue -u $NERSC_USER -h -n cvi-build -t R -o %i | head -1"); J = o.strip()
                if J:     # a held interactive build node (2026-11-21: u18's sbatch waited with no start time; on the node 7 min)
                    rc, o, e = sh(f"F={NR}/slurm/build-{J}-{uid}-full-$(date +%H%M%S).out; cd {B} && srun --jobid={J} -N 1 "
                                  f"--export=ALL,U={R},B={B} bash {NR}/unit_build.sbatch > $F 2>&1; grep '^done' $F", timeout=1500)
                    o = f"on held node {J}: " + (o.strip() or e.strip()[-120:])
                else:
                    rc, o, e = sh(f"cd {B} && sbatch {build_q()} -J cvb-{uid} --export=ALL,U={R},B={B} {NR}/unit_build.sbatch")
                # its fetchers have nothing left that the search can use (u06/u16's ran on 17-35 h retrying unobtainable files,
                # holding xfer slots and NOIRLab connections; cancelled by hand 11-20 11:55)
                sh(f"squeue -u $NERSC_USER -h -o '%i %j' | awk '$2 ~ /^cvf-{uid}-/ {{print $1}}' | xargs -r scancel")
                u.update(stage="search", build="running", regions=f"planet-search/units/{uid}/regions_search.json", order="ldt", search_from=time.strftime("%Y-%m-%d %H:%M"))
                log("unit", uid, "detected; build job", o.strip() or e.strip()[-120:])
            return
        if u["stage"] == "search" and u.get("build") == "running":
            B = f"{DBT}/units/{uid}"
            rc, o, _ = sh(f"test -f {B}/BUILT && echo yes")
            if o.strip() == "yes":
                dst = os.path.join(ALE, "rsi", "decam-big-tno", "private", f"truth_{uid}"); os.makedirs(dst, exist_ok=True)
                subprocess.run(["rsync", "-a", "-e", f"ssh -S {os.path.expanduser('$SSH_CM_DIR/nersc-$NERSC_USER.sock')} -o ControlMaster=no",
                                f"$NERSC_USER@perlmutter.nersc.gov:{B}/truth/", dst + "/"], check=True)
                rem = dict(l.split()[::-1] for l in sh(f"cd {B}/truth && stat -c '%s %n' *.npz")[1].splitlines() if l.strip())
                ok = [f for f, n in rem.items() if os.path.exists(os.path.join(dst, f)) and os.path.getsize(os.path.join(dst, f)) == int(n)]
                if ok and len(ok) == len(rem):
                    sh(f"cd {B}/truth && rm -f " + " ".join(ok))
                u["build"] = "done"; log("unit", uid, "scenarios built; truth labels to the Mac:", len(ok), "of", len(rem))
                cache(u, log)
            elif not any(j[1] == f"cvb-{uid}" for j in jobs):
                rc, o, e = sh(f"cd {B} && sbatch {build_q()} -J cvb-{uid} --export=ALL,U={NR}/{uid},B={B} {NR}/unit_build.sbatch")
                log("unit", uid, "build continues:", o.strip() or e.strip()[-120:])
        if u.get("cache_task") and not u.get("cached"):
            _, s, _ = gl("task", "show", u["cache_task"], "--jmespath", "status", "--format", "unix")
            if s == "SUCCEEDED":
                names = sh(f"cd {NR}/{uid} && ls p[0-9]*/img/*/ 2>/dev/null | grep '^c4d_'")[1].split()
                if names:                         # on Vista now: later units pull them instead of downloading again
                    open(os.path.join(PS, "fill", "relay", "relayed.txt"), "a").write("".join(n + "\n" for n in names))
                sh(f"rm -rf {NR}/{uid}/p[0-9]*/img {NR}/{uid}/v/img"); u["cached"] = True
                log("unit", uid, "downloads cached on Vista (", len(names), "files ); NERSC copies removed")


FILL = "$NERSC_SCRATCH/decam-fill"
FILL_PARTS = {"0": "0/6", "1": "1/6", "2": "2/6", "3": "3/6", "41": "4/12", "42": "10/12", "51": "5/12", "52": "11/12"}


LS6_SSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/ls6-$TACC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "x"]
LS6_D = "$TACC_SCRATCH/units-ls6"
LS6_DEST = os.path.join(PS, "fill", "relay_units_ls6", "dest.tsv")


def ls6(cmd, inp=None, timeout=300):
    p = subprocess.run(LS6_SSH + ["LC_ALL=C; " + cmd], input=inp, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def ls6_gap(uid, n, log):
    """2026-11-20: a unit's NOIRLab gap (missing.tsv) is
    also fetched on Lonestar6, list reversed (the NERSC fetchers work from the front), into units-ls6/queue/<uid>.tsv;
    fill/relay_units_ls6.py puts each file into the NERSC share whose fetcher would take it (same blocks of 32 exposures
    as fetch.py --part k/n; checked against u17's hand-made map, 10,513 of 10,513), where that fetcher finds it on disk."""
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(U.U0, uid, "missing.tsv")) if l.strip()]
    if not rows or not n:
        return
    dest, nexp, last = [], -1, None
    for md5, size, name, tag in rows:
        if name[:17] != last:
            last = name[:17]; nexp += 1
        dest.append(f"{name}\t{NR}/{uid}/p{(nexp // 32) % n}\n")
    rc, o, e = ls6(f"mkdir -p {LS6_D}/queue && cat > {LS6_D}/queue/{uid}.tsv", inp="".join("\t".join(r) + "\n" for r in reversed(rows)))
    if rc != 0:
        log("unit", uid, "Lonestar6 gap list not written:", e.strip()[-150:]); return
    open(LS6_DEST, "a").write("".join(dest))
    log("unit", uid, "NOIRLab gap also queued on Lonestar6:", len(rows), "files")


LS6_UNITS_MAX = 2                                 # 2026-11-20 22:5x: one Lonestar6 job, 8 streams, on u19's gap from the end; was 3 x 64 streams
# 2026-11-22 18:3x: 2 jobs, 15 streams each, sharing the device budget of 15 (fetch_ctl.json)


def ls6_feed(st, cu, log):
    """Every 10 min: each unit in prep with a gap list (queue/<uid>.tsv) and no units-<uid> job gets its own Lonestar6 CPU
    node job (normal, 7 h, 64 streams, <= 7 SU) in its own folder <base>/<uid>, on the files no Lonestar6 job has fetched
    yet; at most LS6_UNITS_MAX such jobs queued or running (2026-11-20 13:30: one shared job made u18 wait behind u17's,
    which ran at 16 MB/s under NOIRLab's refusals)."""
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "LS6_DL_STOP")):
        return                                     # 2026-11-20 19:25, 
    if time.time() - st.get("ls6_feed_t", 0) < 600:
        return
    st["ls6_feed_t"] = time.time()
    prep = sorted(u["id"] for u in cu if u.get("stage") == "prep")
    rc, o, e = ls6(f"squeue -u $USER -h -o '%j'; echo __Q__; ls {LS6_D}/queue/")
    if "__Q__" not in o:
        log("Lonestar6 not reachable for the units gap feed:", e.strip()[-120:]); return
    jobs_txt, q_txt = o.split("__Q__", 1)
    names = [x for x in jobs_txt.split() if x.startswith("units-")]
    have = {x[:-4] for x in q_txt.split() if x.endswith(".tsv")}
    nrun = len(names)
    for uid in prep:
        if nrun >= LS6_UNITS_MAX:
            break
        if uid not in have or f"units-{uid}" in names:
            continue
        rc, o, e = ls6(f"cd {LS6_D} && mkdir -p {uid}/log {uid}/img && ( cat log/done_*.txt {uid}/log/done_*.txt 2>/dev/null; echo __END__; cat queue/{uid}.tsv ) "
                       f"| awk -F'\\t' '/^__END__$/{{f=1;next}} !f{{d[$1]=1;next}} !($3 in d)' > {uid}/list.tsv && wc -l < {uid}/list.tsv")
        try:
            n = int(o.strip().splitlines()[-1])
        except Exception:
            log("Lonestar6 list for", uid, "failed:", (e or o).strip()[-150:]); continue
        if not n:
            continue
        rc, o, e = ls6(f"cd {LS6_D} && sbatch -A $LS6_ACCOUNT -p normal -t 07:00:00 -J units-{uid} "
                       f"--export=ALL,STREAMS=15,HOURS=6.7,UD={LS6_D}/{uid},LIST={LS6_D}/{uid}/list.tsv node.sbatch")
        msg = (o.strip() or e.strip()[-150:]).splitlines()
        log("Lonestar6 gap job for", uid, f"({n} files):", msg[-1] if msg else "?")
        nrun += 1


def build_q():
    """Scenario builds on a regular CPU node (1 h) when one would start within 15 min, else the debug QOS (2026-11-20: u16's
    build waited 10:56-11:45 for a debug slot)."""
    # always regular: Slurm's --test-only start estimate said days (11-22) while regular jobs started within a minute
    # (11:41-11:47), and the debug QOS (2 running, 5 queued per user) is full of small detection jobs (11:52 early build refused)
    return "-q regular -t 01:00:00"


def early_build(u, log):
    """2026-11-20: build and search each region as soon as every one of its exposures
    is detected, while the rest of the unit still downloads and detects. Nothing is left out: a region waits for all its
    exposures, and the unit's full build later skips the scenarios that exist. Search tasks of a prep unit are made only
    for built scenarios (conveyor.py takes units with "early" set); the unit is marked searched only from stage search."""
    uid = u["id"]; R = f"{NR}/{uid}"; B = f"{DBT}/units/{uid}"
    regs = json.load(open(os.path.join(U.U0, uid, "regions_search.json")))
    sh(f"mkdir -p {B}/in")
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write("".join(f"{r['rid']} {r['sid']} {r['n_exp']}\n" for r in regs))
    subprocess.run(U.SCP + [f.name, f"$NERSC_USER@perlmutter.nersc.gov:{B}/list_all.txt"], check=True); os.remove(f.name)
    subprocess.run(U.SCP + [os.path.join(H, "nersc", "ready.py"), os.path.join(H, "nersc", "unit_build.sbatch"), f"$NERSC_USER@perlmutter.nersc.gov:{NR}/"], check=True)
    subprocess.run(["rsync", "-a", "-e", f"ssh -S {os.path.expanduser('$SSH_CM_DIR/nersc-$NERSC_USER.sock')} -o ControlMaster=no",
                    os.path.join(U.U0, uid, "search_in") + "/", f"$NERSC_USER@perlmutter.nersc.gov:{B}/in/"], check=True)
    rc, o, e = sh(f"{DBT}/env/bin/python -I {NR}/ready.py {R} {B}/in/regions_exposures.json {B}/list_all.txt {DBT}/scen > {B}/list_early.txt && wc -l < {B}/list_early.txt", timeout=1800)
    try:
        n = int(o.strip().splitlines()[-1])
    except Exception:
        log("unit", uid, "early build: ready check failed:", (e or o).strip()[-150:]); return
    if not n:
        return
    sh(f"cp -n {DBT}/searchldt5/build_scen.py {DBT}/units/build_scen.py")
    # 2026-11-20 23:30: when an interactive CPU node named cvi-build is
    # held, build there at once (about 30 s per region; the regular queue gave no start estimate that night)
    rc, o, e = sh("squeue -u $NERSC_USER -h -n cvi-build -t R -o %i | head -1"); J = o.strip()
    if J:
        rc, o, e = sh(f"F={NR}/slurm/build-{J}-{uid}-$(date +%H%M%S).out; cd {B} && srun --jobid={J} -N 1 "
                      f"--export=ALL,U={R},B={B},L={B}/list_early.txt bash {NR}/unit_build.sbatch > $F 2>&1; grep '^done' $F", timeout=1500)
        u.update(early=True, regions=f"planet-search/units/{uid}/regions_search.json", order="ldt")
        log("unit", uid, "early build of", n, "regions on held node", J + ":", (o.strip() or e.strip())[-120:])
        return
    rc, o, e = sh(f"cd {B} && sbatch {build_q()} -J cvb-{uid} --export=ALL,U={R},B={B},L={B}/list_early.txt {NR}/unit_build.sbatch")
    if rc == 0:
        u.update(early=True, regions=f"planet-search/units/{uid}/regions_search.json", order="ldt")
    log("unit", uid, "early build of", n, "regions already detected:", o.strip() or e.strip()[-120:])


def keep_downloads(st, cu, jobs, log):
    """resubmit download jobs that died (NERSC login-node failures killed all eight fill jobs on 11-18 at 13:06), and give
    the units' downloads the free xfer slots first: for each unit download job waiting on the per-user limit, one running
    fill job is cancelled (it is resubmitted below and waits behind the unit's job; fetch.py resumes where it was)."""
    if U.nersc_dl_paused():                        # 12 h cool-down from 2026-11-20 17:40 (user); resumes by itself
        return
    names = {j[1] for j in jobs}
    waiting_units = [j for j in jobs if j[1].startswith("cvf-") and j[2] == "PENDING"]
    running_fill = [j for j in jobs if j[1].startswith("fillx") and j[2] == "RUNNING"]
    for j, f in zip(waiting_units, running_fill):
        sh(f"scancel {f[0]}"); names.discard(f[1])
        log("download slot for", j[1], ": fill job", f[1], f[0], "cancelled (resubmitted to wait behind it)")
    rc, o, _ = sh(f"cd {FILL} && ls -d p*/log/SHARE_DONE STOP_ALL 2>/dev/null")
    fin = set(o.split())
    if "STOP_ALL" not in fin:
        for k, part in FILL_PARTS.items():
            if f"fillx{k}" not in names and f"p{k}/log/SHARE_DONE" not in fin:
                rc, o, e = sh(f"cd {FILL} && sbatch -q xfer -C cron --mem=8G -t 48:00:00 -J fillx{k} --export=ALL,PART={part},PARTDIR=p{k} nersc_fill.sbatch")
                log("fill part", k, "had no job; resubmitted:", (o.strip() or e.strip()[-120:]))
    rc, q, _ = sh("squeue -u $NERSC_USER -h -o '%j'")      # fresh: a unit staged in this tick has jobs the tick's first look missed
    names |= set(q.split())                            # (11-19 00:41: u06 part 0 was submitted twice)
    for u in cu:
        if u.get("stage") != "prep":
            continue
        n = len(u.get("fetch_jobs") or [])
        R = f"{NR}/{u['id']}"
        rc, o, _ = sh(f"cd {R} && ls -d p*/log/SHARE_DONE 2>/dev/null")
        fin = set(o.split())
        for k in range(n):
            if f"cvf-{u['id']}-{k}" not in names and f"p{k}/log/SHARE_DONE" not in fin:
                rc, o, e = sh(f"cd {R} && sbatch -q xfer -C cron --mem=8G -t 48:00:00 -J cvf-{u['id']}-{k} --export=ALL,U={R},PART={k}/{n},PARTDIR=p{k} {NR}/unit_fetch.sbatch")
                log("unit", u["id"], "download part", k, "had no job; resubmitted:", (o.strip() or e.strip()[-120:]))


def release(u, log):
    """free a unit's images as soon as they are detected (2026-11-19): pulled-back (v) and fill-linked (m) files are deleted at once (they are on Vista); the unit's own
    NOIRLab downloads (p) go to the Vista cache in batches and are deleted when the batch's Globus copy has succeeded."""
    uid = u["id"]; R = f"{NR}/{uid}"
    for mode in ("v", "links"):
        o = sh(f"python3 -I {NR}/unit_release.py {mode} {R}")[1].strip()
        if o and not o.startswith(("deleted 0 ", "freed 0.0")):
            log("unit", uid, "release", mode + ":", o)
    # Vista-cache batches of at most BATCH_GB, up to BATCH_N in flight, so space frees every ~20 minutes instead of after
    # one multi-terabyte copy (Globus lists a task's copied files only when the whole task has ended) (2026-11-19, user:
    # "make headrooms")
    left = []
    for b in u.get("cache_batches") or ([u["cache_batch"]] if u.get("cache_batch") else []):
        _, s, _ = gl("task", "show", b["task"], "--jmespath", "[status, files, files_transferred, files_skipped]", "--format", "unix")
        f = s.split()
        if not f or f[0] in ("ACTIVE", "INACTIVE"):
            left.append(b); continue
        if f[0] == "SUCCEEDED" and int(f[2]) + int(f[3]) == int(f[1]):
            names = [l.strip().rsplit("/", 1)[-1] for l in sh(f"cat {R}/cache/{b['id']}.txt")[1].splitlines() if l.strip()]
            o = sh(f"python3 -I {NR}/unit_release.py delete {R} {b['id']}")[1].strip()
            open(os.path.join(PS, "fill", "relay", "relayed.txt"), "a").write("".join(n + "\n" for n in names))
            log("unit", uid, "cache batch", b["id"], "on Vista;", o)
        else:
            log("unit", uid, "cache batch", b["id"], "NOT copied:", s, "; files stay, a later batch retries them")
            sh(f"mv {R}/cache/{b['id']}.txt {R}/cache/{b['id']}.failed")
    u["cache_batch"] = None
    final = all(v[2] for v in (u.get("detect") or {}).values())
    while len(left) < BATCH_N:
        bid = f"b{len(u.get('cache_batches_done', [])) + 1:03d}"
        o = sh(f"python3 -I {NR}/unit_release.py list {R} {bid} {BATCH_GB}")[1].strip()
        try:
            nf, gb = int(o.split()[0]), float(o.split()[2])
        except Exception:
            break
        if nf == 0 or (gb < 100 and not final):
            sh(f"rm -f {R}/cache/{bid}.txt"); break
        lst = sh(f"cat {R}/cache/{bid}.txt")[1].split()
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as fh:
            for r in lst:
                fh.write(f"{r} {r.split('/img/', 1)[1]}\n")
        rc, tid, err = gl("transfer", f"{U.NERSC_EP}:{R}/", f"{U.VISTA_EP}:{U.VISTA_CACHE}/", "--batch", fh.name, "--sync-level", "size",
                          "--verify-checksum", "--label", f"conveyor {uid} {bid} to the Vista cache", "--notify", "failed",
                          "--jmespath", "task_id", "--format", "unix")
        os.remove(fh.name)
        if rc == 0 and tid:
            left.append(dict(id=bid, task=tid, files=nf, gb=gb)); u.setdefault("cache_batches_done", []).append(bid)
            log("unit", uid, "cache batch", bid, f"{nf} files {gb} GB to Vista, task", tid)
        else:
            sh(f"rm -f {R}/cache/{bid}.txt"); log("unit", uid, "cache batch submit failed:", err[-200:]); break
    u["cache_batches"] = left


BATCH_GB, BATCH_N = 400, 3


def cache(u, log):
    uid = u["id"]; R = f"{NR}/{uid}"
    rc, o, _ = sh(f"cd {R} && ls -d p[0-9]*/img 2>/dev/null")
    dirs = o.split()
    if not dirs:
        sh(f"rm -rf {R}/v/img"); u["cached"] = True; return
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        for d in dirs:
            f.write(f"{d}/ ./ --recursive\n")
    rc, tid, err = gl("transfer", f"{U.NERSC_EP}:{R}/", f"{U.VISTA_EP}:{U.VISTA_CACHE}/", "--batch", f.name, "--sync-level", "size",
                      "--verify-checksum", "--label", f"conveyor cache {uid}", "--notify", "failed", "--jmespath", "task_id", "--format", "unix")
    os.remove(f.name)
    if rc == 0 and tid:
        u["cache_task"] = tid; log("unit", uid, "caching downloads on Vista, task", tid)


def feed_detection(st, cu, jobs, log):
    if any(j[1].startswith("l5c-") for j in jobs):
        return                                    # the LDT 5 % (ldt5/debug_feed.py) has the debug queue first
    dbg = [j for j in jobs if j[3] == "debug"]
    room = DEBUG_MAX - len(dbg)
    if any(j[1].startswith("cvb-") and j[2] == "PENDING" for j in jobs):
        room = 0                                  # a scenario build waits for a debug slot: builds go first (u16, 11-20 10:56-11:50)
    reg_ok = None
    for u in sorted(cu, key=lambda u: u.get("prio", 9)):     # a15 (prio 0) first: its detected gap files free 5.9 TB
        if u["stage"] != "prep":
            continue
        for share, (left, total, final) in sorted(u.get("detect", {}).items(), key=lambda kv: (not kv[0].startswith("m"), kv[0])):
            if left <= 0:
                continue
            name = f"cvd-{u['id']}-{share}"
            if any(j[1] == name for j in jobs):
                continue
            if reg_ok is None:                    # CPU regular when it would start within 30 min
                rc, o, e = sh("sbatch --test-only -q regular -C cpu -N 1 -A $NERSC_CPU_ACCOUNT -t 03:00:00 --wrap true 2>&1 | tail -1")
                try:
                    t = time.mktime(time.strptime((o + e).split("start at ")[1].split()[0], "%Y-%m-%dT%H:%M:%S"))
                    reg_ok = t - time.time() < 1800
                except Exception:
                    reg_ok = False
            R = f"{NR}/{u['id']}"
            # 2026-11-20: the debug QOS runs 2 jobs per user, about 1,500 exposures an hour;
            # a share with a large backlog also gets a regular CPU job (3 h, queued even when it will not start soon), at
            # most REG_MAX of them; CPU hours are plentiful ($NERSC_CPU_ACCOUNT ~17,000 node-h left on 11-20).
            nreg = sum(1 for j in jobs if j[1].startswith("cvd-") and j[3] != "debug")
            use_reg = (reg_ok or left >= 300) and nreg < REG_MAX
            if use_reg:
                a = f"-q regular -t 03:00:00 -J {name} --export=ALL,DIR={R}/{share},TL=10500"
            elif room > 0:
                a = f"-q debug -t 00:30:00 -J {name} --export=ALL,DIR={R}/{share},TL=1680"; room -= 1
            else:
                continue
            rc, o, e = sh(f"cd {R} && sbatch {a} {NR}/unit_detect.sbatch")
            log("unit", u["id"], "detect", share, f"{left} left:", (o.strip() or e.strip()[-120:]))
            jobs.append(["", name, "PENDING", "regular" if use_reg else "debug"])


def next_unit(st, cu, log):
    if os.path.exists(os.path.join(H, "NO_NEW_UNITS")):
        return
    ahead = [u for u in st["units"] if u.get("stage") in ("selected", "prep") or (u.get("stage") == "search" and u.get("build") != "done")]
    # 2026-11-19: the next unit is chosen as soon as every unit
    # in prep has its mirror and Vista-cache shares detected (only NOIRLab downloads left), so detection never idles;
    # at most 2 units in prep, at most MAX_AHEAD selected / in prep / waiting for search.
    busy = [u for u in cu if u["stage"] == "selected" or (u["stage"] == "prep" and (not u.get("detect") or any(
        v[0] > 0 for k, v in u["detect"].items() if not k.startswith("p"))))]
    prep = [u for u in cu if u["stage"] in ("selected", "prep")]
    waiting = [u for u in st["units"] if u.get("stage") in ("selected", "prep", "search")]
    # 2026-11-20 16:00: 3 units in prep (was 2): u17 and u18 had only NOIRLab gaps left, so no unit was chosen and the GPUs
    # were about to idle; early builds search a unit's ready regions while its gap downloads
    if busy or len(prep) >= PREP_MAX or len(waiting) >= MAX_AHEAD:
        return
    n = max([int(u["id"][1:]) for u in cu if u["id"][1:].isdigit()] + [4]) + 1
    uid = f"u{n:02d}"
    if os.path.exists(os.path.join(U.U0, uid)):
        log("next unit", uid, "folder exists; not selecting"); return
    try:
        U.new(uid)
    except SystemExit as e:
        log("next unit", uid, "selection failed:", str(e)[:200]); return
    info = json.load(open(os.path.join(U.U0, uid, "unit.json")))
    st["units"].append(dict(id=uid, kind="conveyor", stage="selected", prio=len(st["units"]) + 1, made=info["made"]))
    log("next unit", uid, "selected:", json.dumps(info))
