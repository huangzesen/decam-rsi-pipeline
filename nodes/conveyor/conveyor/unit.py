#!/usr/bin/env python3
"""conveyor: make one unit (the next 5 % of DECam's sky) ready for detection at NERSC.

  new UID      select the unit in planet-search/units/UID/ (a copy of ldt5/select_ldt.py: LDT-weighted, scored on
               exposures already at NERSC or cached on Vista, every searched exposure left out), then split it:
                 ls_nersc.txt, stems_<k>.txt   exposures in the DESI Legacy Surveys mirror at NERSC (linked, 8 shares)
                 vista_pull.tsv                exposures whose three files the fill relay already put on Vista (Globus back)
                 missing.tsv                   the rest, downloaded from NOIRLab at NERSC (fetch.py format, tag UID)
               regions over 2,000 exposures are cut into parts at the longest gaps between nights; regions_search.json
               (rid, sid, n_exp, ldt_h) is what the coordinator searches; search_in/regions_exposures.json is the build input.
  known UID    start the known-body positions (JPL Horizons; 4 processes on the Mac) -> known/<rid>.json
  known-done UID   True when every region has known/<rid>.json; then search_in/known/ is filled (parts share their region's)
  stage UID    copy the lists to NERSC units/UID, link the mirror shares, queue the downloads (free xfer jobs), start the
               Globus pull from Vista, add the downloaded names to the fill's skip list
Used by conveyor.py; can be run by hand. Never writes into an existing unit folder for `new`."""
import datetime, glob, json, math, os, secrets, shutil, subprocess, sys, tempfile, time

H = os.path.dirname(os.path.abspath(__file__)); PS = os.path.dirname(H); U0 = os.path.join(PS, "units")
MIRROR = os.path.join(PS, "linker", "data_real", "nersc_decam_mirror", "mirror_exposures.tsv")
MIRROR_BASE = "/global/cfs/cdirs/cosmo/staging/decam/"
RELAYED = os.path.join(PS, "fill", "relay", "relayed.txt")
SEARCHED = ["archive5/regions_exposures.json", "archive10/stepA/regions_exposures.json", "archive10/stepB/regions_exposures.json",
            "archive15/regions_exposures.json", "ldt5/regions_exposures.json"]
NR = "$NERSC_SCRATCH/units"; DBT = "$NERSC_SCRATCH/dbt"
NERSC_EP = "$GLOBUS_NERSC_COLLECTION"; VISTA_EP = "$GLOBUS_VISTA_COLLECTION"
VISTA_CACHE = "$TACC_SCRATCH/decam-fill/img"
SSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "$NERSC_USER@perlmutter.nersc.gov"]
SCP = ["scp", "-q", "-o", f"ControlPath={os.path.expanduser('$SSH_CM_DIR/nersc-$NERSC_USER.sock')}"]
PYA = os.path.expanduser("~/anaconda3/bin/python")          # has astropy (known_pos.py)



PAUSE_F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NERSC_DL_PAUSE")


def nersc_dl_paused():
    """NOIRLab downloads at NERSC paused until the epoch in conveyor/NERSC_DL_PAUSE (2026-11-20 17:40, : NOIRLab refused most NERSC connections)."""
    try:
        return time.time() < float(open(PAUSE_F).read().split()[0])
    except (OSError, ValueError, IndexError):
        return False

def sh(cmd, timeout=900):
    p = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout); return p.returncode, p.stdout, p.stderr


def day(s):
    return datetime.date(2000 + int(s[4:6]), int(s[6:8]), int(s[8:10]))


def new(uid):
    D = os.path.join(U0, uid)
    if os.path.exists(D):
        sys.exit(f"{D} exists: not writing into it")
    os.makedirs(D)
    # since 2026-11-20 16:20 the conveyor's own selector (ldt5's without the LDT filter)
    src = open(os.path.join(H, "select_units.py")).read()
    assert src.count("PS = os.path.dirname(HERE)\n") == 1
    open(os.path.join(D, "select_ldt.py"), "w").write(src.replace("PS = os.path.dirname(HERE)\n", "PS = os.path.dirname(os.path.dirname(HERE))   # units/<uid>\n"))
    searched = [os.path.join(PS, p) for p in SEARCHED] + sorted(p for p in glob.glob(os.path.join(U0, "*", "regions_exposures.json")) if f"/{uid}/" not in p)
    cmd = ["python3", "-I", os.path.join(D, "select_ldt.py"), "--present", MIRROR, RELAYED, "--searched", *searched]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=PS)
    open(os.path.join(D, "select.log"), "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    if p.returncode != 0 or not os.path.exists(os.path.join(D, "regions.json")):
        sys.exit(f"selector failed: {p.stderr[-400:]}")
    prepare(uid)


def prepare(uid):
    D = os.path.join(U0, uid)
    regs = json.load(open(os.path.join(D, "regions.json")))["regions"]; rexp = json.load(open(os.path.join(D, "regions_exposures.json")))
    mp = {}
    for l in open(MIRROR):
        if l.startswith("c4d_"):
            f = l.rstrip("\n").split("\t"); mp[f[0]] = MIRROR_BASE + f[4]
    relayed = set(l.strip() for l in open(RELAYED)) if os.path.exists(RELAYED) else set()
    files = {}
    for f in glob.glob(os.path.join(D, "manifests", "*.tsv")):
        for l in open(f):
            m = l.rstrip("\n").split("\t")
            files.setdefault(m[2][:17], {})[m[2]] = (m[0], m[1])
    seen, mirror, pull, miss = set(), [], [], []
    for r in regs:                                                     # best regions first
        for s in rexp[r["id"]]:
            if s in seen:
                continue
            seen.add(s); fs = files.get(s, {})
            if s in mp:
                mirror.append(s)
            elif fs and all(n in relayed for n in fs):
                pull.extend(sorted(fs))
            else:
                miss.extend(f"{md5}\t{size}\t{n}\t{uid}" for n, (md5, size) in sorted(fs.items()))
    open(os.path.join(D, "ls_nersc.txt"), "w").write("".join(f"{s} {mp[s]}\n" for s in mirror))
    K = 8; size = max(1, -(-len(mirror) // K))
    for k in range(K):
        open(os.path.join(D, f"stems_{k}.txt"), "w").write("\n".join(mirror[k * size:(k + 1) * size]) + "\n")
    open(os.path.join(D, "vista_pull.txt"), "w").write("".join(n + "\n" for n in pull))
    open(os.path.join(D, "missing.tsv"), "w").write("".join(x + "\n" for x in miss))
    # search regions: parts for regions over 2,000 exposures, fresh sids
    out_rexp, out_reg = {}, []
    os.makedirs(os.path.join(D, "search_in", "known"), exist_ok=True)
    for r in regs:
        st = sorted(rexp[r["id"]], key=lambda s: s[4:])
        if len(st) <= 2000:
            out_rexp[r["id"]] = st; out_reg.append(dict(rid=r["id"], sid="s" + secrets.token_hex(4), n_exp=len(st), ldt_h=r.get("ldt_hours", 0))); continue
        k = math.ceil(len(st) / 1800); cuts = []
        for j in range(1, k):
            lo, hi = int(len(st) * (j / k - 0.15)), int(len(st) * (j / k + 0.15))
            cuts.append(max(range(max(lo, 1), min(hi, len(st) - 1)), key=lambda i: (day(st[i]) - day(st[i - 1])).days))
        b = [0] + sorted(set(cuts)) + [len(st)]
        for j in range(len(b) - 1):
            rid = r["id"] + "abcdefgh"[j]; out_rexp[rid] = st[b[j]:b[j + 1]]
            out_reg.append(dict(rid=rid, sid="s" + secrets.token_hex(4), n_exp=b[j + 1] - b[j], ldt_h=r.get("ldt_hours", 0), part_of=r["id"]))
    json.dump(out_rexp, open(os.path.join(D, "search_in", "regions_exposures.json"), "w"))
    json.dump(out_reg, open(os.path.join(D, "regions_search.json"), "w"), indent=1)
    exp = len(seen); tb = sum(int(sz) for s in seen for _, sz in files.get(s, {}).values()) / 1e12
    info = dict(uid=uid, made=time.strftime("%Y-%m-%d %H:%M"), regions=len(regs), scenarios=len(out_reg), exposures=exp, tb=round(tb, 2),
                mirror=len(mirror), vista_files=len(pull), noirlab_files=len(miss),
                noirlab_tb=round(sum(int(x.split("\t")[1]) for x in miss) / 1e12, 2), ldt_h=[min(r.get("ldt_hours", 0) for r in regs), max(r.get("ldt_hours", 0) for r in regs)] if regs else None)
    json.dump(info, open(os.path.join(D, "unit.json"), "w"), indent=1)
    print(json.dumps(info))


def known(uid):
    D = os.path.join(U0, uid)
    src = open(os.path.join(PS, "ldt5", "known_pos.py")).read().replace(
        "PS = os.path.dirname(HERE)   # ldt5 sits one level below planet-search", "PS = os.path.dirname(os.path.dirname(HERE))   # units/<uid> sits two levels below planet-search")
    assert "units/<uid> sits two levels" in src
    open(os.path.join(D, "known_pos.py"), "w").write(src)
    rids = [r["id"] for r in json.load(open(os.path.join(D, "regions.json")))["regions"]]
    os.makedirs(os.path.join(D, "log"), exist_ok=True)
    for k in range(4):
        g = ",".join(rids[k::4])
        if g:
            subprocess.Popen(["nohup", PYA, "-I", os.path.join(D, "known_pos.py"), g], stdout=open(os.path.join(D, "log", f"known_{k}.log"), "w"),
                             stderr=subprocess.STDOUT, cwd=PS, start_new_session=True)
    print(uid, "known-body positions started for", len(rids), "regions")


def known_done(uid):
    D = os.path.join(U0, uid)
    rids = [r["id"] for r in json.load(open(os.path.join(D, "regions.json")))["regions"]]
    if not all(os.path.exists(os.path.join(D, "known", r + ".json")) for r in rids):
        return False
    for r in json.load(open(os.path.join(D, "regions_search.json"))):
        shutil.copy(os.path.join(D, "known", r.get("part_of", r["rid"]) + ".json"), os.path.join(D, "search_in", "known", r["rid"] + ".json"))
    return True


def under_cap():
    """True while NERSC scratch use is below the fill's LOW_KB (decam-fill/limits)."""
    rc, o, _ = sh("q=$(lfs quota -q -u $NERSC_USER /pscratch | awk 'NR==1{print $2}'); . $NERSC_SCRATCH/decam-fill/limits; echo $q $LOW_KB")
    try:
        q, low = (int(x.strip("*[]")) for x in o.split()[:2])
        return q < low
    except Exception:
        return False


def start_pull(uid):
    """Globus pull of the unit's exposures that are in the Vista cache into units/<uid>/v/img; returns the task id."""
    D = os.path.join(U0, uid); R = f"{NR}/{uid}"
    pull = [l.strip() for l in open(os.path.join(D, "vista_pull.txt")) if l.strip()]
    if not pull:
        sh(f"touch {R}/v/log/PULL_DONE"); return None
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        for n in pull:
            f.write(f"{n[4:10]}/{n} {n[4:10]}/{n}\n")
    p = subprocess.run(["globus", "transfer", f"{VISTA_EP}:{VISTA_CACHE}/", f"{NERSC_EP}:{R}/v/img/", "--batch", f.name, "--sync-level", "size",
                        "--verify-checksum", "--label", f"conveyor pull {uid}", "--notify", "failed", "--jmespath", "task_id", "--format", "unix"],
                       capture_output=True, text=True, env=dict(os.environ, LC_ALL="C", LANG="C"), timeout=600)
    os.remove(f.name)
    return p.stdout.strip() or f"failed: {p.stderr.strip()[-160:]}"


def stage(uid, xfer_slots=7):
    D = os.path.join(U0, uid); R = f"{NR}/{uid}"
    rc, out, err = sh(f"mkdir -p {R}/log {R}/v/img {R}/v/log {NR}/slurm {DBT}/units/{uid}/in/known {DBT}/units/{uid}/log")
    files = ["ls_nersc.txt", "missing.tsv", "vista_pull.txt"] + [os.path.basename(p) for p in glob.glob(os.path.join(D, "stems_*.txt"))]
    subprocess.run(SCP + [os.path.join(D, f) for f in files] + [f"$NERSC_USER@perlmutter.nersc.gov:{R}/"], check=True)
    subprocess.run(SCP + [os.path.join(H, "nersc", f) for f in ("unit_fetch.sbatch", "unit_farm.py", "remaining.py", "unit_detect.sbatch", "unit_build.sbatch")]
                   + [f"$NERSC_USER@perlmutter.nersc.gov:{NR}/"], check=True)
    print(sh(f"cd {R} && {DBT}/env/bin/python -I {NR}/unit_farm.py {R}", timeout=3600)[1].strip())
    # downloads: free xfer jobs, at most the free slots (15 running per user); while NERSC downloads are paused the shares
    # are still counted (Lonestar6 gap shares use the same n) and units_side.keep_downloads submits them after the pause
    n_miss = sum(1 for _ in open(os.path.join(D, "missing.tsv")))
    jobs = []
    if n_miss:
        rc, q, _ = sh("squeue -u $NERSC_USER -h -q xfer -o %i | wc -l")
        n = max(1, min(xfer_slots, 15 - int(q.strip() or 0), math.ceil(n_miss / 2000)))
        for k in range(n):
            if nersc_dl_paused():
                jobs.append("paused"); continue
            rc, o, e = sh(f"cd {R} && sbatch -q xfer -C cron --mem=8G -t 48:00:00 -J cvf-{uid}-{k} --export=ALL,U={R},PART={k}/{n},PARTDIR=p{k} {NR}/unit_fetch.sbatch")
            jobs.append(o.strip().split()[-1] if rc == 0 and o.strip() else f"failed: {e.strip()[-120:]}")
        # the fill skips what this unit downloads (fetch.py reads blank.skip.txt at each chunk)
        sh(f"cut -f3 {R}/missing.tsv >> $NERSC_SCRATCH/decam-fill/blank.skip.txt")
    # pull from the Vista cache with Globus, only while NERSC scratch is under the download cap (2026-11-19, ; u06's 1.3 TB pull had taken scratch from 16.7 to 17.6 TiB); otherwise units_side starts it later
    task = start_pull(uid) if under_cap() else "deferred"
    info = json.load(open(os.path.join(D, "unit.json"))); info.update(staged=time.strftime("%Y-%m-%d %H:%M"), fetch_jobs=jobs, pull_task=task)
    json.dump(info, open(os.path.join(D, "unit.json"), "w"), indent=1)
    print(uid, "staged: fetch jobs", jobs, "pull task", task)
    return info


if __name__ == "__main__":
    c, uid = sys.argv[1], sys.argv[2]
    {"new": new, "prepare": prepare, "known": known, "stage": stage}.get(c, lambda u: print(known_done(u)))(uid)
