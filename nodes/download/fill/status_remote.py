#!/usr/bin/env python3
"""NERSC side of fill/dash/build.py: one JSON with the download parts' state (latest fetch log line, files and bytes done,
bad md5, errors, waiting or running) of the fill (decam-fill) and of the LDT-weighted fourth 5 % (ldt5, since
2026-11-18), detection progress of ldt5's and archive15's shares, user scratch use and quota, and our Slurm jobs.
Run with dbt/env python (3.11)."""
import glob, json, os, re, subprocess, time
now = time.time()


def scan(F, manifest, label):
    size = {}
    for l in open(f"{F}/{manifest}"):
        m, s, n, t = l.rstrip("\n").split("\t"); size[n] = (int(s), t)
    out = []
    for p in sorted(glob.glob(f"{F}/p[0-9]*"), key=lambda x: (len(x), x)):
        d = dict(part=(label + "-" if label else "") + os.path.basename(p), set=label or "fill", mbs=0, run_mbs=None, files=0, tb=0.0,
                 a15_files=0, bad_md5=0, errors=0, failed=0, state="not started", log_age_min=None)
        names = set()
        for f in glob.glob(f"{p}/log/done_*.txt"):
            names.update(x.strip() for x in open(f) if x.strip())
        d["files"] = len(names); d["tb"] = round(sum(size.get(n, (0, ""))[0] for n in names) / 1e12, 3)
        d["a15_files"] = sum(1 for n in names if size.get(n, (0, ""))[1] == "a15")
        logs = sorted(glob.glob(f"{p}/log/fetch_*.log"), key=os.path.getmtime)
        if logs:
            age = (now - os.path.getmtime(logs[-1])) / 60; d["log_age_min"] = round(age, 1)
            L = [l for l in open(logs[-1]) if "MB/s now" in l]
            if L:
                m = re.search(r"(\d+) MB/s now, (\d+) MB/s run, ([\d.]+) TB, files (\d+), bad md5 (\d+), errors (\d+), failed (\d+)", L[-1])
                if m:
                    d["mbs"] = int(m.group(1)) if age < 5 else 0; d["run_mbs"] = int(m.group(2))
                    d["bad_md5"] = int(m.group(5)); d["errors"] = int(m.group(6)); d["failed"] = int(m.group(7))
            d["state"] = "downloading" if age < 5 else "waiting"
        g = f"{p}/log/guard.log"
        if os.path.exists(g) and d["state"] != "downloading":
            last = open(g).read().strip().splitlines()[-1:]
            if last and "waiting" in last[0]: d["state"] = "waiting for space"
        if os.path.exists(f"{p}/log/SHARE_DONE"): d["state"] = "finished"
        out.append(d)
    total = dict(files=len(size), tb=round(sum(v[0] for v in size.values()) / 1e12, 3))
    return out, total


parts, _ = scan("$NERSC_SCRATCH/decam-fill", "blank.tsv", "")
L5 = "$NERSC_SCRATCH/ldt5"
lparts, ltotal = scan(L5, "missing.tsv", "ldt5") if os.path.exists(f"{L5}/missing.tsv") else ([], {})
det = []
for base, pat in ((L5, "l5_*"), (L5, "p[0-9]*"), ("$NERSC_SCRATCH/archive15", "s15_*")):
    for d in sorted(glob.glob(f"{base}/{pat}")):
        if not os.path.isdir(f"{d}/img"):
            continue
        n_img = sum(1 for _ in glob.iglob(f"{d}/img/*/*_ooi_*.fits.fz"))
        n_det = sum(1 for _ in glob.iglob(f"{d}/det/*/c4d_*.npz")) + sum(1 for _ in glob.iglob(f"{d}/det/c4d_*.npz"))
        det.append(dict(share=os.path.basename(d), set="archive15" if pat.startswith("s15") else "ldt5", exposures=n_img, detected=n_det))
q = subprocess.run(["lfs", "quota", "-q", "-u", "$NERSC_USER", "/pscratch"], capture_output=True, text=True).stdout.split()
jobs = []
for line in subprocess.run(["squeue", "-u", "$NERSC_USER", "-h", "-o", "%i|%j|%q|%t|%M|%R"], capture_output=True, text=True).stdout.splitlines():
    i, j, qo, st, tm, r = line.split("|")
    jobs.append(dict(job=i, name=j, qos=qo, state={"R": "running", "PD": "waiting"}.get(st, st), time=tm, where=r))
print(json.dumps(dict(at=time.strftime("%Y-%m-%d %H:%M"), scratch_tib=round(int(q[1]) / 2**30, 2), quota_tib=round(int(q[2]) / 2**30, 1) if len(q) > 2 and q[2].isdigit() else 20.0,
                      parts=parts, ldt5_parts=lparts, ldt5_total=ltotal, detection=det, jobs=jobs)))
