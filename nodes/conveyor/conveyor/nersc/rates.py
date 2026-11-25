#!/usr/bin/env python3
"""conveyor (NERSC): one JSON line with the download rate now (sum of the latest "MB/s now" of every fetch log written in the
last 5 minutes: the fill's parts and every unit's parts), the number of such logs, scratch use, and the fill's limits."""
import glob, json, os, re, subprocess, time
now = time.time(); mbs = 0; n = 0
for f in glob.glob("$NERSC_SCRATCH/decam-fill/p*/log/fetch_*.log") + glob.glob("$NERSC_SCRATCH/units/*/p*/log/fetch_*.log"):
    if now - os.path.getmtime(f) > 300:
        continue
    with open(f, "rb") as fh:
        fh.seek(max(0, os.path.getsize(f) - 4000)); tail = fh.read().decode(errors="replace").splitlines()
    m = [re.search(r"(\d+) MB/s now", l) for l in tail]; m = [x for x in m if x]
    if m:
        mbs += int(m[-1].group(1)); n += 1
q = subprocess.run(["lfs", "quota", "-q", "-u", "$NERSC_USER", "/pscratch"], stdout=subprocess.PIPE, universal_newlines=True).stdout.split()
lim = dict(l.split("#")[0].strip().split("=") for l in open("$NERSC_SCRATCH/decam-fill/limits") if "=" in l)
print(json.dumps(dict(mbs=mbs, logs=n, scratch_tib=round(int(q[1].strip("[]*")) / 2**30, 2), low_tib=round(int(lim["LOW_KB"]) / 2**30, 2), high_tib=round(int(lim["HIGH_KB"]) / 2**30, 2))))
