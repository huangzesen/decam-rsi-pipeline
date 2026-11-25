#!/usr/bin/env python3
"""Read the archive5 jobs' progress from TACC (read-only, through the existing SSH masters) and update
archive5/regions.json: per region tb_downloaded, exp_detected, detections, status.

Remote: $SCRATCH/archive5/log/done_*.txt (files fetched), det/detect.log ("<stem> ok N det ..." lines),
log/status_*.json (fetch rates). Also writes archive5/tacc/remote_status.json (rates, errors) for the page.
Run: python3 -I tacc/pull_status.py   (from archive5/)
"""
import json, os, subprocess, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); A5 = os.path.dirname(HERE)
HOST = {"ls6": "$TACC_USER@ls6.tacc.utexas.edu", "vista": "$TACC_USER@vista.tacc.utexas.edu"}
REMOTE = "$TACC_SCRATCH/archive5"


def remote(m, cmd):
    s = os.path.expanduser(f"$SSH_CM_DIR/{m}-$TACC_USER.sock")
    if not os.path.exists(s):
        return None
    r = subprocess.run(["ssh", "-S", s, "-o", "ControlMaster=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST[m], cmd],
                       capture_output=True, text=True, timeout=120)
    return r.stdout if r.returncode == 0 else None


R = json.load(open(os.path.join(A5, "regions.json")))
rexp = json.load(open(os.path.join(A5, "regions_exposures.json")))
size = {}; exp_region = defaultdict(set)
for rid in rexp:
    for line in open(os.path.join(A5, "manifests", rid + ".tsv")):
        md5, sz, name, _ = line.rstrip("\n").split("\t")
        size[name] = int(sz); exp_region[name[:17]].add(rid)

done_files = set(); det = {}; rs = {}
DIRS = [("ls6", REMOTE, "ls6"), ("vista", REMOTE, "vista"), ("vista", REMOTE + "/n2", "vista-n2")]
for m, rdir, label in DIRS:
    out = remote(m, f"cd {rdir} 2>/dev/null && cat log/done_*.txt 2>/dev/null; echo '@@DET'; "
                    f"grep -h ' ok \\| ERROR' det/detect.log 2>/dev/null; echo '@@ST'; cat log/status_*.json 2>/dev/null; echo; "
                    f"echo '@@DU'; du -sb img det 2>/dev/null; true")
    if out is None:
        rs[label] = {"reachable": False}; continue
    a, rest = out.split("@@DET", 1); b, rest = rest.split("@@ST", 1); c, d = rest.split("@@DU", 1)
    done_files |= {x.strip() for x in a.splitlines() if x.strip()}
    nerr = 0
    for line in b.splitlines():
        p = line.split()
        if len(p) >= 5 and p[3] == "ok":
            det[p[2]] = int(p[4])
        elif "ERROR" in line:
            nerr += 1
    st = {}
    try:
        st = json.loads(c.strip()) if c.strip() else {}
    except Exception:
        pass
    du = {x.split()[1]: int(x.split()[0]) for x in d.splitlines() if len(x.split()) == 2}
    rs[label] = dict(reachable=True, fetch=st, detect_errors=nerr, img_bytes=du.get("img", 0), det_bytes=du.get("det", 0))

for r in R["regions"]:
    rid = r["id"]; stems = rexp[rid]
    names = [n for n, s in size.items() if rid in exp_region[n[:17]]]
    r["tb_downloaded"] = round(sum(size[n] for n in names if n in done_files) / 1e12, 3)
    r["exp_detected"] = sum(1 for s in stems if s in det)
    r["detections"] = sum(det.get(s, 0) for s in stems)
    r["exp_deleted"] = 0
    nodes = [v for k, v in rs.items() if k.startswith(r.get("machine") or "-") and v.get("fetch")]
    fetch_over = bool(r.get("pilot")) and bool(nodes) and all(v["fetch"].get("files_left") == 0 for v in nodes)   # every file tried; the rest failed for good
    if r["exp_detected"] and (r["exp_detected"] >= len(stems) or fetch_over):
        r["status"] = "extracted"
    elif r["exp_detected"]:
        r["status"] = "detecting"
    elif r["tb_downloaded"]:
        r["status"] = "downloading"
R["asof"] = time.strftime("%Y-%m-%d %H:%M")
json.dump(R, open(os.path.join(A5, "regions.json.tmp"), "w"), indent=1)
os.replace(os.path.join(A5, "regions.json.tmp"), os.path.join(A5, "regions.json"))
json.dump(dict(asof=R["asof"], machines=rs), open(os.path.join(HERE, "remote_status.json"), "w"), indent=1)
pil = [r for r in R["regions"] if r.get("pilot")]
print(R["asof"], {m: (v.get("fetch", {}).get("mb_s_run"), v.get("fetch", {}).get("tb_run"), v.get("detect_errors")) for m, v in rs.items()})
for r in pil:
    print(r["id"], r["machine"], r["status"], r["tb_downloaded"], "/", r["tb"], "TB;", r["exp_detected"], "/", r["exposures"], "exp;", r["detections"], "det")
