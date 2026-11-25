#!/usr/bin/env python3
"""Status of every DECam data movement for the private claude.ai Data artifact "DECam Data Movement"
(<private-page-url>; 2026-11-17). Collects, read-only:
  NERSC  status_remote.py (per download part: speed, files, TB, errors; scratch use; our Slurm jobs), one ssh call
  Vista  taccinfo (scratch use, SU balance), one ssh call
  Globus every task of ours since 2026-11-17 (archive10 a10n, archive5, fill relay)
  local  fill/relay (relayed files), fill/release (released folders), blank.json, events.jsonl (hand-written)
and writes out/status.json (one list per dataset), appending one row per build to history.jsonl.
Then: upload out/status.json to the artifact (Artifact publish, asset: true) and point the datasets at it (docs.py).
Usage: python3 -I fill/dash/build.py"""
import json, os, re, subprocess, time
from datetime import datetime
D = os.path.dirname(os.path.abspath(__file__)); F = os.path.dirname(D); P = os.path.dirname(F)
OUT = os.path.join(D, "out"); os.makedirs(OUT, exist_ok=True)
ENV = dict(os.environ, LC_ALL="C", LANG="C")
NS = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "$NERSC_USER@perlmutter.nersc.gov"]
VS = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/vista-$TACC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "$TACC_USER@vista.tacc.utexas.edu"]
def run(cmd, timeout=600):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=timeout); return p.stdout if p.returncode == 0 else ""
    except Exception:
        return ""
now = datetime.now().strftime("%Y-%m-%d %H:%M")
# NERSC
R = json.loads(run(NS + ["$NERSC_SCRATCH/dbt/env/bin/python -I $NERSC_SCRATCH/decam-fill/status_remote.py"]) or "{}")
parts = R.get("parts", []); jobs_n = R.get("jobs", [])
kind = {"p0": "free (login node)", "p1": "free (login node)", "p2": "free (login node)", "p3": "free (login node)", "p4": "CPU node (charged)", "p5": "CPU node (charged)"}
jobname = {"p0": "fillx0", "p1": "fillx1", "p2": "fillx2", "p3": "fillx3", "p4": "fillc4", "p5": "fillc5"}
js = {j["name"]: j for j in jobs_n}
share = {"p0": "0/6", "p1": "1/6", "p2": "2/6", "p3": "3/6", "p41": "4/12", "p42": "10/12", "p51": "5/12", "p52": "11/12"}
for p in parts:
    nm = "fillx" + p["part"][1:]
    j = js.get(nm, {}); p["kind"] = f"free (login node), share {share.get(p['part'], '?')}" if j.get("qos", "xfer") == "xfer" else "CPU node (charged)"
    p["job"] = j.get("job", "—"); p["job_state"] = j.get("state", "ended")
lparts = R.get("ldt5_parts", []); ltot = R.get("ldt5_total", {}); detn = R.get("detection", [])
for p in lparts:                                  # the LDT-weighted fourth 5 % (2026-11-18): its own download jobs ldt5x<k>
    k = p["part"].split("-p")[-1]; j = js.get(f"ldt5x{k}", {})
    p["kind"] = f"free (login node), LDT 5 % share {k}/7"; p["job"] = j.get("job", "—"); p["job_state"] = j.get("state", "ended")
    if p["state"] == "waiting" and not j: p["state"] = "finished" if p["files"] else p["state"]
# Vista
vt = run(VS + ["/usr/local/etc/taccinfo"])
vista_scratch_gb, su = None, None
for line in vt.splitlines():
    if "/scratch" in line:
        try: vista_scratch_gb = float(line.split("|")[1].split()[1])
        except Exception: pass
    if "TG-NAIRR" in line:
        try: su = int(line.split("|")[1].split()[1])
        except Exception: pass
# Globus tasks of ours since 2026-11-17
gt = json.loads(run(["globus", "task", "list", "--limit", "200", "--filter-requested-after", "2026-11-17T00:00:00", "-F", "json"]) or '{"DATA": []}').get("DATA", [])
def purpose(lab):
    lab = lab or ""
    if lab.startswith("archive10"): return "second 5 % downloads to Vista"
    if lab.startswith("archive5"): return "first 5 % images to Vista"
    if lab.startswith("decam fill relay"): return "fill relay to Vista"
    return "other"
tasks = []
for t in gt:
    if purpose(t.get("label")) == "other": continue
    st = t["status"]; nice = t.get("nice_status") or ""
    state = "queued" if (st == "ACTIVE" and nice == "Queued") else {"ACTIVE": "copying", "SUCCEEDED": "done", "FAILED": "failed or cancelled", "INACTIVE": "paused"}.get(st, st.lower())
    tasks.append(dict(task=t["task_id"][:8], label=t.get("label") or "", purpose=purpose(t.get("label")), state=state,
                      tb=round((t.get("bytes_transferred") or 0) / 1e12, 3), files=t.get("files_transferred") or 0, files_total=t.get("files") or 0,
                      mbs=round((t.get("effective_bytes_per_second") or 0) / 1e6), requested=(t.get("request_time") or "")[:16].replace("T", " ")))
order = {"copying": 0, "queued": 1, "paused": 2, "failed or cancelled": 3, "done": 4}
tasks.sort(key=lambda r: (order.get(r["state"], 5), r["requested"]))
# local state
blank = json.load(open(os.path.join(F, "blank.json")))
size = {}
for l in open(os.path.join(F, "blank.tsv")):
    m, s, n, tg = l.rstrip("\n").split("\t"); size[n] = int(s)
rel = os.path.join(F, "relay", "relayed.txt"); relayed = [l.strip() for l in open(rel)] if os.path.exists(rel) else []
on_vista_tb = round(sum(size.get(n, 0) for n in relayed) / 1e12, 3)
RT = json.load(open(os.path.join(F, "release", "tasks.json"))) if os.path.exists(os.path.join(F, "release", "tasks.json")) else []
released_tb = round(sum((t["released"] or {}).get("tb", 0) for t in RT if isinstance(t.get("released"), dict)), 2) + 1.90   # + a10n_2 released by hand 21:52
a5 = [t for t in RT if t["root"].endswith("archive5")]; a10 = [t for t in RT if t["root"].endswith("archive10")]
a5_done = sum(1 for t in a5 if isinstance(t.get("released"), dict)); a10_done = sum(1 for t in a10 if isinstance(t.get("released"), dict)) + 1
fetched_tb = round(sum(p["tb"] for p in parts), 3); files_done = sum(p["files"] for p in parts); mbs = sum(p["mbs"] for p in parts + lparts)
# 2026-11-21: the NERSC fill parts above ended on 11-20; downloads now run on every device (conveyor units at NERSC and
# Lonestar6, Vista's gg node, the Mac). "Fetched" = blank files already held: on Vista (relay/relayed.txt, which
# vista_known.py fills with Vista and Lonestar6 names) or relayed from the Mac to NERSC; speed = all devices together
# from dl_watch.log's last line.
_held = set(relayed)
_mr = os.path.join(F, "relay_mac", "relayed.txt")
if os.path.exists(_mr):
    _held |= {l.strip() for l in open(_mr) if l.strip()}
_held = [n for n in _held if n in size]
if sum(size[n] for n in _held) / 1e12 > fetched_tb:
    fetched_tb = round(sum(size[n] for n in _held) / 1e12, 3); files_done = len(_held)
try:
    _last = [l for l in open(os.path.join(F, "dl_watch.log")) if " MB/s, " in l][-1]
    mbs = max(mbs, sum(int(x) for x in re.findall(r"(\d+) MB/s, \d+/\d+ streams", _last)))
except (OSError, IndexError):
    pass
a15_done = sum(p["a15_files"] for p in parts); a15_total = sum(1 for l in open(os.path.join(F, "relay", "a15_names.txt")) if l.strip()) if os.path.exists(os.path.join(F, "relay", "a15_names.txt")) else 30543
left_tb = blank["blank_tb"] - fetched_tb
eta_days = round(left_tb * 1e12 / (mbs * 1e6) / 86400, 1) if mbs > 0 else None
# Globus copy speed NERSC -> Vista: bytes moved by all our tasks since the last build / time between builds
# (first build, or a gap over 2 h: the copying tasks' own average rates instead)
gl_tb = sum((t.get("bytes_transferred") or 0) for t in gt if purpose(t.get("label")) != "other") / 1e12
hp = os.path.join(D, "history.jsonl")
prev = [h for h in (json.loads(l) for l in open(hp) if l.strip()) if h.get("gl_tb") is not None][-1:] if os.path.exists(hp) else []
dt = (datetime.strptime(now, "%Y-%m-%d %H:%M") - datetime.strptime(prev[0]["t"], "%Y-%m-%d %H:%M")).total_seconds() if prev else 0
if prev and 120 <= dt <= 7200 and gl_tb >= prev[0]["gl_tb"]:
    globus_mbs = round((gl_tb - prev[0]["gl_tb"]) * 1e6 / dt)
else:
    globus_mbs = round(sum((t.get("effective_bytes_per_second") or 0) for t in gt if purpose(t.get("label")) != "other" and t["status"] == "ACTIVE" and t.get("nice_status") != "Queued") / 1e6)
summary = [dict(asof=now, mbs=mbs, fetched_tb=fetched_tb, blank_tb=blank["blank_tb"], fetched_pct=round(100 * fetched_tb / blank["blank_tb"], 2),
                files_done=files_done, files_total=blank["blank_files"], on_vista_tb=on_vista_tb, released_tb=released_tb,
                nersc_tib=R.get("scratch_tib"), nersc_quota_tib=R.get("quota_tib", 20.0), vista_scratch_tb=round(vista_scratch_gb / 1000, 1) if vista_scratch_gb else None,
                vista_su=su, eta_days=eta_days, parts_downloading=sum(1 for p in parts if p["state"] == "downloading"),
                parts_waiting=sum(1 for p in parts if p["state"] in ("waiting for space", "waiting")),
                copies_active=sum(1 for t in tasks if t["state"] == "copying"), copies_queued=sum(1 for t in tasks if t["state"] == "queued"),
                a15_done=a15_done, a15_total=a15_total, blank_exposures=blank["blank_exposures"], globus_mbs=globus_mbs, globus_tb=round(gl_tb, 2))]
holdings = [
    dict(item="Legacy Surveys mirror", where="NERSC community file system (DESI), read in place", exposures=321619, tb=138.9,
         status="permanent", note="41 % of DECam's exposures; images only counted"),
    dict(item="First 5 % images", where=f"NERSC scratch to Vista ({a5_done} of {len(a5)} copies released)", exposures=26313, tb=8.43,
         status="moving" if a5_done < len(a5) else "on Vista", note="searched 11-14; detection lists stay at NERSC"),
    dict(item="Second 5 % downloads", where=f"NERSC scratch to Vista ({a10_done} of 4 folders released)", exposures=17404, tb=9.4,
         status="moving" if a10_done < 4 else "on Vista", note="searched 11-16; 13,665 files the third 5 % links to stay at NERSC"),
    dict(item="Third 5 % gaps", where="NOIRLab to NERSC, kept there for its search", exposures=10181, tb=blank["first_tb"],
         status=f"{a15_done:,} of {a15_total:,} files", note="downloaded first in every part"),
    dict(item="Rest of the gap", where="NOIRLab to NERSC, then Globus to Vista", exposures=blank["blank_exposures"] - 10181,
         tb=round(blank["blank_tb"] - blank["first_tb"], 1), status=f"{fetched_tb:.2f} TB fetched in all", note="NERSC copy deleted once on Vista"),
]
l5_files = sum(p["files"] for p in lparts); l5_tb = round(sum(p["tb"] for p in lparts), 2)
l5_det = sum(d["detected"] for d in detn if d["set"] == "ldt5"); l5_img = sum(d["exposures"] for d in detn if d["set"] == "ldt5")
a15_det = sum(d["detected"] for d in detn if d["set"] == "archive15"); a15_img = sum(d["exposures"] for d in detn if d["set"] == "archive15")
if lparts:
    holdings.insert(4, dict(item="Fourth 5 %, LDT-visible", where="NERSC: 15,034 exposures in the Legacy Surveys mirror, 8,385 from NOIRLab",
                            exposures=23419, tb=13.67, status=f"{l5_files:,} of {ltot.get('files', 25155):,} files downloaded, {l5_det:,} of 23,419 exposures detected",
                            note="downloads first; detection on shared GPU nodes"))
for i, h in enumerate(holdings): h["order"] = i
# history
open(hp, "a").write(json.dumps(dict(t=now, mbs=mbs, gl_mbs=globus_mbs, gl_tb=round(gl_tb, 4), nersc_tib=R.get("scratch_tib"), fetched_tb=fetched_tb, on_vista_tb=on_vista_tb, released_tb=released_tb)) + "\n")
history = [json.loads(l) for l in open(hp) if l.strip()][-500:]
# events
ev = []
for name, f in (("relay", os.path.join(F, "relay", "relay.log")), ("release", os.path.join(F, "release", "release.log"))):
    if os.path.exists(f):
        for l in open(f).read().splitlines()[-60:]:
            if len(l) > 15 and l[2] == "-": ev.append(dict(t="2026-" + l[:11].replace(" ", " "), source=name, text=l[15:].strip()[:220]))
ep = os.path.join(D, "events.jsonl")
if os.path.exists(ep):
    ev += [json.loads(l) for l in open(ep) if l.strip()]
a5e = os.path.join(P, "archive5", "events.jsonl")          # the first 5 % search's log (its old dashboard is merged here)
if os.path.exists(a5e):
    ev += [dict(t=e.get("t") or e.get("time"), source="first 5 %", text=(e.get("text") or e.get("event") or "")[:220]) for e in (json.loads(l) for l in open(a5e) if l.strip())]
ev.sort(key=lambda e: e["t"], reverse=True)
jobs = [dict(job=j["job"], name=j["name"], queue=j["qos"].replace("regular_1", "regular").replace("regular_0", "regular"), state=j["state"], time=j["time"], where=j["where"]) for j in jobs_n]
searches = json.load(open(os.path.join(D, "searches.json")))   # hand-kept: one row per 5 % search
for r in searches:                                               # live progress for the searches still waiting for data
    if r["search"].startswith("fourth") and r["state"] != "searched" and lparts:   # (searched 11-19)
        r["note"] = (f"chosen for LDT nights Nov 20 - Jan 31; downloads {l5_tb:.2f} of {ltot.get('tb', 4.94):.2f} TB "
                     f"({l5_files:,} of {ltot.get('files', 25155):,} files); detection {l5_det:,} of 23,419 exposures")
    if r["search"].startswith("third") and r["state"] != "searched" and a15_img:
        r["note"] = f"detection {a15_det:,} of {a15_img:,} exposures on NERSC; its 10,181 missing exposures come in the fill"
S = dict(summary=summary, holdings=holdings, parts=parts + lparts, transfers=tasks, jobs=jobs, history=history, events=ev[:120], searches=searches)
try:                                                  # per-device downloads and transfers from fill/dl_watch.json (2026-11-20)
    import importlib.util
    _sp = importlib.util.spec_from_file_location("dl_rows", os.path.join(D, "dl_rows.py")); _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
    S.update(_m.rows())
except Exception as e:
    print("dl_rows failed:", type(e).__name__, str(e)[:200])
json.dump(S, open(os.path.join(OUT, "status.json.tmp"), "w")); os.replace(os.path.join(OUT, "status.json.tmp"), os.path.join(OUT, "status.json"))   # atomic: the page updater uploads it
print(json.dumps(summary[0]))
print(len(tasks), "transfers,", len(jobs), "jobs,", len(ev), "events,", len(history), "history rows")
