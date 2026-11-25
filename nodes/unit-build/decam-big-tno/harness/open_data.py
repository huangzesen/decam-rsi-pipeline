#!/usr/bin/env python3
"""Design 2 data for the agent: whole regions with their known bodies labelled, on the agent host (CPU, no allocation).
Since 2026-11-14 16:30 the four graded regions are here too, with their labels; regions.json marks them "graded". Nothing is dropped from the labels any more. Earlier text below describes the
first form (graded regions kept off the host). 2026-11-14;
labelled open data approved the same day ("Yes let's prep this").

Open regions = regions of the 5 % search that are not graded in run r2, are of normal size, and share no exposure and no
known body with the four graded regions (R021 shares one body: its label is dropped). The four graded regions never go
to the host. Layout on the host, mounted read-only at /app/open in the agent's container:
    OPEN/scen/<sid>/{det_*.npy, exposures.json, region.json}     the same files a graded region has
    OPEN/labels/<sid>.npz                                        det_body, det_night, known [{name, H, weight, nights}], planted []
The labels are the grader's truth files as they are (names and H included since 2026-11-14 16:30), so tools/score.py (= harness/grade.py) scores a program's output on any region exactly as the
grader does.

  open_data.py list              the open regions, sizes, findable bodies
  open_data.py copy [N=2]        stream every missing region NERSC scratch -> host through this Mac (nothing stored here)
  open_data.py labels            write the labels (scratch folder on the Mac) and copy them to the host
  open_data.py verify            compare file sizes on NERSC and on the host
"""
import json, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN_RIDS = ["REG_C", "R006", "R008", "R011", "R018", "R021", "R023", "R026", "R027", "R030", "R031", "R032", "R033", "R034",
             "R036", "R037", "R038", "R039", "R040"]
GRADED_TRUTH = os.path.join(HERE, "private", "truth_r2x")          # H <= 8 and the fainter known bodies (relabel_faint.py)
SCEN = "$NERSC_SCRATCH/dbt/scen"
OPEN = "$VM_WORK/tasks-ale/decam-big-tno-open"
NSSH = ("ssh -S $SSH_CM_DIR/nersc-$NERSC_USER.sock -o ControlMaster=no -o BatchMode=yes $NERSC_USER@perlmutter.nersc.gov")
HSSH = "ssh -o BatchMode=yes -o ConnectTimeout=20 $VM_HOST"
LOG = os.path.join(HERE, "runs", "r2", "open_data.log")


GRADED = json.load(open(os.path.join(HERE, "runs", "r2", "config.json")))["regions"]      # sid -> rid, the graded four


def regions():
    reg = {r["rid"]: r["sid"] for r in json.load(open(os.path.join(HERE, "harness", "search38", "regions38.json")))}
    reg["REG_C"] = "scen07"
    out = [(rid, sid, os.path.join(GRADED_TRUTH, sid + ".npz")) for sid, rid in GRADED.items()]
    for rid in OPEN_RIDS:
        sid = reg[rid]
        t = os.path.join(HERE, "private", "truth_openx", sid + ".npz")   # truth / truth38 plus the fainter known bodies
        out.append((rid, sid, t))
    return out


def graded_names():
    return set()                                              # nothing is hidden any more: no label is dropped


def log(msg):
    line = time.strftime("%F %T ") + msg
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def sh(cmd, **kw):
    return subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], capture_output=True, text=True, **kw)


def sizes(side, sid):
    cmd = (f"{NSSH} 'cd {SCEN}/{sid} && stat -c \"%n %s\" *'" if side == "nersc"
           else f"{HSSH} 'cd {OPEN}/scen/{sid} 2>/dev/null && stat -c \"%n %s\" *'")
    p = sh(cmd, timeout=120)
    return dict(l.split() for l in p.stdout.splitlines() if len(l.split()) == 2)


def copy_one(item):
    rid, sid, _ = item
    a = sizes("nersc", sid)
    if not a:
        log(f"{rid} {sid}: no files on NERSC, skipped"); return
    if sizes("host", sid) == a:
        log(f"{rid} {sid}: already on the host"); return
    t0 = time.time()
    p = sh(f"{NSSH} 'tar cf - -C {SCEN} {sid}' | {HSSH} 'mkdir -p {OPEN}/scen && tar xf - -C {OPEN}/scen'", timeout=6 * 3600)
    b = sizes("host", sid); gb = sum(int(v) for v in a.values()) / 1e9; dt = time.time() - t0
    ok = p.returncode == 0 and a == b
    log(f"{rid} {sid}: {'copied' if ok else 'FAILED'} {gb:.2f} GB in {dt:.0f} s ({gb * 1000 / max(dt, 1):.0f} MB/s)"
        + ("" if ok else " " + p.stderr[-300:].replace("\n", " ")))


cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
R = regions()
if cmd == "list":
    G = graded_names(); tot = [0, 0]
    for rid, sid, t in R:
        z = np.load(t); K = json.loads(str(z["known"]))
        fk = [k for k in K if k["nights"] >= 5]; drop = [k for k in fk if k["name"] in G]
        tot[0] += len(z["det_body"]); tot[1] += len(fk) - len(drop)
        print(f"{rid} {sid} {len(z['det_body']) / 1e6:7.1f} M det, findable {len(fk)}" + (", graded" if sid in GRADED else ""))
    print(f"total {tot[0] / 1e6:.0f} M detections, {tot[1]} labelled findable bodies, {len(R)} regions")
elif cmd == "copy":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    items = [r for r in R if not only or r[0] in only]
    log(f"copy of {len(items)} regions, {n} streams")
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(copy_one, items))
    log("copy done")
elif cmd == "labels":
    G = graded_names(); tmp = tempfile.mkdtemp(prefix="dbt-open-labels-"); card = {}
    for rid, sid, t in R:
        z = np.load(t); K = json.loads(str(z["known"])); body = z["det_body"].copy(); night = z["det_night"]
        assert not json.loads(str(z["planted"])) and body.max() < 100000
        known = [dict(name=k["name"], H=k["H"], weight=k["weight"], nights=k["nights"]) for k in K]   # everything we know
        np.savez_compressed(os.path.join(tmp, sid + ".npz"), det_body=body, det_night=night, known=json.dumps(known),
                            planted="[]")
        card[sid] = dict(graded=sid in GRADED, detections=int(len(body)), findable=sum(1 for k in known if k["nights"] >= 5),
                         weight=sum(k["weight"] for k in known if k["nights"] >= 5))
    json.dump(card, open(os.path.join(tmp, "regions.json"), "w"), indent=1)
    p = sh(f"tar cf - -C {tmp} . | {HSSH} 'mkdir -p {OPEN}/labels && tar xf - -C {OPEN}/labels && ls {OPEN}/labels | wc -l'",
           timeout=3600)
    log(f"labels: {len(card)} regions, {sum(c['findable'] for c in card.values())} findable bodies, host files "
        f"{p.stdout.strip()} {p.stderr[-200:]}")
    subprocess.run(["rm", "-rf", tmp])
elif cmd == "verify":
    bad = 0
    for rid, sid, _ in R:
        a, b = sizes("nersc", sid), sizes("host", sid)
        ok = bool(a) and a == b; bad += not ok
        print(rid, sid, "ok" if ok else "MISSING or different", f"{sum(int(v) for v in a.values()) / 1e9:.2f} GB")
    print("all regions on the host" if not bad else f"{bad} regions not complete")
