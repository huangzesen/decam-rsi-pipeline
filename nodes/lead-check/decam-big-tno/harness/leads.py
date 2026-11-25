#!/usr/bin/env python3
"""Send a graded round's leads to the novelty check (planet-search/rsi/candidates/novelty_queue/inbox/).

A lead (harness/grade.py) is a candidate inside the global top K whose detections are >= 80 % unlabelled and cover >= 5
nights. Its places come from the sealed scenario on Vista (row ids -> ra, dec, exposure mid-time, S/N), read on the login
node; nothing else leaves Vista. Every lead of the round (--max, default 200), best rank first, skipping any that shares
half its detections with a lead sent before (ledger runs/<run>/leads_sent.json); such a re-find is recorded in that
entry's "seen" list. The queue's verdicts land in its RESULTS.md. Then harness/lead_check.py RUN GEN starts detached
(orbit, size, in-data repeat test; 2026-11-13).
Usage: python3 -I harness/leads.py RUN GEN [--max 200] [--dry] [--no-check]
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INBOX = os.path.expanduser("$ALE_HOME/planet-search/rsi/candidates/novelty_queue/inbox")
VISTA = "$TACC_USER@vista.tacc.utexas.edu"
VSSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/vista-$TACC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", VISTA]
VPY = "$TACC_WORK/vista/dbt/env/bin/python"
SCEN = "$TACC_SCRATCH/dbt/scen"
PRE = "module load gcc/13.2.0 python3/3.11.8 >/dev/null 2>&1; "
try:                                            # a design 2 run (all-open rules): its regions are on NERSC scratch, not on Vista
    if len(sys.argv) > 1 and json.load(open(os.path.join(HERE, "runs", sys.argv[1], "config.json"))).get("design") == 2:
        os.environ.setdefault("DBT_SITE", "nersc")     # lead_check.py, started below, inherits it
except (OSError, ValueError):
    pass
if os.environ.get("DBT_SITE") == "nersc":      # search38: the 5 % search regions' scenarios are on NERSC scratch
    VSSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes",
            "$NERSC_USER@perlmutter.nersc.gov"]
    VPY, SCEN, PRE = "$NERSC_SCRATCH/dbt/env/bin/python", "$NERSC_SCRATCH/dbt/scen", ""
REMOTE = r'''
import json, sys, numpy as np
S = "%s"; req = json.load(sys.stdin); out = {}
for sid, rows in req.items():
    L = lambda k: np.load(f"{S}/{sid}/det_{k}.npy", mmap_mode="r")
    E = json.load(open(f"{S}/{sid}/exposures.json"))
    ra, dec, snr, ex = L("ra"), L("dec"), L("snr"), L("exp")
    out[sid] = {str(r): [E[int(ex[r])]["mjd_mid"], float(ra[r]), float(dec[r]), float(snr[r])] for r in rows}
json.dump(out, sys.stdout)
'''.replace("%s", SCEN)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("gen")
    ap.add_argument("--max", type=int, default=200); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    RUN = os.path.join(HERE, "runs", a.run)
    g = json.load(open(os.path.join(RUN, "gens", a.gen, "grade", "grade.json")))
    cfg = json.load(open(os.path.join(RUN, "config.json"))); letter = cfg.get("names") or {s: chr(65 + i) for i, s in enumerate(cfg["sids"])}
    ledger_p = os.path.join(RUN, "leads_sent.json")
    ledger = json.load(open(ledger_p)) if os.path.exists(ledger_p) else []
    pick = []
    for l in sorted(g["leads"], key=lambda l: l["rank"]):
        s = set(l["det"])
        old = [x for x in ledger + pick if x["sid"] == l["sid"] and len(s & set(x["det"])) >= 0.5 * min(len(s), len(x["det"]))]
        if old:
            for x in old:
                if x in ledger and x.get("gen") != a.gen and a.gen not in x.setdefault("seen", []):
                    x["seen"].append(a.gen)
            continue
        pick.append(l)
        if len(pick) >= a.max:
            break
    if not pick:
        if not a.dry:
            json.dump(ledger, open(ledger_p + ".tmp", "w")); os.replace(ledger_p + ".tmp", ledger_p)
        print(f"{a.gen}: {len(g['leads'])} leads, none new"); return
    req = {}
    for l in pick:
        req.setdefault(l["sid"], []).extend(l["det"])
    assert "'" not in REMOTE
    p = subprocess.run(VSSH + [f"{PRE}{VPY} -I -c '{REMOTE}'"], input=json.dumps(req), capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        sys.exit(f"reading places failed: {p.stderr[-400:]}")
    P = json.loads(p.stdout)
    os.makedirs(INBOX, exist_ok=True)
    for l in pick:
        pts = sorted(P[l["sid"]][str(r)] for r in l["det"])
        spec = dict(id=f"dbt-{a.run}-{a.gen}-{letter[l['sid']]}-rank{l['rank']}-p{l['p']:.5f}",
                    points=[[m, ra, dec, round(0.08 + 0.6 / max(s, 3.0), 3), "DECam"] for m, ra, dec, s in pts],
                    note=f"decam-big-tno {a.run} {a.gen}: global rank {l['rank']}, {l['nights']} nights, {len(pts)} detections")
        if a.dry:
            print(spec["id"], len(pts), "points", f"{pts[0][0]:.1f}-{pts[-1][0]:.1f}"); continue
        tmp = os.path.join(RUN, spec["id"] + ".json")
        json.dump(spec, open(tmp, "w")); os.replace(tmp, os.path.join(INBOX, spec["id"] + ".json"))
        ledger.append(dict(sid=l["sid"], det=l["det"], gen=a.gen, id=spec["id"]))
    if not a.dry:
        json.dump(ledger, open(ledger_p + ".tmp", "w")); os.replace(ledger_p + ".tmp", ledger_p)
        if not a.no_check:
            os.makedirs(os.path.join(RUN, "lead_check"), exist_ok=True)
            subprocess.Popen(f"nohup {os.path.expanduser('~/anaconda3/bin/python')} -I {os.path.join(HERE, 'harness/lead_check.py')} {a.run} {a.gen} "
                             f"> {os.path.join(RUN, 'lead_check', a.gen + '.log')} 2>&1 &", shell=True, cwd=HERE,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"{a.gen}: {len(g['leads'])} leads, sent {len(pick)} to the novelty check" + ("" if a.dry or a.no_check else "; lead check started"))


if __name__ == "__main__":
    main()
