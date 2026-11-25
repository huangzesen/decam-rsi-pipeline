#!/usr/bin/env python3
"""OPTIONAL, not part of the port flow (only before `tailor.py grade`, on approval). Put the four graded regions of the r2 climb (detection files only; truth never leaves the Mac) on a tailoring
target's scratch, from NERSC $SCRATCH/dbt/scen with Globus (tailor/, 2026-11-18). Dry by default: lists what each
target lacks (read-only ls over the SSH master) and prints the one Globus command per target. `--go` submits them; use it
only on approval.

    python3 -I tailor/stage.py                 # every enabled TACC target
    python3 -I tailor/stage.py --target gb200-vista [--go]

Sizes on NERSC (2026-11-18): scen13 2.4 GB, scen10 2.9 GB, scen08 3.2 GB, scen14 2.5 GB (11 GB).
Vista has scen13 and scen10; Lonestar6 has none. Vista and Lonestar6 scratch purge files unread for 10 days, so stage
shortly before a run. Globus maps TACC logins by the ePPN in the TACC Accounts Portal (EPPN@access-ci.org, accepted for
Vista on 2026-11-17); the Lonestar6 collection has not been used yet."""
import argparse, json, os, subprocess, sys, tempfile
T = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(T)
NERSC = "$GLOBUS_NERSC_COLLECTION"            # NERSC Perlmutter
COLL = {"vista": "$GLOBUS_VISTA_COLLECTION",   # TACC Vista (used 2026-11-17)
        "ls6": "$GLOBUS_LS6_COLLECTION"}     # TACC Lonestar6 GCS v5.4 Filesystems (globus endpoint search, 2026-11-18)
SRC = "$NERSC_SCRATCH/dbt/scen"
ENV = dict(os.environ, LC_ALL="C", LANG="C")


def remote_has(t, sids):
    p = subprocess.run(["ssh", "-S", os.path.expanduser(t["sock"]), "-o", "ControlMaster=no", "-o", "BatchMode=yes", t["host"],
                        "export LC_ALL=C; for s in " + " ".join(sids) + f"; do test -d {t['scen']}/$s && echo $s; done"],
                       capture_output=True, text=True, timeout=300)
    if p.returncode == 255:
        sys.exit(f"{t['machine']}: no SSH master (the user logs in first)")
    return set(p.stdout.split())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target"); ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    targets = json.load(open(os.path.join(T, "targets.json")))
    sids = json.load(open(os.path.join(HERE, "runs/r2/config.json")))["sids"]
    keys = [a.target] if a.target else [k for k, v in targets.items() if not k.startswith("_") and v.get("enabled") and v.get("site") == "tacc"]
    done_machines = set()
    for k in keys:
        t = targets[k]
        if t.get("site") != "tacc":
            print(f"{k}: site {t.get('site')}: nothing to stage (NERSC holds the regions)"); continue
        if t["machine"] in done_machines:
            print(f"{k}: same scratch as an earlier target on {t['machine']}"); continue
        done_machines.add(t["machine"])
        have = remote_has(t, sids); need = [s for s in sids if s not in have]
        print(f"{k} ({t['machine']}): has {sorted(have) or 'none'}; needs {need or 'nothing'}")
        if not need:
            continue
        batch = "".join(f"{s}/ {s}/ --recursive\n" for s in need)        # relative to the two base folders below
        cmd = ["globus", "transfer", f"{NERSC}:{SRC}/", f"{COLL[t['machine']]}:{t['scen']}/", "--batch", "-", "--sync-level", "size", "--verify-checksum",
               "--preserve-timestamp", "--label", f"dbt tailor regions to {t['machine']}", "--notify", "failed"]
        print("  globus batch:\n" + "".join("    " + l for l in batch.splitlines(True)) + "  command: " + " ".join(cmd[:4]) + " --batch <the lines above> " + " ".join(cmd[6:]))
        if a.go:
            p = subprocess.run(cmd, input=batch, capture_output=True, text=True, env=ENV)
            print("  submitted:" if p.returncode == 0 else "  FAILED:", (p.stdout + p.stderr).strip()[-300:])
    if not a.go:
        print("dry run: nothing submitted (add --go on approval)")


if __name__ == "__main__":
    main()
