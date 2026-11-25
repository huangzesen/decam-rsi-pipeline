#!/usr/bin/env python3
"""Files that reached Vista's DECam cache without the NERSC relay (2026-11-20): Vista's own download (share 2,
blank_vista2.tsv, jobs dfill2_*; done logs decam-fill/log/done_*of4.txt there) the Lonestar6 -> Vista relay
(fill/relay_ls6/relayed.txt). The Mac's files go through NERSC, so relay.py lists them itself. Each cycle appends the names not yet listed to fill/relay/relayed.txt, which the conveyor's
unit selection and preparation read as "on Vista" (pulled to NERSC with Globus instead of downloaded again). Append
only; relay.py appends to the same file and these names are on NERSC's skip list, so the two never overlap.
Usage: python3 -I fill/vista_known.py once | loop [MINUTES=30]   (stop: touch fill/relay/STOP_VISTA_KNOWN)"""
import os, subprocess, sys, time
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "relay", "relayed.txt")
SSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/vista-$TACC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "x"]


def cycle():
    have = set(l.strip() for l in open(R))
    p = subprocess.run(SSH + ["LC_ALL=C; cat $TACC_SCRATCH/decam-fill/log/done_*of4.txt 2>/dev/null; true"], capture_output=True, text=True, timeout=600)
    vista = [l.strip() for l in p.stdout.splitlines() if l.strip().startswith("c4d_")] if p.returncode == 0 else []
    ls6 = [l.strip() for l in open(os.path.join(H, "relay_ls6", "relayed.txt")) if l.strip()]
    new = [n for n in dict.fromkeys(vista + ls6) if n not in have]
    if new:
        open(R, "a").write("".join(n + "\n" for n in new))
    print(time.strftime("%m-%d %H:%M:%S"), f"Vista download {len(vista)}, Lonestar6 relay {len(ls6)}, added {len(new)}", "" if p.returncode == 0 else "(Vista not reachable)", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["once"]:
        cycle()
    else:
        m = float(sys.argv[2]) if len(sys.argv) > 2 else 30
        while not os.path.exists(os.path.join(H, "relay", "STOP_VISTA_KNOWN")):
            try:
                cycle()
            except Exception as e:
                print("cycle error:", type(e).__name__, str(e)[:200], flush=True)
            time.sleep(m * 60)
