#!/usr/bin/env python3
"""conveyor (NERSC): free a unit's images as soon as they are detected (2026-11-19). An exposure counts as detected when its list det/<YYMMDD>/<stem>.npz exists or its
detection failed twice (det/detect.log).
  links U          m* shares: for each detected exposure, files linked from the whole-archive fill
                   ($NERSC_SCRATCH/decam-fill/..., already copied to Vista by fill/relay.py) are deleted with
                   their links; links into the Legacy Surveys mirror (/global/cfs) are only unlinked. Prints "freed N GB".
  list U BATCH [GB] p* shares (at most about GB per batch) (NOIRLab downloads of this unit, not yet on Vista): writes U/cache/BATCH.txt with the files
                   (paths relative to U) of detected exposures not in an earlier batch; prints "N files X GB"
  delete U BATCH   after the Globus copy of BATCH succeeded: deletes those files, renames the list to BATCH.done
  v U              v share (pulled back from the Vista cache): deletes the detected exposures' files
Usage: python3 -I unit_release.py links|list|delete|v U [BATCH]"""
import glob, os, sys
FILL = "$NERSC_SCRATCH/decam-fill/"
mode, U = sys.argv[1], os.path.abspath(sys.argv[2])


def detected(D):
    done = {os.path.basename(p)[:17] for p in glob.glob(f"{D}/det/*/c4d_*.npz") if ".tmp." not in p}
    errs = {}
    lg = f"{D}/det/detect.log"
    if os.path.exists(lg):
        for l in open(lg):
            if " ERROR " in l and len(l.split()) > 2:
                errs[l.split()[2]] = errs.get(l.split()[2], 0) + 1
    return done | {s for s, n in errs.items() if n >= 2}


if mode == "links":
    freed = 0; n = 0
    for D in sorted(glob.glob(f"{U}/m[0-9]*")):
        det = detected(D)
        for p in glob.glob(f"{D}/img/*/c4d_*"):
            if os.path.basename(p)[:17] not in det or not os.path.islink(p):
                continue
            t = os.readlink(p)
            if t.startswith(FILL) and os.path.exists(t):
                freed += os.path.getsize(t); os.remove(t); n += 1
            os.remove(p)
    print(f"freed {freed / 1e9:.1f} GB ({n} fill files)")
elif mode == "list":
    B = sys.argv[3]; os.makedirs(f"{U}/cache", exist_ok=True)
    cap = float(sys.argv[4]) * 1e9 if len(sys.argv) > 4 else float("inf")     # batch size limit in GB
    seen = set()
    for f in glob.glob(f"{U}/cache/*.txt") + glob.glob(f"{U}/cache/*.done"):
        seen.update(l.strip() for l in open(f))
    out, size = [], 0
    for D in sorted(glob.glob(f"{U}/p[0-9]*")):
        det = detected(D)
        for p in glob.glob(f"{D}/img/*/c4d_*.fits.fz"):
            r = os.path.relpath(p, U)
            if os.path.basename(p)[:17] in det and r not in seen and size < cap:
                out.append(r); size += os.path.getsize(p)
    if out:
        open(f"{U}/cache/{B}.txt", "w").write("".join(r + "\n" for r in out))
    print(f"{len(out)} files {size / 1e9:.1f} GB")
elif mode == "delete":
    B = sys.argv[3]; L = f"{U}/cache/{B}.txt"; n = 0; size = 0
    for r in open(L):
        p = os.path.join(U, r.strip())
        if r.strip().startswith("p") and "/img/" in r and os.path.exists(p):
            size += os.path.getsize(p); os.remove(p); n += 1
    os.rename(L, f"{U}/cache/{B}.done")
    print(f"deleted {n} files {size / 1e9:.1f} GB")
elif mode == "v":
    D = f"{U}/v"; det = detected(D) if os.path.isdir(D) else set(); n = 0; size = 0
    for p in glob.glob(f"{D}/img/*/c4d_*.fits.fz"):
        if os.path.basename(p)[:17] in det:
            size += os.path.getsize(p); os.remove(p); n += 1
    print(f"deleted {n} files {size / 1e9:.1f} GB")
