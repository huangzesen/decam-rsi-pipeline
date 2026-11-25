#!/usr/bin/env python3
"""DECam fill, Mac share -> NERSC -> Vista (2026-11-20).

mac_fill.sh downloads the Mac's share (blank_mac.tsv: 4 TB carved from NERSC's remaining list, now in NERSC's
blank.skip.txt) with fetch.py into the Crucial-8TB buffer $DATA_DRIVE/decam-fill-mac/img/<YYMMDD>/.
One cycle: files fetch.py finished (log/done_*.txt, md5-checked against the archive) that are still in the buffer go to
NERSC $NERSC_SCRATCH/decam-fill/pmac/img/<YYMMDD>/ with rsync over the NERSC ssh master (one file measured at
47 MB/s; the first version pushed straight to Vista over the Vista master at only ~3 MB/s for 4 lanes, backup
relay_mac.py.bak-1120-vista); rsync checks each file after transfer and --remove-source-files deletes the buffer copy only
after it arrived. Then the names are appended to pmac/log/done_mac.txt there, so fill/relay.py (which walks every
decam-fill/p*/ part) carries them to Vista with Globus and deletes them at NERSC, and lists them in relay/relayed.txt.
Names that reached NERSC go to relay_mac/relayed.txt here.
Usage: python3 -I fill/relay_mac.py once | loop [MINUTES=5]   (stop: touch fill/relay_mac/STOP)"""
import os, subprocess, sys, tempfile, time
H = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(H, "relay_mac"); os.makedirs(S, exist_ok=True)
B = "$DATA_DRIVE/decam-fill-mac"
PM = "$NERSC_SCRATCH/decam-fill/pmac"
DST = PM + "/img/"
SOCK = os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock")
RSH = f"ssh -S {SOCK} -o ControlMaster=no -o BatchMode=yes"
LANES = 2
# per lane, KiB/s: 2 x 42,000 = ~86 MB/s up. Uploading at the full 1 Gb/s line rate (106 MB/s, 18:17-19:15) halved the
# NOIRLab download on the same link (100 -> 40-65 MB/s): the download's ACKs queue behind the upload (2026-11-20)
BWLIMIT_KIB = int(os.environ.get("RELAY_MAC_BWLIMIT", "42000"))
MAX_FILES = int(os.environ.get("RELAY_MAC_MAX", "400"))     # ~80 GB per cycle


def log(*x):
    line = time.strftime("%m-%d %H:%M:%S ") + " ".join(map(str, x)); print(line, flush=True)
    open(os.path.join(S, "relay.log"), "a").write(line + "\n")


def cycle():
    if not os.path.exists(SOCK):
        log("NERSC ssh master is gone; waiting (the user logs in)"); return
    relayed_p = os.path.join(S, "relayed.txt")
    pend_p = os.path.join(S, "pending.txt")
    if os.path.exists(pend_p) and os.path.getsize(pend_p):     # at NERSC but not yet in done_mac.txt
        pend = open(pend_p).read()
        r = subprocess.run(RSH.split() + ["$NERSC_USER@perlmutter.nersc.gov", f"mkdir -p {PM}/log && cat >> {PM}/log/done_mac.txt"],
                           input=pend, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            open(relayed_p, "a").write(pend); os.remove(pend_p)
    relayed = set(l.strip() for l in open(relayed_p)) if os.path.exists(relayed_p) else set()
    done = []
    for f in sorted(os.listdir(os.path.join(B, "log"))):
        if f.startswith("done_") and f.endswith(".txt"):
            done += [l.strip() for l in open(os.path.join(B, "log", f)) if l.strip().startswith("c4d_")]
    new = [n for n in dict.fromkeys(done) if n not in relayed and os.path.exists(os.path.join(B, "img", n[4:10], n))][:MAX_FILES]
    if not new:
        log(f"no new files ({len(done)} done on the Mac, {len(relayed)} relayed)"); return
    t0 = time.time(); size = sum(os.path.getsize(os.path.join(B, "img", n[4:10], n)) for n in new)
    lanes = [new[i::LANES] for i in range(LANES)]
    procs = []
    for k, part in enumerate(lanes):
        if not part:
            continue
        lf = tempfile.NamedTemporaryFile("w", delete=False, suffix=f".lane{k}.txt")
        lf.write("".join(f"{n[4:10]}/{n}\n" for n in part)); lf.close()
        p = subprocess.Popen(["/opt/homebrew/bin/rsync", "-a", "--partial-dir=.rsync-partial", "--remove-source-files",
                              f"--rsync-path=mkdir -p {DST} && rsync", f"--bwlimit={BWLIMIT_KIB}",
                              f"--files-from={lf.name}", "-e", RSH, os.path.join(B, "img") + "/", f"$NERSC_USER@perlmutter.nersc.gov:{DST}"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append((p, part, lf.name))
    ok = []
    for p, part, lfn in procs:
        out, err = p.communicate()
        os.remove(lfn)
        arrived = [n for n in part if not os.path.exists(os.path.join(B, "img", n[4:10], n))]   # removed only after transfer
        ok += arrived
        if p.returncode != 0:
            log(f"rsync lane exit {p.returncode}: {err.strip()[-300:]}")
    if ok:
        r = subprocess.run(RSH.split() + ["$NERSC_USER@perlmutter.nersc.gov", f"mkdir -p {PM}/log && cat >> {PM}/log/done_mac.txt"],
                           input="".join(n + "\n" for n in ok), capture_output=True, text=True, timeout=600)
        if r.returncode != 0:                      # the files are at NERSC already; list them there next cycle
            open(os.path.join(S, "pending.txt"), "a").write("".join(n + "\n" for n in ok))
            log("could not append to NERSC done_mac.txt (kept in pending.txt):", r.stderr[-200:]); return
        open(relayed_p, "a").write("".join(n + "\n" for n in ok))
    dt = time.time() - t0
    log(f"{len(ok)} of {len(new)} files at NERSC, {size / 1e9:.1f} GB in {dt:.0f} s ({size / 1e6 / max(dt, 1):.0f} MB/s)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["once"]:
        cycle()
    else:
        m = float(sys.argv[2]) if len(sys.argv) > 2 else 5
        while not os.path.exists(os.path.join(S, "STOP")):
            try:
                cycle()
            except Exception as e:
                log("cycle error:", type(e).__name__, str(e)[:300])
            time.sleep(m * 60)
        log("STOP")
