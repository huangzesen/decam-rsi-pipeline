#!/usr/bin/env python3
"""Store documents for "DECam Data Movement" (<private-page-url>): datasets pointing at the
uploaded out/status.json and out/atlas.json, plus (--page) dash/meta and files/index.html.
Usage: python3 -I fill/dash/docs.py STATUS_BLOB_URL [ATLAS_BLOB_URL] [--page]  (atlas datasets are written only when ATLAS_BLOB_URL is given)"""
import json, os, sys
from datetime import datetime, timezone
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out", "docs"); os.makedirs(OUT, exist_ok=True)
args = [a for a in sys.argv[1:] if not a.startswith("--")]; url = args[0]; aurl = args[1] if len(args) > 1 else None; now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
D = {
    "summary": ("Totals now", "Download speed, TB fetched of the gap, TB relayed to Vista and released at NERSC, both scratch spaces, time left."),
    "holdings": ("Where the data is", "Each DECam holding (NERSC mirror, first, second and third 5 %, the rest of the gap): place, exposures, TB, status."),
    "parts": ("Download parts", "The six NERSC download jobs: where each runs, its state, speed now, TB and files done, damaged copies and errors."),
    "transfers": ("Globus copies", "Every NERSC-to-Vista Globus copy of ours since 2026-11-17: purpose, state, TB and files copied, speed."),
    "jobs": ("NERSC jobs", "Our Slurm jobs on Perlmutter at the time of the update: download parts and the third 5 % detection."),
    "history": ("Speed and scratch over time", "One row per status update: total download MB/s, NERSC scratch TiB, TB fetched, relayed and released."),
    "events": ("Log", "Relay and release log lines, hand-written milestones and the first 5 % search's log, newest first."),
    "searches": ("The 5 % searches", "One row per search of a 5 % part of DECam's sky: regions, exposures, program, recall, reward, leads, GPU-hours, state."),
    "dl_hist": ("Speed per device over time", "One row per 10 minutes: NOIRLab download MB/s on NERSC, Lonestar6, Vista and the Mac, and copy MB/s on each route between devices."),
    "dl_now": ("Downloads by device now", "Latest 10 minutes per device: state, MB/s, files done, share of files with a bad md5, refused or failed connections."),
    "dl_xfer": ("Transfers between devices now", "Latest copy speed on each route between devices, and the last speed seen while a copy ran."),
    "dl_streams": ("Stream budgets", "Each time the download controller limited a device: reason, streams before and after, and when its 30-minute cool-down ends."),
    "dl_probes": ("Download probes", "Short test downloads (60 s, 16 streams) from the idle devices after a device slows down: speed, bad md5, errors."),
}
A = {
    "slices": ("Archive slices", "Every planned slice of the usable public archive (2026-11-12): survey class, program, windows, exposures, GB, A100-hours, dates, mean position."),
    "slice_classes": ("Survey classes", "Slices, windows, exposures, TB and A100-hours per cadence and sky class."),
    "measured": ("Measured speeds", "Download and copy rates measured on each route since 2026-11-10."),
    "regions": ("Search regions", "The 3-degree regions of the first, second and third 5 % searches: place, area, exposures, TB, years."),
}
for k, t, d, u, n in [(k, *v, url, "status.json") for k, v in D.items()] + ([(k, *v, aurl, "atlas.json") for k, v in A.items()] if aurl else []):
    json.dump({"title": t, "description": d, "source": {"kind": "file", "url": u, "name": n, "format": "json"},
               "updated": {"at": now, "by": "Claude (fill/dash/build.py)"}, "staleAfterHours": 6,
               "schedule": "Rebuilt by Claude from NERSC, Vista and Globus status when it checks the download."}, open(os.path.join(OUT, f"ds_{k}.json"), "w"))
if "--page" in sys.argv:
    json.dump({"title": "DECam Data Movement", "lang": "en"}, open(os.path.join(OUT, "meta.json"), "w"))
    json.dump({"text": open(os.path.join(H, "index.html")).read()}, open(os.path.join(OUT, "index.json"), "w"))
print(sorted(os.listdir(OUT)))
