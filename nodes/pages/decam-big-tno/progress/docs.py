#!/usr/bin/env python3
"""Store documents for the "Big TNO Hill Climb" artifact (<private-page-url>), written to
progress/out/docs/ for one ArtifactData batch. Usage: python3 -I progress/docs.py BLOB_URL [--page]
BLOB_URL = /_blob/<id> of the uploaded progress/out/score.json. --page also writes dash/meta and files/index.html."""
import json, os, sys
from datetime import datetime, timezone
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out", "docs"); os.makedirs(OUT, exist_ok=True)
url = sys.argv[1]; now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
D = {
    "summary": ("Run totals", "Best score, known bodies found, rounds graded and candidate counts for the hill climb."),
    "rounds": ("Rounds", "Each round of the hill climb: state, score on NERSC A100s (and the first Vista GB200 grade), known bodies found per region, leads and the agent's method."),
    "regions": ("Known bodies per region", "Known large bodies found in each sealed region by each round graded on NERSC."),
    "funnel": ("Candidate checks", "How many leads pass each check, from the scored list to follow-up by OBSERVER_NAME; counts only."),
    "events": ("Run log", "Rounds started, written, graded and new bests, newest first."),
    "data_regions": ("Sealed regions", "Per sealed region: area, years, exposures, detections, size of the detection lists and of the images behind them, known large bodies to find."),
    "data_years": ("Exposures by year", "Exposures, nights and median 5-sigma depth per region and year."),
    "run1_summary": ("Run 1 totals", "Run 1 (the task's current form): best recall and its GPU-hours, known bodies found, the starting program's numbers, rounds graded."),
    "run1_rounds": ("Run 1 rounds", "Each program of Run 1 (one per round in rounds 1 and 2; rounds 3-20 two per round, one by Claude Fable 5.2 and one by GPT 6 Astra; from round 21 one per round by GPT 6.2 Astra at effort max): state, recall, GPU-hours, known bodies found per graded region, bodies gained and lost against the best before its round, leads, the agent's method and what reading the program showed."),
    "run1_regions": ("Run 1: known bodies per region", "Known large bodies found in each of the four graded regions by each program of Run 1, with the run time and which agent wrote it."),
    "run1_events": ("Run 1 log", "Run 1 rounds started, written, graded and new bests, newest first."),
    "candidates": ("Candidates", "One row per candidate body (re-finds in later rounds merged): its checks, orbit, size and image-check counts; no sky positions, dates or images."),
}
for k, (t, d) in D.items():
    json.dump({"title": t, "description": d, "source": {"kind": "file", "url": url, "name": "score.json", "format": "json"},
               "updated": {"at": now, "by": "<coordinator-session> session (Claude)"}, "staleAfterHours": 12,
               "schedule": "Rebuilt by the <coordinator-session> session after each round is written or graded.",
               "edit": {"via": "claude", "at": now}}, open(os.path.join(OUT, f"ds_{k}.json"), "w"))
if "--page" in sys.argv:
    json.dump({"title": "Big TNO Hill Climb", "lang": "en"}, open(os.path.join(OUT, "meta.json"), "w"))
    json.dump({"text": open(os.path.join(HERE, "index.html")).read(), "edit": {"via": "claude", "at": now}}, open(os.path.join(OUT, "index.json"), "w"))
print("\n".join(sorted(os.listdir(OUT))))
