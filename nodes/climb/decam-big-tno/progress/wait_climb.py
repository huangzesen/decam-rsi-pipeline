#!/usr/bin/env python3
"""Tracker updater's wait (2026-11-19): waits up to 540 s for
runs/r2/climb.log to gain a line that changes the tracker (a round started, written, graded or a new best) after byte
offset LAST. Prints "NEW <size>", "SAME <size>" or "STOP" (progress/STOP_TRACKER exists). Usage: python3 -I progress/wait_climb.py LAST"""
import os, sys, time
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); L = os.path.join(H, "runs", "r2", "climb.log")
last = int(sys.argv[1]) if len(sys.argv) > 1 else 0; t0 = time.time()
KEYS = (" started ", " written", " reward ", " is the best", " leads:", "loop ended")
while time.time() - t0 < 540:
    if os.path.exists(os.path.join(H, "progress", "STOP_TRACKER")):
        print("STOP"); sys.exit()
    n = os.path.getsize(L)
    if n > last:
        with open(L) as f:
            f.seek(last); new = f.read()
        if any(k in new for k in KEYS):
            print("NEW", n); sys.exit()
    time.sleep(20)
print("SAME", os.path.getsize(L))
