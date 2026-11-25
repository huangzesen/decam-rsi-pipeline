#!/usr/bin/env python3
"""Wait until out/status.json is newer than mtime LAST (from the previous call), at most 540 s; print its mtime and age.
Used by the page updater agent between uploads. Usage: python3 -I fill/dash/wait_new.py LAST_MTIME"""
import os, sys, time
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "status.json"); last = float(sys.argv[1]) if len(sys.argv) > 1 else 0
t0 = time.time()
while time.time() - t0 < 540:
    if os.path.exists(os.path.join(os.path.dirname(p), "..", "STOP")): print("STOP"); sys.exit(0)
    m = os.path.getmtime(p)
    if m > last + 1: print(f"NEW {m:.0f}"); sys.exit(0)
    time.sleep(15)
print(f"SAME {last:.0f}")
