#!/usr/bin/env python3
"""NERSC side of fill/relay.py: delete files the relay has seen land on Vista. Reads paths relative to decam-fill/ on stdin;
accepts only p<k>/img/<YYMMDD>/c4d_* and pmac/img/<YYMMDD>/c4d_* files (no '..', no absolute paths). Prints how many it removed."""
import os, re, sys
ROOT = "$NERSC_SCRATCH/decam-fill"; OK = re.compile(r"^(p\d+|pmac)/img/\d{6}/c4d_[A-Za-z0-9_.]+$"); n = 0   # pmac = the Mac share (relay_mac.py, 2026-11-20)
for line in sys.stdin:
    x = line.strip()
    if OK.match(x):
        p = os.path.join(ROOT, x)
        if os.path.isfile(p) and not os.path.islink(p):
            os.remove(p); n += 1
print(n)
