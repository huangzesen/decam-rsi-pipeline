Development tools of gen023 (not imported by finder.py). Earlier generations' tools are in /app/history/gen021/dev_tools.
Paths inside are those of the session (/tmp/p5 = planted set, /tmp/w = work folder with prep caches and run outputs).

- plant_zp.py, cls.py: gen021's planter and scorer, unchanged (PLANT_ZP=<prep cache> plant_zp.py SRC DST N SEED RMIN RMAX; cls.py CAND.json DST [found.npy]).
- study.py CAND.json [pmin] [tol]: for the confident candidates of an UNPLANTED run: own orbit fit, then every raw catalogue
  detection within tol of the track in every exposure, with S/N, sharpness, flag, static count and whether the prep kept it.
- snstat.py: same-night partners of clean detections and of planted ones; bodies that gain seed nights from them.
- seedstat.py: what confirmed detections add to the seed sets (planted share, missed bodies that gain a usable pattern).
- missab.py found.npy: missed planted bodies of the classes ">= 5 nights in a season" and "4 + far season of >= 2":
  kept nights, seed nights, span, median S/N, best grown linkage (needs grown.pkl of a FINDER_DEBUG=1 run).

finder.py switches added: FINDER_REACH4 (reach of the carry of 4-night arcs and of the arc pool, days; 1100 = gen019),
FINDER_TRK (0: no tracklet-pair pass), FINDER_TRKSHARE, FINDER_TRKKP, FINDER_TRKDT, FINDER_TRKTOL,
FINDER_TRKSMIN, FINDER_TRKTMAX, FINDER_TRKNC, FINDER_TRKOUT, FINDER_TRKPEN, FINDER_CONFSEED (0: no confirmed seeds),
FINDER_QCONF (0: gen019's carry queue), FINDER_CONFFA (1: gen019's chance test of the carry).
`FINDER_PASSES=99 FINDER_NOCARRY=1 FINDER_TRKSHARE=0.5` runs the tracklet-pair pass alone (10-60 s on the CPU).
