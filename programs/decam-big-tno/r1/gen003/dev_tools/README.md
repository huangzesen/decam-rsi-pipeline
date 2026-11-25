Development tools of gen003 (not imported by finder.py, not needed to run it). Synthetic bodies only: random two-body orbits.

- plant.py SRC DST N SEED [RMIN RMAX]: copy of a patch with N planted bodies (appended and regrouped by exposure) + truth.npz.
- score.py candidates.json DST [NTOP]: found bodies as the task scores them, candidates matching no planted body, carried classes.
- funnel.py OUT DST: where bodies are lost (needs a run with FINDER_DEBUG=1; DST/seedable.npy and stats.npy are optional).
- cdbg.py OUT DST: what the carry vote saw for arcs of planted bodies (needs OUT/carry.pkl from FINDER_DEBUG=1).

finder.py development switches: FINDER_DEVCACHE (prep cache file), FINDER_DEBUG (dumps raw*.pkl, grown.pkl, carry.pkl,
debug.pkl), FINDER_RR=lo,hi (one distance slice in au), FINDER_PASSES=0,1, FINDER_LOADRAW / FINDER_LOADGROWN / FINDER_LOAD3
(restart from dumps: growth + carry only, ~2 min on CPU), FINDER_NOCARRY, FINDER_NOFIXG.
