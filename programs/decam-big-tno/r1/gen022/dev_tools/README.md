Development tools of gen022 (not imported by finder.py, not needed to run it). The older tools are in
`/app/history/gen019/dev_tools/` (README there); `plant_colour.py` and `cls.py` are copies from it.

Paths inside are those of the session: /tmp/p4 = planted set (`plant_colour.py /app/data/sample /tmp/p4 3000 11 39.5 41.5`),
/tmp/w/cache_p4.npz = prep cache (`FINDER_DEVCACHE`), /tmp/w/base = a run with `FINDER_DEBUG=1` (raw5.pkl, grown.pkl, g3.pkl).

- replay.sh NAME BUDGET [ENV=VAL ...]: growth + carry replayed from the dumps of /tmp/w/base (`--procs 1`, 4 threads) into
  /tmp/w/r_NAME; rows of one table differ in nothing but the switches given. Run two side by side at most, and nothing else.
- cls.py CAND DST [FOUND.npy]: found bodies by cadence class and brightness, candidates not matching (gen010's).
- src_eval.py CAND: candidates true / mixed / with no planted detection by origin (`src` 0 search, 1 vote, 2 arc pool,
  3 bright pool) and by p.
- lost_stat.py FOUND.npy: census of the lost bodies - fewer than 5 kept nights, no pure 3-night linkage, longest pure
  arc 3 or 4 nights - and how many far nights (outside the arc) they have by S/N of the clean detections.
- lost_reach.py FOUND.npy: the same far nights inside a reach of 1,100 / 2,600 / 5,000 days around the arc.
- trk_stat.py: same-night pairs of kept detections (0.8" + 4.5"/h) and how many nights of each planted body have one.
- gate_stat.py: the flux gate (`FINDER_FGATE`, off) against the old magnitude test, on arcs made of 4 members of a
  planted body: share of the body's other detections and of random kept detections that pass.
- ccdrep.py [CELL_PX]: number of exposures with a detection in the same CCD pixel cell, for all / kept / seed detections
  (takes 4 minutes: three sorts of 35 million keys).

finder.py switches added: FINDER_BPOOL (0: no bright pool = gen019), FINDER_SNB, FINDER_BPOOLT, FINDER_BPOOLT3,
FINDER_BPOOLREACH, FINDER_BPOOLNEAR, FINDER_BPOOLQ, FINDER_BPOOLF3, FINDER_FGATE (+ FGINT, FGCOL, FGEXT, FGRHO).
