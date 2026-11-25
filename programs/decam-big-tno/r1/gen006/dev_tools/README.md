Development tools of gen003 (not imported by finder.py, not needed to run it). Synthetic bodies only: random two-body orbits.

- plant.py SRC DST N SEED [RMIN RMAX]: copy of a patch with N planted bodies (appended and regrouped by exposure) + truth.npz.
- score.py candidates.json DST [NTOP]: found bodies as the task scores them, candidates matching no planted body, carried classes.
- funnel.py OUT DST: where bodies are lost (needs a run with FINDER_DEBUG=1; DST/seedable.npy and stats.npy are optional).
- cdbg.py OUT DST: what the carry vote saw for arcs of planted bodies (needs OUT/carry.pkl from FINDER_DEBUG=1).

finder.py development switches: FINDER_DEVCACHE (prep cache file), FINDER_DEBUG (dumps raw*.pkl, grown.pkl, carry.pkl,
debug.pkl), FINDER_RR=lo,hi (one distance slice in au), FINDER_PASSES=0,1, FINDER_LOADRAW / FINDER_LOADGROWN / FINDER_LOAD3
(restart from dumps: growth + carry only, ~2 min on CPU), FINDER_NOCARRY, FINDER_NOFIXG.

Added by gen004 (paths inside are those of the session: /tmp/p2, /tmp/w/cache_*.npz - edit before use):
- plant_faint.py: plant.py with magnitudes skewed to the faint end (21.0 + 3.6 sqrt(u)).
- pat.py: share of scored planted bodies with 3 seed nights in a 16/32/64-day window + a 4th night, seeds against all kept.
- funnel2.py: found rate by night pattern (nights, most nights in 64 d / 130 d, most nights outside the best 130 d).
- real.py OUT: the candidates of an unplanted run with their features and the seed share of their detections.
- ndstat.py CAND [DST]: hit / miss / opportunity nights of the non-detection test by class (true, mixed, none, unplanted).
- auc.py CAND DST [UNPLANTED_CAND]: ranking of true against chance linkages (<= 7 nights) for variants of the score.
- finder.py switches added: FINDER_NOPOOL (skip the arc pool), FINDER_P4 (spare-time pass with the wide seed set),
  FINDER_SNRSEED (lowest seed S/N, default 5.0); FINDER_DEBUG now also dumps g3.pkl (3-night linkages, for FINDER_LOAD3).

Added by gen005 (paths inside are those of the session: /tmp/p2 = planted set, /tmp/w/cache_p2.npz = prep cache, /tmp/w/info.pkl):
- pat2.py: night pattern of every scored planted body (nights per season, most nights in 32/64/130 d) -> /tmp/w/info.pkl.
- funnel3.py OUT: found / raw linkage / grown, by cadence class (ge5in130, 4in64, 4in130, 3in32, 3in64, 3in130, max2) and nights.
- cdbg2.py OUT: outcome of the carry vote for 3-night seeds of planted bodies (needs OUT/carry.pkl and OUT/funnel3.pkl).
- tdbg.py OUT: what the vote sees of a seed's own far detections (needs a run with FINDER_TRUTH=/tmp/p2/truth.npz FINDER_DEBUG=1):
  are they inside the box, does each agree with the model, do two nights share a parameter point, and what the best cell held.
- finder.py switches added: FINDER_TRUTH (truth.npz, for tdbg.py), FINDER_SNRB (lowest S/N of a P5 seed, 6.5), FINDER_NBMAX,
  FINDER_GS5 (distance step of P5), FINDER_MINN5 (3: P5 also returns 3-night linkages), FINDER_MAG3 (magnitude agreement of
  3-night linkages, 0 = off), FINDER_NGV (night groups of the vote); FINDER_LOADRAW / FINDER_LOADGROWN may now be combined with
  FINDER_PASSES (run one pass on top of dumped results).

Added by gen006 (paths inside: /tmp/p2 = planted set made with plant_faint.py ... 3000 11 39.5 41.5, /tmp/w/cache_p2.npz, /tmp/w/info.pkl from pat2.py):
- ana.py OUT: found / not matching, rank of true against non-matching candidates, found by cadence class x brightness
  (median over nights of the best S/N of the night: faint < 6, mid 6-9, bright); writes OUT/found.pkl.
- pot.py OUT: for the bodies ana.py reports lost, per class: do they have 3 / 4 seed-quality nights inside 32 / 64 / 130 d
  (all seeds, S/N >= 5.5, S/N >= 6.5), and seed pairs in two or three seasons.
- finder.py switches added: FINDER_MINN3 (4: the 64-day pass keeps no 3-night linkages), FINDER_ARC3 / FINDER_SNR3 (shortest arc
  and second-brightest S/N of a 3-night linkage of the 64-day and season-long passes), FINDER_K3 (most 3-night linkages per
  window and distance), FINDER_N3MAX (budget of returned 3-night linkages per pass in the single-process branch),
  FINDER_CLEANV (0: the carry vote takes every kept detection, as in gen005), FINDER_Q3 (0: carry queue by arc only),
  FINDER_P5B (season-long pass again on seeds down to FINDER_SNRC = 5.5 in spare time; off), `--procs 1` runs the
  single-process (GPU) branch on the CPU.
