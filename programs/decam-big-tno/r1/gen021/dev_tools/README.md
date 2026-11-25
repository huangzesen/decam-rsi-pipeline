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

Added by gen007 (paths inside: /tmp/p2 = gen006's planted set, /tmp/w/cache_p2.npz):
- incr.py CAND DST BASEFOUND.npy: bodies found by CAND that a baseline had not (BASEFOUND = /tmp/found.npy written by a
  complete run of score.py on the baseline - do not pipe score.py into head, it dies before it saves), their nights per
  season, and the candidates matching nothing.
- finder.py switches added: FINDER_MAGP (magnitude agreement of a pair, 1.0 mag; 0 = off), FINDER_OLDORDER (gen006's order
  of the distances), FINDER_ADAPT (0: no pair budget on schedule) and FINDER_MPMIN (its floor, 2e6 pairs; set it to ~2e4
  to see it act on the sample with `--procs 1`), FINDER_P5B (now on; 0 = off) and FINDER_SPARE5 (its share of the spare
  time, 0.6), FINDER_P7 (1: the two-season pass, off) with FINDER_GS7 / FINDER_CELL7 / FINDER_MAXPAIRS7 /
  FINDER_MAXSEEDS7 / FINDER_MINN7 / FINDER_GD7 / FINDER_SPARE7.
- `--procs 1` runs torch on all cores: do not run it beside a timed 8-core run.

Added by gen009 (paths inside: /tmp/p3 = planted set made with plant_real.py ... 3000 11 39.5 41.5, /tmp/w/cache_*.npz = prep caches,
/tmp/w/cnt.npy = static count of every row, written by sharpstat.py):
- plant_real.py: plant_faint.py with the sharpness a real point source has: 0.97 + N(0, (3.4 / S/N)^2 + 0.08^2) instead of
  1.00 +- 0.08. Use this one: with the old planted sets the sharpness cuts cost nothing, on the sky they cost 15-30 % of the
  faint detections.
- sharpstat.py SCEN: histogram of the sharpness by S/N for static (>= 3 nights), two-night and one-night detections; writes
  /tmp/w/cnt.npy. ~3 min.
- audit.py CAND.json SCEN [PMIN] [CNT.npy]: for each candidate with p >= PMIN (not planted), an independent orbit fit and every
  catalogue detection within 2" of its track in every exposure: S/N, sharpness, flag, static count, held or not. Shows what the
  cuts take from a real body.
- astro.py: positional scatter of detections around bright reference stars by S/N (real astrometric error against the planted model).
- verif.py SCEN CACHE [T]: share of clean detections that another night's same-band exposure could have shown (a "verified
  non-static" flag); measured, not used (17 % of the clean detections against 46 % of the planted ones).
- useful.py SCEN CACHE [T] [RAD]: on how many other nights could a body of the detection's magnitude be seen within RAD deg;
  measured, not used (80 % of the seeds have >= 4 such nights within 1.3 deg).
- finder.py switches added: FINDER_SHW (sharpness scatter x S/N, 3.4; 0 = the fixed windows of gen007), FINDER_SHS (share of it
  that the seed window follows, 1; 0 = seeds keep 0.5-1.8), FINDER_FLOODCAP (a flooded exposure keeps this multiple of the
  median count of its brightest one-night detections, 1; 0 = flooded exposures are dropped as in gen007).
- plant_colour.py: plant_real.py with one redness per body (g-r 0.4-1.1, r-i 0.1-0.5, r-z 0.2-0.9) instead of the colours
  the code assumes (0.6, 0.25, 0.4): tests the magnitude agreement cuts and the scatter penalty of the score.

Added by gen012 (finder.py here = three-way merge of gen010 and gen011 on gen008, plus this generation's changes; the tools
of gen010 (cls.py, cfun.py, p8ana.py, pur.py, sparse.py, ucand.py) and gen011 (g3stat.py, lowp.py, usestat.py, ccdstat.py,
nearstat2.py) were copied in as they were - their descriptions are in /app/history/gen010 and gen011 dev_tools/README.md):
- zpstat.py: per-exposure zero-point offset measured on the static stars of the sample (same star, same band, bright,
  unsaturated: median residual per exposure), independent of finder.zp_calib. Writes /tmp/t/zp_off.npy. ~10 s.
- plant_zp.py: plant_colour.py whose bodies follow the true zero points: `PLANT_ZP=/tmp/w/cache_s2.npz python3 plant_zp.py
  /app/data/sample /tmp/p5 3000 11 39.5 41.5` (the cache is any prep cache written by this finder.py: it holds `mzc`, the
  re-measured zero points). Without PLANT_ZP it is plant_colour.py. Use this one to test anything that compares magnitudes:
  bodies planted with the catalogue's magzero agree with it by construction, real ones do not.
- feat.py CAND.json [DST] [-l]: non-detection counts (strong and marginal band), magnitude scatter, fitted redness and p of
  the candidates, split into planted-true and non-matching when DST is given; -l lists the non-matching ones.
- finder.py switches added: FINDER_ZPCAL (1; 0 = the catalogue's magzero as it is), FINDER_ZPSNR (40, lowest S/N of a
  calibration star), FINDER_COLR (0.25, redness hypotheses of the magnitude tests; 0 = gen009's fixed colours), FINDER_NDSOFT
  (2.5, width in S/N of the marginal band of the non-detection test), FINDER_NDW2 (0.6, cost of a clean miss there; 0 = off),
  FINDER_SRW (0.09", scatter of a real 3-night seed's fit for the prior of carried linkages; 0 = off), FINDER_R3HALF (0.25",
  largest fit rms of a 3-night linkage admitted once its store is half full).
