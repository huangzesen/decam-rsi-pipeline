Development tools of gen024 (not imported by finder.py, not needed to run it). Older tools: `/app/history/gen022/dev_tools/`
and `/app/history/gen019/dev_tools/` (`plant_colour.py`, `cls.py`, `src_eval.py` are copies from there).

Paths inside are those of the session: /tmp/p4 = planted set (`plant_colour.py /app/data/sample /tmp/p4 3000 11 39.5 41.5`),
/tmp/w/cache_p4.npz = prep cache (`FINDER_DEVCACHE`), /tmp/w/base = a baseline run with `FINDER_DEBUG=1` (raw*.pkl, grown.pkl;
`grown_full.pkl` = the growth cache of the first replay, which had the time to finish the growth; g3.pkl was an empty list
because the 600-s baseline never reached its carry stage - give the baseline ~800 s to get the 3-night seeds).

- replay.sh NAME BUDGET [ENV=VAL ...]: growth + carry replayed from the dumps of /tmp/w/base into /tmp/w/r_NAME
  (`--procs 1`, 4 threads, ~3 minutes with budget 300). Rows of one table differ in nothing but the switches given.
- cls.py CAND DST [FOUND.npy]: found bodies by cadence class (a-f) and brightness; writes the list of found bodies.
- cad.py [FOUND.npy]: scored bodies (and found ones) by the largest number of kept / seed nights inside 128 days.
- rank.py CAND: true candidates above the k-th candidate that is not a found body, and counts by p.
- rank2.py CAND [1]: the same against the candidates with NO planted detection (what the real sky gives by chance).
- src_eval.py CAND: candidates true / mixed / with no planted detection by origin (`src` 4 = one-night carry) and by p.
- one_eval.py RUNDIR, one_eval2.py RUNDIR: the linkages of the one-night carry in carry.pkl against the truth, by chance
  level `fa` and by rms / expected rms.
- lost2.py FOUND.npy [RUNDIR]: census of the lost bodies by their longest pure arc of the growth, its fit, and what far
  nights they have; with RUNDIR (a run with a loose `FINDER_ONE`) the chance level each lost 4-night arc got.
- why.py FOUND.npy RUNDIR: per lost body with a pure well-fitting 4-night arc, what the one-night vote saw of it (needs
  the run made with `FINDER_TRUTH=/tmp/p4/truth.npz`).
- dbl.py FOUND.npy: far nights seen twice (two clean detections >= 20 min apart) among found / lost bodies.
- ratio.py CAND: rms / expected rms of true, mixed and chance candidates by number of nights.
- whatif.py CAND: the fit-quality term of `score()` re-applied offline with other weights and shapes.
- mixed.py CAND FOUND.npy: candidates that are neither pure nor chance, by their number of true nights.

finder.py switches added: FINDER_ONE (0: no one-night carry), FINDER_ONEREACH, FINDER_ONEMTOL, FINDER_ONET, FINDER_ONERR,
FINDER_ONERFL, FINDER_PR1, FINDER_ONESN2 / FINDER_ONE2 / FINDER_ONE2KEEP (second tier, off), FINDER_WRR (0: no fit penalty),
FINDER_RRC, FINDER_RRFLOOR.
