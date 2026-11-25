# footprint — whole-archive cost from NSC DR2 exposure metadata

Started 2026-10-29 (the experiment program). Scales the OF201 field's measured search
cost (50-1000 au, jobs <slurm-job-id> / 1109527 / 1109528) to every DECam patch-window in NSC DR2.

| Step | Script | Output |
|---|---|---|
| exposure table (412,116 exposures, Data Lab, anonymous) and single-detection counts for 300 random DECam exposures | `fetch_exposures.py` | `data_real/nsc_exposures/slices/`, `sample_n1.csv` |
| mid-times, nights, heliocentric observer, galactic latitude, coverage | `prep.py` | `data_real/nsc_exposures/exposures.npz` |
| predicted single-detection count per DECam exposure | `fit_n1.py` | `n1_pred` in `exposures.npz` |
| pair-count geometry: two DECam disks (r = 1.1 deg) in the BK frame, pairs within R | `proxy.py` | `g_table.npz` |
| the field through the proxy, against its measured raw pairs | `field.py` | `field.json` |
| measured single-detection densities: one 0.25 deg cone per patch on `nsc_dr2.object` (ndet = 1, grouped by MJD = exposure) | `fetch_cones.py` | `data_real/nsc_exposures/cones.jsonl` |
| cones and regression against exact counts | `check_cones.py` | printed |
| every patch-window (7,056 patches with 1.5 deg cores, 2.6 deg kept; 2-season windows split at conjunction; >= 5 nights) | `estimate.py` (`--cones` for measured densities) | `est_c4d.json` (regression), `est_c4d_cones.json` (cones; use this) |
| totals per method and distance, where the cost sits, seed-exposure caps | `summarize.py` | printed / `--json` |
| counts drawn from the costliest windows against the prediction | `validate_n1.py` | CSV |
| every DECam science exposure's pointing, 2012-2026 (NOIRLab Astro Data Archive, raw, 845,585; for the log page's sky tab, `../log/make_sky.py`) | `fetch_archive_pointings.py` | `data_real/noirlab_decam/decam_raw_<year>.json` |
| Rubin's actual visits from its public nightly scheduler reports (406 nights, 127,542 visits; for the sky tab) | `fetch_rubin_visits.py` | `data_real/rubin_visits/visits_<night>.json` |
| share of the archive in windows of >= 3 nights (2026-11-11; per exposure within 0.25 / 0.5 / 1 deg and 15 / 30 / 60 days, and the task's 1 deg x 60-day field-windows by size and sky class; run with `~/anaconda3/bin/python -I`, 4 min) | `window_share.py` | `window_share.json`, `window_share.log` |
| the same list with what grouping needs (2026-11-11; 846,721 exposures: proposal, exact start time, exposure number, object name, release date, file stamp `c4d_YYMMDD_HHMMSS`) | `fetch_archive_meta.py` | `data_real/noirlab_decam_meta/decam_meta_<year>.json` (134 MB) |
| windows of >= 3 nights sorted into 14 families (cadence x sky), 90 parents (program inside a family, >= 50 windows) and 226 groups (a parent, or a part of it when production would cost > 50 A100-hours; `--max-cost`, `--min-windows`), for `../../DECAM_PLAN.md`; PIs of the main programs by `fetch_proposal_pis.py` -> `proposal_titles.json`; run with `~/anaconda3/bin/python -I`, 10 s | `groups.py` | `groups.json` (families, parents, groups `g001`-`g226`, statistics, cost, sample windows), `groups_windows.json` (27,252 windows with labels), `groups.log` |
| which known TNOs (JPL SBDB, 2026-10-31) had >= 5 detectable nights in two seasons; ED138 looked up in the real catalogue with `../diag/nsc_presence.py` | `known_yield.py` | `known_yield.json`, `data_real/ed138_nsc_presence.csv` |
| inventory by sky patch x observing season (2026-11-12; local metadata only): every known TNO placed on the 594,565 exposures with files, then today's windows against re-formed ones (overlapping, 120-day, whole season, 2 and 3 deg cells) by nights, predicted tiers, data and A100-hours; the dense / sparse split. `--cache DIR` keeps the placements (first run 4 min, then seconds) | `season_inventory.py` | `season_inventory.json`; `../../DECAM_PLAN.md` section 9 |
| the same for the target of bodies >= 200 km, H <= 7 (2026-11-12): reach of single exposures, near arm (detection + linking) and far arm (stacks), known H <= 7 bodies recoverable by each, the window question for them | `bright_target.py` (imports `season_inventory.py`, same cache) | `bright_target.json`; `../../DECAM_PLAN.md` section 9c |
| the DECam images already at NERSC (DESI Legacy Surveys staging area, `/global/cfs/cdirs/cosmo/staging/decam`) against the whole archive (2026-11-15; metadata only): 321,619 exposures = 41% of the science exposures with InstCal files, by year, filter, sky, exposure time and program; 15,408 of the 26,835 search windows are entirely there | `nersc_mirror.py` | `nersc_mirror.json`, `data_real/nersc_decam_mirror/mirror_exposures.tsv` |

Facts this rests on:
- Pair-stage time follows the raw pair count (2.5-4.5 ns per pair on a GH200 in every run).
- The proxy reproduces the field's measured raw pairs per hypothesis with one factor, 1.20-1.27, from 50 to 1000 au
  (`field.json`). The factor reflects clustering of detections; the archive estimate applies it unchanged [E].
- The per-exposure regression (`fit_n1.py`) has 0.52 dex scatter. Its cross-validated total is within 4%, but in the
  windows that carry the cost it is 5-10x high (Large Magellanic Cloud, bulge) or 5x low (the DES deep field at CDF-S),
  so `est_c4d.json` is wrong where it matters. The cones replace it: 131,482 exposures measured, the rest scaled by
  their window's measured / predicted ratio. Cones read 1.12x the exact counts (median of 138, 0.25 dex scatter),
  divided out (`--cone-scale`).
- Result and table: `summary_c4d_cones.json`, BENCHMARK.md "Whole-archive estimate". Re-run:
  `python -I footprint/estimate.py data_real/nsc_exposures/exposures.npz footprint/field.json footprint/est_c4d_cones.json --cones data_real/nsc_exposures/cones.jsonl`
  (14 min, one core), then `python -I footprint/summarize.py footprint/est_c4d_cones.json footprint/field.json --json footprint/summary_c4d_cones.json`.
- The speed-ups per method are the GH200 ratios measured at 60-62 and 84-89 au, applied to the GB200 brute-force times.
  Seed-only cuts run as fast as removal (job <slurm-job-id>, 84-89 au) and keep recall (98.4% with veto + sharp + flags).
- The seed-cap rows (`cap 300/150/100`, cost x (0.8 q^2 + 0.2 q)) are withdrawn: on this field a cap loses 12-26% of
  injected bodies for 1.2-1.4x speed (job <slurm-job-id>). Use the `cap none` rows; `clean, <= 1000 exp` leaves out the deep
  fields, which need per-night stacks or another treatment.
