# KBMOD baseline for the planet-finder RSI task

A good-faith run of the field's standard open-source shift-and-stack pipeline, KBMOD, on the RSI task's data, wrapped in
the task's interface, so its score on the sealed fields can sit next to the RSI programs'. It is a proof that the
pipeline runs and finds planted bodies, not a tuned baseline: the user capped development at 1 interactive node-hour
and asked for no tuning sweep (2026-11-10, "dont use too much of my compute btw.. just test as a proof").

Built blind, like the RSI agents: only the four practice fields (`rsi/scen_dev/`) were used; nothing under `private/`,
`scen_sealed/`, grades, candidates or NERSC `scen/`, `runs/` was read.

## Contents

| Path | What |
|---|---|
| `submission/finder.py` | the finder, task interface `python -I finder.py --scenario S --out OUT.json --budget-s 900` |
| `nersc/build_env_kbmod.sh` | how `env_kbmod` was built on Perlmutter (login node) |
| `tools/run_node.py` | development runner on one GPU node (finder_node.py's rules, plus a per-item python and `FINDER_INJECT`) |
| `tools/mkinj.py` | planted bodies for a practice field (the same generator as r2 gen006's `devtools/mkinj.py`) |
| `tools/score_inject.py` | recovery of planted bodies: >= 3 nights within 2" by one of a program's 25 highest-p candidates |
| `tools/coverage_cost.py` | GPU hours a full KBMOD search of a field would take, with the finder's own grid rules |
| `inject/` | the planted bodies used |
| `runs/` | logs, outputs and timings of the practice runs on the A100 |

## KBMOD version and build

- dirac-institute/kbmod, `main` at commit `60b8c8216318ccd633eb9be05e8545c292ee27fa` (2026-11-02, "Update Python support
  and keep CUDA convolution builds launchable"; reports version `1.1.1.dev2020+g60b8c8216`). The repository has no
  release tag after v1.1.0 (April 2023), and the reprojection/EBD tools and the current search API exist only on
  `main`, so the commit is pinned instead of a tag. Submodules: eigen `bc3b398`, pybind11 `f5fbe86`.
- Environment `$NERSC_SCRATCH/rsi-planet/env_kbmod` (new; the grading `env` was not touched): conda-forge
  Python 3.12, numpy 2.2.6, scipy 1.18.1, astropy 8.0.2, reproject, plus torch 2.7.1+cu126 (KBMOD pins torch <= 2.7.1).
  KBMOD compiled from source on a login node with `cudatoolkit/12.9` (nvcc 12.9.41) and `gcc-native/13`
  (`pip install .`, CMake `CUDA_ARCHITECTURES all`); `kbmod.search.HAS_CUDA` and `HAS_OMP` are true. The conda
  env carries its own libstdc++/libgomp in `env_kbmod/lib`, which `finder_node.py` puts on `LD_LIBRARY_PATH`, so the
  finder runs in the grader's stripped environment. Script: `nersc/build_env_kbmod.sh`.

## What the finder does, and what is KBMOD's

KBMOD searches straight-line trajectories through a stack of aligned images. The task's fields are sparse: 4-15 nights
with 1-3 exposures each over 3-60 days. So the finder:

1. **Window.** Picks the shortest run of 3 nights (ties: more exposures) and searches only the exposures in it. KBMOD's
   cost grows as (span)^2 through the velocity grid, and 3 nights is what a hit needs. On the practice fields the
   window spans 2.0-11.1 days and holds 4-6 of the 5-13 exposures.
2. **Common grid.** Reads every CCD (image and data-quality mask) of the window's exposures and of up to 12 exposures
   in all, subtracts a 128-px block-median sky, and resamples each exposure onto one tangent-plane grid at the native
   0.263"/px (about 29,700 x 26,900 px for a DECam field), bilinearly, on the GPU. Images are put on one zero point with
   typical TNO colours (g-r 0.6, r-i 0.25, r-z 0.4, r-Y 0.45), the job of KBMOD's `color_scale`. Variance per image:
   the robust sky noise.
3. **Static sky.** KBMOD is meant to run on difference images. There is no template here, so the finder makes one: the
   per-pixel median of all loaded exposures (at least 3 per pixel), subtracted from the window's images. The
   footprints of static sources (template smoothed by a 1.5 px Gaussian above 3 sigma, grown by 2 px) are masked,
   because the template is not PSF-matched; this is the role of KBMOD v1's "global mask" of pixels flagged in several
   images.
4. **EBD reprojection.** For each explicit barycentric distance guess d (KBMOD's EBD frame; heliocentric here, from the
   observer positions in `exposures.csv`), every window image is resampled into the frame where a body at distance d
   has its parallax removed: a pixel's direction u, the point d*u, seen from the observer at that exposure. This is the
   same transform as `kbmod.reprojection_utils` (ray from the observer meets the sphere |P| = d), done on the GPU.
5. **KBMOD search**, `kbmod.run_search.SearchRunner.run_search` on 8192 x 8192-px tiles of start pixels (padded for
   motion): PSF-weighted psi/phi likelihood images (Gaussian PSF from each exposure's FWHM), the GPU grid search over an
   `EclipticCenteredSearch` velocity grid, GPU sigma-G filtering (`sigmaG_lims` [25, 75]), `num_obs` = images - 2
   (at least 3), `lh_level` 6, the peak-offset stamp filter (3 px), near-duplicate removal and `nn_start_end`
   clustering at KBMOD's default `cluster_eps` of 20 px, `results_per_pixel` 1.
   Light-curve test after the search: chi^2 per degree of freedom of the per-image fluxes psi/phi about KBMOD's flux,
   with a 20% allowance for colour, PSF and zero-point errors, at most 3. Sigma-G cannot reject 2 outliers among 4-6
   points, and without this test every top slot went to trajectories joining two bright single-image transients; the
   original KBMOD (Whidden et al. 2019) filtered light curves with a Kalman filter for the same reason.
6. **Grid.** Distance cells in 1/r start with the one holding 42 au and go alternately inwards and outwards over
   28-150 au; a cell is narrow enough that the curvature it leaves is under 1 px and the parallax drift across it
   stays below the body's own motion. Velocities per cell: the range of bound TNO-like orbits in the cell (motion
   within KBMOD's default +-pi/15 of the ecliptic, (v_t/v_c)^2 0.8-1.25, radial speed +-0.2 v_c), stepped so
   neighbouring trajectories part by at most 2 px over the window (KBMOD's docs: about a pixel at the last image).
7. **Budget.** Cells run in that order until 45 s before the budget. If the first cell cannot cover every tile (planned
   at 9e10 trajectory-images/s before the first tile, then at the measured rate), its grid is narrowed once around
   circular ecliptic motion (to no less than 30% per axis) so the whole field is searched; later cells cover only the
   tiles that fit. Output is rewritten after every tile.
8. **Plumbing to the task's output** (not KBMOD): each kept trajectory (data under it on >= 3 nights, valid points
   after sigma-G on >= 2; the 600 best per tile) gives sky directions at the window's exposures; a heliocentric state
   is fitted by intersecting those rays with spheres |P| = r and a straight line in emission time (light time), r
   scanned over 26-200 au for the best bound fit (own code; two-body propagation as in the grader).
   `p` = 1 / (1 + exp(-(lh - 9) / 1.5)), monotonic in KBMOD's likelihood; `mag` from KBMOD's flux. Duplicates (within
   12" at the first and last exposure, across tiles and cells) are merged, keeping the higher likelihood.

Not used (on purpose): seeding from detection pairs, triplet linking, orbit-population priors or score calibration
from the RSI programs. KBMOD's own `reproject_work_unit` and `correct_parallax*` were not called. The first resamples
with `reproject_adaptive` on the CPU: `kbmod.reprojection.reproject_image` took 26.4 s for one 4094 x 2046 CCD layer
on a Perlmutter login node, and KBMOD reprojects three layers (science, variance, mask), so one distance guess for a
6-exposure window of 60 CCDs is about 28,000 CPU-seconds, about an hour on the grader's 8 threads, against a 900 s
budget. The second needs the `de432s` ephemeris from the internet, which the grader blocks. The finder implements the
same EBD transform with the observer positions the task gives, and resamples bilinearly on the GPU.

## Practice runs on the A100

Final code, run the grader's way (`tools/run_node.py`: fresh copy, stripped environment, 8 threads, one A100 each, 4
finders on one Perlmutter node, `--budget-s 900`, killed at 930 s, no network), job <slurm-job-id>, outputs in
`runs/final/`. `python3 tools/validate_runs.py runs/final/*` gives ALL OK (the check.py test: exit 0, inside the
budget, valid output, no state the grader would drop). Two of the four runs had planted bodies added in memory
(`FINDER_INJECT`); without it the hook does nothing.

| field | exposures / nights | window: nights, span, images | load + template | distance cells searched | run time | candidates |
|---|---|---|---|---|---|---|
| scen01 | 7 / 4 | 60210-60221, 11.1 d, 5 | 65 s | 1 (38.7-45.9 au), grid narrowed from 82,680 to 10,602 velocities to cover all 16 tiles | 854 s, ok | 200 |
| scen02 (planted) | 13 / 6 | 60634-60643, 9.0 d, 6 | 103 s | 2 (38.3-43.4 au; 7,524 and 9,000 velocities) | 840 s, ok | 200 |
| scen03 | 12 / 8 | 60544-60553, 9.0 d, 4 | 102 s | 2 (38.6-43.3 au; 12,544 and 14,490 velocities) | 860 s, ok | 200 |
| scen04 (planted) | 5 / 4 | 58394-58396, 2.0 d, 4 | 50 s | 6 (28.0-68.6 au; 2,080-3,510 velocities each) | 853 s, ok | 200 |

(The log's "N distance cells searched" also counts cells whose tiles were all skipped for time; the column above
counts cells that ran.) scen01's first window exposure has a zero point 3.3 mag below the others (cloud); KBMOD
weights it by its variance, so that night carries almost nothing.

## Planted bodies: KBMOD against the RSI program r2 gen006

Same 20 bodies per field (`tools/mkinj.py`, seed 7, magnitudes 22.0-25.5, distances mostly 35-50 au), planted in
memory into the CCD pixels the same way by both programs (gen006's own `FINDER_INJECT` hook: Gaussian PSF from the
CCD's FWHM, flux 10^(-0.4 (mag - magzero - colour))). gen006 (`rsi/runs/r2/gens/gen006/submission`, unchanged) ran in
the grading `env` on the same kind of node, 900 s, job <slurm-job-id>. A body counts when one of the program's 25
highest-p candidates is within 2" of it on >= 3 nights (`tools/score_inject.py`); 17 of 20 bodies per field are in the
field on >= 3 nights. The real bodies in these fields compete for the same 25 slots.

| mag | bodies | KBMOD scen02 | gen006 scen02 | KBMOD scen04 | gen006 scen04 |
|---|---|---|---|---|---|
| 22.0-22.5 | 1 + 1 | 1 | 1 | 1 | 1 |
| 22.5-23.0 | 4 + 4 | 0 | 4 | 3 | 1 |
| 23.0-23.5 | 2 + 2 | 0 | 2 | 0 (1 below the top 25) | 0 |
| 23.5-24.0 | 1 + 1 | 0 | 0 | 0 | 0 (1 below the top 25) |
| 24.0-25.5 | 9 + 9 | 0 | 0 | 0 | 0 |
| all | 17 + 17 | **1** | **7** | **4** | **2** |

Together: KBMOD 5 of 34 in its top 25 (6 anywhere in its 200), gen006 9 of 34 (10). Neither finds a planted body
fainter than 23.5 on these two fields. KBMOD's misses on scen02 are mostly coverage: in 900 s it searched only
38.3-43.4 au and +-12 deg of the ecliptic, and 13 of the 17 bodies lie outside that (planted at 30-50 au with a hot
component of ~16 deg; two of them, at 43.7 and 48.5 au, were still found). Of the 4 inside, one is mag 24.8, one has (v_t/v_c)^2 = 0.77, just outside the grid, one
(23.05) is found at lh 7.5, far below the top 25, and one (22.79) is missed. The 22.67 body at 48.5 au was found in the
42 au cell (lh 11.7, a hit on its own) but merged into a stronger trajectory of the same body from the 39.4 au cell,
whose straight line in that cell's frame misses the middle night by 2.4": a body far from the cell's distance keeps
parallax curvature that KBMOD's linear model cannot follow. On the 2-day scen04 window KBMOD searches 28-69 au and
beats gen006 there. An earlier run of the same configuration with a 3" duplicate merge (`runs/batch2/`) gave KBMOD 2 and 3; the first run without the
light-curve test (`runs/batch1/`) gave 0 and 0, its 200 slots all taken by trajectories that join two bright
single-image transients (asteroids, cosmic rays), which sigma-G cannot reject among 4-6 points.

## Development record

- `runs/debug/`, `runs/batch1/`, `runs/batch2/`: earlier code (no light-curve test; then 3" merge and 5 px clustering).
  Changes after the first look, all before the final run: footprint mask of static sources, the chi^2 light-curve
  test, KBMOD's default 20 px clustering radius, a 12" merge, nights counted by data coverage under the trajectory,
  and planning the first tile with a prior rate. No parameter sweep; nothing was tuned on the planted bodies' scores.
- NERSC use: interactive jobs <slurm-job-id> (21:26), 61084114 (14:19), 61084468 (14:23) on `$NERSC_GPU_ACCOUNT`, 0.84 node-hours in
  all; build and smoke tests on a login node. The practice fields are in `$NERSC_SCRATCH/rsi-planet/scen_dev`
  (22 GB), development files in `.../rsi-planet/kbmod_dev`, the KBMOD clone in `.../rsi-planet/kbmod_build`.

## What full coverage would take

KBMOD's GPU search ran at 0.6-1.5e11 trajectory-images per second on the A100 (4 finders per node, as in grading).
`tools/coverage_cost.py` applies the finder's own grid rules (distance cells over 28-150 au, KBMOD's default
+-pi/15 angle range, 2 px end-point spacing) to every start pixel of a field (29,679 x 26,892) at 1.1e11/s:

| field | window: span, images, cells | A100-hours, window, 28-150 au | (38-48 au only) | all nights: span, images | A100-hours, all nights |
|---|---|---|---|---|---|
| scen01 | 11.1 d, 5, 13 | 6.0 | (1.6) | 33.0 d, 7 | 184 |
| scen02 | 9.0 d, 6, 19 | 1.8 | (0.36) | 54.0 d, 13 | 4,220 |
| scen03 | 9.0 d, 4, 21 | 2.2 | (0.40) | 56.9 d, 12 | 4,180 |
| scen04 | 2.0 d, 4, 13 | 0.17 | (0.02) | 29.0 d, 5 | 39 |

The budget is 0.25 A100-hours per field. So inside 900 s KBMOD covers the whole window only on short windows (2-day
scen04: 28-69 au), and on 9-11-day windows one or two cells near 38-46 au (on the 11-day window one cell, with
its grid narrowed to 13%); a search that stacks every
night of a field (where the faintest, weight-4 bodies live) costs 40-4,000 A100-hours per field with this method.
KBMOD's own CPU reprojection would add about an hour of 8-thread CPU time per distance guess per field (above).

## Sealed run (for the main session)

Same rules as the RSI grading (`rsi/harness/ls6/finder_node.py`: fresh copy in node-local `/tmp`, stripped environment,
8 CPU threads, one A100 per field, `--budget-s 900`, killed at 930 s, no network), only the python differs: launching
`finder_node.py` with `env_kbmod`'s python makes it run the finder with that python and put `env_kbmod/lib` on
`LD_LIBRARY_PATH`. From the Mac, through the existing master (`SSH="ssh -S $SSH_CM_DIR/nersc-$NERSC_USER.sock -o
ControlMaster=no -o BatchMode=yes -o ProxyCommand=false $NERSC_USER@perlmutter.nersc.gov"`):

```bash
SCR=$NERSC_SCRATCH/rsi-planet
R=$SCR/kbmod_sealed/$(date +%Y%m%d-%H%M%S)          # a new folder, not under $SCR/runs
SIDS=sAAAA,sBBBB,...                                 # the 10 sealed field ids, as in rsi_loop's config
$SSH "mkdir -p $R/subs $R/out"
rsync -a -e "ssh -S $SSH_CM_DIR/nersc-$NERSC_USER.sock -o ControlMaster=no -o BatchMode=yes -o ProxyCommand=false" \
  rsi/baselines/kbmod/submission/ $NERSC_USER@perlmutter.nersc.gov:$R/subs/kbmod/
rsync -a -e "ssh -S $SSH_CM_DIR/nersc-$NERSC_USER.sock -o ControlMaster=no -o BatchMode=yes -o ProxyCommand=false" \
  rsi/harness/ls6/finder_node.py $NERSC_USER@perlmutter.nersc.gov:$R/finder_node.py
# 10 fields on one 4-GPU node = 3 waves of <= 15.5 min (~47 min, about 0.8 node-hours)
$SSH "cd $R || exit 1; setsid nohup salloc -J kbmod-sealed -A $NERSC_GPU_ACCOUNT -C gpu -q interactive -N 1 -t 55 \
  --gpus-per-node=4 --immediate=120 srun -N 1 -n 1 --cpu-bind=none -c 128 --gpus-per-node=4 \
  $SCR/env_kbmod/bin/python -I $R/finder_node.py --subs kbmod=$R/subs/kbmod --scen-root $SCR/scen \
  --sids $SIDS --cap-s 900 --out $R/out > int.log 2>&1 < /dev/null & disown; echo ok"
```

Or `-N 3` with `srun -N 3 --ntasks-per-node=1 ...` (finder_node shards the runs by `SLURM_PROCID`): one wave of 16 min
on 3 nodes. The outputs land in `$R/out/kbmod/<sid>/{candidates.json, info.json}`, the layout `nersc.py` fetches into
`gens/<g>/ls6_runs/`, so `grade.py --runs-dir` scores them as it scores a round. The submission folder holds only
`finder.py` (no data files). The `interactive` QoS takes 2 jobs per user across every agent, and the RSI r2 loop
grabs one about every 2 h.
