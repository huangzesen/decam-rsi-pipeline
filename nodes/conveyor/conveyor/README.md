# conveyor: continuous discovery, 5 % of the DECam sky at a time

Pipeline map (what runs where, logs, stop files, restarts): `../PIPELINE.md` (planet-search/PIPELINE.md).

2026-11-18. Started 13:00 PST.

## How it works

One coordinator on the Mac (`conveyor.py loop 3`, a tick every 3 minutes; log `conveyor.log`, state `state.json`).

**Search, on whichever GPU starts first.** Worker jobs wait in the queues of both sites. A worker's lanes (one GPU each)
pull regions from a to-do list (`ROOT/queue`) and claim each with `mkdir ROOT/claims/<unit>_<sid>`. The coordinator copies
every claim to the other site, so a region runs once. Each region runs the hill climb's best program as of the moment it
starts (`ROOT/best/CURRENT`, updated by the coordinator from `rsi/decam-big-tno/runs/r2/state.json`). A lane with nothing
to do for 15 minutes ends, so an idle worker costs at most 15 minutes. Budget: 1,800 s per region, 7,200 s for regions
over 1,000 exposures (these only go to lanes with about 100 GB of host memory or more).

| Worker kind | Where | Lanes | Regions | Kept queued |
|---|---|---|---|---|
| `nersc-s1` | NERSC shared GPU, 1 A100, 32 CPUs | 1 | normal | up to 6, only for regions ready now (this queue starts within about an hour) |
| `nersc-s2` | NERSC shared GPU, 2 A100s, one lane (112 GB) | 1 | normal + big | up to 5, regions ready now |
| `vista-gh` | Vista gh, 1 GH200 | 1 | all | 2, while work is open or on the way (wait was about 1.5 days on 11-18) |
| `vista-ghdev` | Vista gh-dev, 2 h | 1 | normal | 2 (started the same evening on 11-18) |
| `vista-gb` | Vista gb, 4 GB200 | 4 | all | 1 (2 running / 3 submitted per user; wait about 3.5 days on 11-18) |

Roots: NERSC `$NERSC_SCRATCH/conveyor`, Vista `$TACC_SCRATCH/conveyor` (`queue/ claims/ out/
timing/ log/ best/ slurm/ workers/`). Scenarios: NERSC `dbt/scen/<sid>` (built there), copied to Vista `dbt/scen/<sid>` with
Globus while Vista has a worker queued. Results are copied to `rsi/decam-big-tno/runs/conveyor/<unit>/{out,timing,log}`;
timing lines read `seconds rc gen site device isolation`.

A site starts **unverified**: it gets one region at a time until it finishes one. If its first region fails, the site gets
no more regions (`state.json` `site_bad`) until someone checks it and removes that entry. NERSC was verified 11-18 12:59
(salloc test: 304 s, exit 0, gen043, A100, no network). Vista is not verified yet.

**Data, unit by unit** (`units_side.py`, `unit.py`). A unit is the next 5 %, chosen by a copy of `ldt5/select_ldt.py`. The
selection is LDT-weighted, scored on exposures in the NERSC mirror or the Vista cache, and leaves out every searched
exposure. Units live in `planet-search/units/<uid>/` (`unit.json` holds the totals). Stages:

| Stage | What happens |
|---|---|
| `selected` | known-body positions start on the Mac (4 processes, Horizons); `stage` at NERSC `units/<uid>`: mirror exposures linked into 8 shares (`m0`-`m7`), the rest downloaded from NOIRLab by free xfer jobs (`p<k>`; these names go on the fill's skip list), or pulled back from the Vista cache with Globus (`v`) |
| `prep` | detection jobs per share as images arrive. They use CPU debug while NERSC's other CPU queues wait for the reboot, or CPU regular when Slurm says it starts within 30 min, after the LDT 5 % has the debug queue. When every share is final and detected and the known bodies are done: build input to `dbt/units/<uid>/in`, a build job, and the unit moves to `search` |
| `search` | rebuilt until `BUILT`; the known-body labels go to `rsi/decam-big-tno/private/truth_<uid>` and are deleted at NERSC; the unit's downloads are cached on Vista and removed at NERSC (their names go to `fill/relay/relayed.txt`); regions are searched as their scenarios appear |
| `searched` | every region finished |

The next unit is selected as soon as every unit in prep has its mirror and Vista-cache shares detected (only NOIRLab downloads left; 2026-11-19), with at most 2 units in prep and 3 selected, in prep or waiting for search. Download slots go to units first: a unit download waiting on the 15-job xfer limit makes one fill job step aside (cancelled, resubmitted, it resumes where it was).
Unit ids: `u05`, `u06`, ... (the fourth 5 % is `ldt5`, adopted for search only; its detection and build run through
`ldt5/debug_feed.py`).

## Scratch cap, releases, monitors (2026-11-19)

.

- **Cap**: `decam-fill/limits` at NERSC: downloads start a file only below 14.5 TiB and stop at 15.0 TiB (was 19.0/19.4).
- **Release** (`units_side.release`, node side `nersc/unit_release.py`, every tick, each unit in prep or search): an
  exposure's images are freed as soon as its detection list exists (or its detection failed twice). Files pulled back
  from the Vista cache (`v`) and fill files linked into `m*` shares (already on Vista through `fill/relay.py`) are deleted
  at once; mirror links are only unlinked. A unit's own NOIRLab downloads (`p*`) go to the Vista cache in Globus batches
  (`cache/bNNN.txt`, at least 100 GB unless the unit is final) and are deleted when the batch's task SUCCEEDED with every
  file copied; their names go to `fill/relay/relayed.txt`. A failed batch keeps its files (`bNNN.failed`) for the next one.
- **The third 5 % as unit `a15`** (prio 0, detected first): `units/a15/` holds its 57 regions as 62 scenarios; its
  10,160 gap exposures from the fill (30,480 files, `fill/relay/keep_a15.txt`, all on Vista) are linked as `m0`-`m7`,
  `s15_*` link to `archive15/s15_*` (mirror and archive10 exposures, already detected). Once detected, the gap files
  are deleted (about 5.9 TB).
- **Watchdog** (`watchdog.py`, run at the end of every tick; node side `nersc/rates.py`): writes `health.json` and
  appends `ALERT ...` / `OK ...` lines to `alerts.log` when a check changes (repeats hourly while it lasts): a site not
  reachable, scratch above the cap (critical above 18.5 TiB), downloads under 50 MB/s for 30 min while under the cap,
  the fill relay silent 90 min, a unit's detection without progress for 2 h, a search site held back, the climb log
  quiet 4 h. The main session watches `alerts.log` and the milestones in `conveyor.log` with a Monitor.

## Controls

- `touch conveyor/STOP`: the loop ends after its tick; jobs keep running. `python3 -I conveyor.py stop-workers` cancels all
  worker jobs (`cvw`) on both sites.
- `touch conveyor/NO_NEW_UNITS`: no further unit is selected.
- `python3 -I conveyor.py status`: program, per-unit counts, workers, staging.
- Restart the loop after editing `units_side.py` or `conveyor.py`: the running loop keeps the code it imported.

## Costs

One unit is about 46 normal regions (0.5 h each) and 10 big ones (2 h each), so about 43 GPU-hours. On NERSC shared that is
about 11 GPU node-hours (`$NERSC_GPU_ACCOUNT`). On Vista gh it is about 43 SU; on gb, about 11 SU (4 GPUs per node-hour; Vista had
1,949 SU on 11-18). Detection is about 10-15 CPU node-hours per unit (`$NERSC_CPU_ACCOUNT`). Downloads and Globus are free.

## Not automated yet

Grading against the known bodies and the lead pipeline (novelty check, lead check, LDT list) per unit. For now these are
run by hand from `runs/conveyor/<unit>/` (ldt5 labels: `harness/searchldt5/searchldt5.py truth`). The third 5 %
(archive15) is in the conveyor since 2026-11-19 as unit `a15` (see above).

2026-11-20 05:52 PST: a unit's Vista pull whose Globus submission failed is now retried like a deferred one (`units_side.py`, bak `.bak-1120-pullretry`). u16's pull had failed at submission on 11-19 18:33 ("operation timed out"), so its `v` share (14,724 files, ~4,900 exposures, ~18 % of the unit) could never become final and u16 would have waited forever; resubmitted as task 805628c9 at ~1 GB/s. u06/u16's NOIRLab gaps (11,236 files, 2.2 TB) are fetched on Lonestar6 and put into their share folders by `fill/relay_units_ls6.py` (see `fill/README.md`).

## 2026-11-20: no waiting, nothing skipped

- A first try that left NOIRLab-only exposures out of new units (flag `NERSC_ONLY`) was withdrawn the same hour, before
  any unit used it; its code is kept only as `unit.py.nerscfirst-unused`. `unit.py` is unchanged.
- Units are still chosen mostly from data at NERSC (the selector scores cells on present exposures) and keep all their
  exposures. Each new unit's NOIRLab gap is also fetched on Lonestar6, automatically since 11:44: `units_side.ls6_gap`
  (at staging) writes `units-ls6/queue/<uid>.tsv` there (list reversed) and appends name -> NERSC share to
  `fill/relay_units_ls6/dest.tsv` (same blocks as fetch.py --part; matched u17's hand-made map 10,513 of 10,513);
  `units_side.ls6_feed` (every 10 min; per unit since 13:30) gives each unit in prep its own Lonestar6 CPU job
  (normal, 7 h, 64 streams, -J units-<uid>, folder `units-ls6/<uid>/`, list `<uid>/list.tsv` = its gap minus files already
  fetched), at most `LS6_UNITS_MAX` at once (2; 3 from 17:10 for u19's 7.2 TB gap) (u17's shared job ran at 16 MB/s and would have held u18 for ~5 h); `fill/relay_units_ls6.py` (re-reads the map each cycle)
  puts each file into its NERSC share, where that share's fetcher finds it on disk.
- **Early build** (`units_side.early_build`, `nersc/ready.py`): while a unit is in prep, every 10 min a region whose every
  exposure already has a detection list is built (`unit_build.sbatch` with `L=list_early.txt`, which never writes BUILT)
  and searched at once; `conveyor.py` takes prep units with `early` set, tasks only for built scenarios. A region waits
  for all its exposures; the unit's full build later skips what exists, and the unit is marked searched only from
  stage search. The full build waits while an early build job runs. Tested on u16 (49 of 57 regions ready; the other
  8 hold an exposure without a detection list and wait for the full build) and u17 (0 ready at 11:50).
- **Builds first**: scenario builds run on a regular CPU node (1 h) when one would start within 15 min (`build_q`), else
  in the debug QOS, and no debug-queue detection is submitted while a build waits (u16's build sat pending 10:56-11:45
  behind u17's detection jobs; the debug QOS runs 2 jobs per user; resubmitted by hand on regular at 11:47).
- Detection: a share with >= 300 exposures left also gets a regular CPU job (3 h), queued even when it will not start
  soon, at most `REG_MAX = 8` at once; the choice is per share.
- `MAX_AHEAD` 3 -> 4 units waiting; NERSC shared GPU workers: normal 6 -> 8, big 5 -> 6.
- `PREP_MAX` 2 -> 3 units in selected/prep (16:00): with only NOIRLab gaps left in u17 and u18 no unit was chosen and
  the GPUs were about to idle; early builds search each unit's ready regions while its gap downloads.
- Backups: `units_side.py.bak-1120-nobottleneck`, `units_side.py.bak-1120-early`, `conveyor.py.bak-1120-nobottleneck`,
  `conveyor.py.bak-1120-early`, `nersc/unit_build.sbatch.bak-1120-early`.

## 2026-11-20 16:55: units without LDT weighting (user chose "Drop LDT, NERSC first")

The LDT-visible sky of Nov 20 - Jan 31 was used up: the next unit, u19, came out empty at 16:12 ("DECam covers 0 cells"
after the 1.5 h/night filter; 344,721 usable exposures still unsearched, 205,451 of them at NERSC). u19 was taken out of
state.json (backup state.json.bak-1120-u19empty; its folder is units/_empty_u19_20261120). `unit.py new` now copies
`conveyor/select_units.py` (ldt5's selector with `--ldt off` by default: no LDT filter or weight; cells still scored on
exposures at NERSC or cached on Vista; `--ldt weight` restores ldt5) instead of ldt5/select_ldt.py (backup
unit.py.bak-1120-noldt). Dry run before switching: 47 regions, 175 cells, 1,583 deg2, 47,671 exposures (27.6 TB), 41,307 of
them at NERSC; 13,918 not at NERSC (8.1 TB, part of it cached on Vista); six regions over 2,000 exposures; ecliptic latitude
-22 to +41 deg. About twice the exposures of an LDT-weighted unit, because the cells are denser.

2026-11-20 22:50 PST: **downloads region by region**. `nersc/prio_missing.py UNIT REGIONS_EXPOSURES N`
rewrites a unit's missing.tsv with only the files still to fetch, the regions with the fewest missing exposures first and
each region's exposures together, so the fetcher completes one region after another and each is detected, built (early
build) and searched at once (old lists `missing.tsv.bak-20261120-2250`, also copied back to `units/<u>/`). u17 (47
regions to complete, 4,561 files), u18 (17, 2,257), u19 (47, 36,783): one NERSC fetch share each (p0, `fetch_jobs` cut
to 1 in state.json, backup `state.json.bak-1120-u19n3`; p1-p4 marked SHARE_DONE) at 3 streams; Lonestar6 takes u19's list
from the end (one job). Helpers for u19's Vista share detection: `units/u19/v/helpers/` (3 CPU jobs, each a quarter by
detect_pad --only). Stream budgets and cool-downs: `fill/README.md` (same date).

2026-11-20 23:20-23:35 PST: **interactive nodes as a fast lane** (the
regular CPU and shared GPU queues gave no start estimates). A held RUNNING job named `cvi-build` (salloc -q interactive -C
cpu -N 1 ... -J cvi-build --no-shell) makes `early_build` run the build there with srun at once instead of sbatch (about
30 s per region; units_side.py, backup `.bak-1120-fastbuild`); without one it queues as before. Detection of a large
share can run as srun steps on interactive CPU nodes named `cvd-<unit>-<share>` (the conveyor then does not add its own),
and a search worker on an interactive GPU node named `cvw` (`srun --jobid=J ... bash worker.sh ROOT SCEN PY nersc 4 1
14100 900; scancel J`). Interactive allocations last at most 4 h.

2026-11-21 02:10 PST: interactive job <slurm-job-id> (cvd-u19-v, 3 CPU nodes) sat idle from 00:48, when its three detect_pad.py steps ended, until I cancelled it at 02:10 (~1.3 h x 3 nodes, about 4 CPU node-hours wasted). Lesson for the interactive fast lane: end every helper allocation with `scancel $J` after its last step, as the GPU worker already does (`...; scancel J`).

2026-11-21 03:20 PST: full unit builds also go to a held interactive build node (cvi-build) when one runs, synchronously by srun (u18: queued sbatch had no start time; on the node 24 scenarios in 7 min). conveyor.py now reloads units_side.py every tick (edits apply without a restart; backup conveyor.py.bak-1121-reload); loop restarted 03:19 (pid 84203, stdout loop.stdout). New interactive 4-GPU worker 61727460 (cvw, 4 h, releases itself after 15 idle min) for u18's 11 new scenarios.

2026-11-21 03:28 PST: new units get the region-first download order automatically: after staging, units_side.py runs nersc/prio_missing.py on NERSC with N = the unit's fetch jobs and copies the list back (u20 was reordered by hand at 03:25, 7 shares).

2026-11-21 03:37 PST: the loop restarted at 03:19 writes stdout to loop.stdout; loop.out is now a link to it (watchdog.py reads loop.out); the old log is loop.out.until-20261121-0319. The loop's own log is conveyor.log as before.
2026-11-21 12:25 PST: Vista = one gg node, 10 streams, no GPU. Cancelled the second download node 1157827 (parts 2-3 of blank_vista2) and every Vista GPU worker (gh-dev 1159199 running, gb 1158655, gh 1156063/4, gh-dev 1159247 pending); conveyor.py KINDS vista-gh/ghdev/gb max 0 (backup conveyor.py.bak-1121-gganly), loop restarted (pid 53028). Follow-on 1159284 (gg, 24 h, all four parts, afterany:1157826) keeps one node downloading after 1157826 ends. fetch_ctl.json total 10 (dl_watch cap 10), so the node shares 10 streams across its fetch copies. The cancelled worker's claim (u20 scen12) is released by the conveyor to NERSC.
2026-11-21 12:24 PST: Vista GPU searches back on: conveyor KINDS vista gh 2, gh-dev 2, gb 1 again; regions are already 1,800 s pieces (3 per 2 h gh-dev job); TACC lets one gh-dev job run per user (1 running / 3 submitted), so one runs and one waits behind it. gh-dev 1159289 running, 1159290 waiting; gh 1159295/6 and gb 1159297 queued (estimates 11-23 ~04:20 and 11-24 ~21:30 PST). Scenarios (detection lists) go NERSC -> Vista by Globus as before. $VISTA_ACCOUNT expires 2027-01-05 with 1,918 SU; gh-dev full time ~24 SU/day.

2026-11-23 01:1x: u19 had 23 regions blocked by 63 exposures whose weight (mostly) or mask file the archive could not deliver (every copy cut short or failing its checksum; re-fetch job <slurm-job-id> kept 0 of 70). conveyor/nersc/noweight_detect.py (on NERSC units/noweight_detect.py) builds a per-CCD weight from the image noise (1/(1.4826 MAD)^2) and an empty mask where the file is missing, then runs archive10/detect_pad.py process() into the share det/; stems listed in u19/p0/det/NOWEIGHT.txt. Debug job <slurm-job-id>. The conveyor early build then picks the regions up.
