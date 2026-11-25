# DECam discovery pipeline: one map (start here)

Written 2026-11-21. Every stage below has its own README
with the dated history; this page says what runs where today, how to see it, and how to stop or restart it. Keep it
current when a stage, a site or a setting changes (the dated detail still goes into the stage's own README).

Goal: find large distant bodies (about 200 km and up) in every public DECam exposure, 5 % of the sky at a time, with the
hill climb's latest best program, and vet every candidate to the end. State on 2026-11-21 12:30 PST: about 35 % of the
DECam sky searched (7 slices of 5 % plus u18), u17 finishing, u19 and u20 being prepared; 82 TB of the 154 TB not held
at NERSC still to download; 460 active candidate cards (155 of them known bodies, kept and tagged).

## Stages

| # | Stage | Where it runs | Code | Log / status | Stop |
|---|---|---|---|---|---|
| 1 | Download from NOIRLab (md5-checked, resumable) | NERSC `xfer` jobs `cvf-<unit>-<k>` per unit; Lonestar6 job `units-<unit>`; Vista one `gg` node (`dfill2_*`, list `blank_vista2.tsv`); the Mac (`blank_mac.tsv` into a 200 GB buffer on Crucial-8TB) | `fill/fetch.py` (one copy per device dir), `fill/mac_fill.sh`, `conveyor/units_side.py` (NERSC and Lonestar6 unit jobs) | each device's `log/fetch_*.log`; together: `fill/dl_watch.log` | per device: `log/STOP` beside its fetch.py; Mac `STOP_ALL` in its log dir |
| 1a | Stream control (5-10 per device, -1 and 30 min cool-down on each limit, +1 after 3 clean cycles under 200 MB/s, cap 10) | Mac | `fill/dl_watch.py loop 10`; each device's `fetch_ctl.json` (+ `.weights` at NERSC) | `fill/dl_watch.log` | `fill/dl_watch.STOP` |
| 2 | Relays | Mac drives Globus and rsync | `fill/relay_mac.py loop 2` (Mac buffer -> NERSC), `fill/relay.py loop 30` (NERSC -> Vista, then delete at NERSC), `fill/relay_ls6.py loop 30` (Lonestar6 -> Vista), `fill/relay_units_ls6.py loop 10` (Lonestar6 unit files -> NERSC), `fill/vista_known.py loop 30` (names on Vista into `fill/relay/relayed.txt`) | `fill/relay*/relay.log`, `fill/relay/vista_known.log` | `fill/relay*/STOP`; `fill/relay/STOP_VISTA_KNOWN` |
| 3 | Units: select 5 % of sky, stage, detect, build scenarios, cache images to Vista, queue searches | NERSC (CPU `debug`/`regular` detection `cvd-*`, builds on a held interactive CPU node `cvi-build` or `cvb-*` jobs) | `conveyor/conveyor.py loop 3` (reloads `units_side.py` every tick) | `conveyor/loop.stdout` (= `loop.out`), `conveyor/conveyor.log`, `conveyor/alerts.log`, `conveyor/state.json` | `conveyor/STOP` |
| 4 | Search (one GPU per region, 1,800 s per normal region, 7,200 s per region over 1,000 exposures) | NERSC `gpu_shared` workers `cvw` (kinds nersc-s1/s2, queued by the conveyor) and an interactive 4-GPU node taken by hand; Vista `gh-dev` (one runs at a time, TACC limit), `gh` x2 and `gb` x1 queued | `conveyor/worker.sh`, `nersc_worker.sbatch`, `vista_worker.sbatch`; program = climb's current best | NERSC `$NERSC_SCRATCH/conveyor/slurm/worker-*.out`; Vista `$TACC_SCRATCH/conveyor/slurm/` | `scancel` the `cvw` jobs; conveyor KINDS `max` |
| 5 | Score a unit and send its leads | Mac | `rsi/decam-big-tno/harness/searchunit/scoreunit.py score|leads <unit>`, then `runs/s<unit>/after_leadcheck.sh` | `runs/s<unit>/` | by hand |
| 6 | Lead vetting: novelty against MPC files and ITF, confirmation in other images | Mac (and NERSC for CFHT) | `rsi/candidates/novelty_watch.py`, `rsi/candidates/confirm_watch.py`; refits and archive checks by the TRKSUB01 session | `rsi/candidates/novelty_queue/RESULTS.md`, `confirm_queue/RESULTS.md` | `novelty_queue/STOP`, `confirm_queue/STOP` |
| 7 | Candidate database, catalogue page, Drive copy | Mac | `zsh rsi/decam-big-tno/candidates/refresh.sh` (zsh, not bash), then publish `progress/objects/index.html` to <private-page-url> | `candidates/index.json` | by hand |
| 8 | The program: hill climb r2 (two agents a round, graded on NERSC) | Mac + NERSC | `rsi/decam-big-tno/harness/climb.py run --run r2 ...` | `runs/r2/climb.log`, `runs/r2/review.json` | `runs/r2/STOP` |
| 9 | Pages | Mac | `pages/team-site/decam/loop.sh` (<team-site-url>, every 30 min); `fill/dash/build.py` every 10 min (sampler) feeding the DECam Data Movement page <private-page-url> (kept by a background agent) | `decam/loop.log`, `fill/dash/sampler.err` | `decam/STOP`, `fill/dash/STOP` |

Paths without a project prefix are inside `planet-search/`; `rsi/decam-big-tno/` is `ale-astronomy/rsi/decam-big-tno/`.

## Mac loops (restart from the folder named)

| Loop | Folder | Command |
|---|---|---|
| conveyor | `planet-search/conveyor` | `nohup python3 -I conveyor.py loop 3 >> loop.stdout 2>&1 &` (restart between ticks: wait until it has no child process) |
| download control | `planet-search` | `python3 -I fill/dl_watch.py loop 10 >> fill/dl_watch.log 2>&1` |
| Mac download | `planet-search` | `STREAMS=8 bash fill/mac_fill.sh >> $DATA_DRIVE/decam-fill-mac/log/mac_fill.out 2>&1` |
| relays | `planet-search` | `python3 -I fill/relay_mac.py loop 2`, `fill/relay.py loop 30`, `fill/relay_ls6.py loop 30`, `fill/relay_units_ls6.py loop 10`, `fill/vista_known.py loop 30 >> fill/relay/vista_known.log 2>&1` |
| data page sampler | `planet-search` | `while true; do [ -f fill/dash/STOP ] && break; python3 -I fill/dash/build.py > /dev/null 2>> fill/dash/sampler.err; sleep 600; done` |
| climb | `rsi/decam-big-tno` | `python3 -I harness/climb.py run --run r2 --target 75 --goal 9.0 --workers 1 >> runs/r2/climb.out` |
| lead watchers | `planet-search/rsi/candidates` | `~/anaconda3/bin/python -I novelty_watch.py`, `~/anaconda3/bin/python -I confirm_watch.py` |
| site | `pages/team-site` | `decam/loop.sh >> decam/loop.log 2>&1` |

## Sites and limits

- **NERSC** (needs the user's login once a day: `bash $HPC_TOOLS/nersc-sshproxy-legacy.sh -u $NERSC_USER`
  (the sshproxy binaries fail on this Mac), then a master on `$SSH_CM_DIR/nersc-$NERSC_USER.sock` opened in the user's own
  Terminal, never through a `!` prompt here; `$HPC_TOOLS/nersc-keepmaster.sh` reopens a dropped master
  while the 24 h certificate is valid, log `$HPC_TOOLS/logs/nersc-keepmaster.log`). While the master is gone every NERSC step pauses; jobs
  already on NERSC keep running. Interactive QOS: 4 CPU nodes and 1 GPU node per user at a time. Scratch kept under
  15 TiB (fill limits 14.5/15.0). GPU pool `$NERSC_GPU_ACCOUNT` is shared by 16 people; detection runs on CPU (`$NERSC_CPU_ACCOUNT`).
- **Vista** (`$VISTA_ACCOUNT`, 1,918 SU on 11-21, expires 2027-01-05; the user is not the PI): one `gg` download node
  at 10 streams (follow-on job queued after each); GPU: `gh-dev` 2 h jobs start in minutes (1 running per user),
  `gh` waits about 2 days, `gb` about 3-4 days. Scratch purges files not read for 10 days.
- **Lonestar6**: one unit download job at 10 streams (`LS6_UNITS_MAX` 1).
- **Mac**: 8-9 streams; heavy compute never here. **GCP**: cannot reach the NOIRLab archive; no downloads or relays there.

## Checks after a pause (e.g. a NERSC login gap)

1. `ssh -S $SSH_CM_DIR/nersc-$NERSC_USER.sock -O check $NERSC_USER@perlmutter.nersc.gov`.
2. `squeue -u $NERSC_USER` at NERSC: `cvw` workers, `cvi-build`, `cvd-*`, `cvf-*`; an expired interactive GPU node needs a
   new `salloc -q interactive -C gpu ... -J cvw --no-shell` and a worker started detached (setsid nohup srun ...).
3. `tail conveyor/loop.stdout` shows ticks; `tail fill/relay_mac/relay.log` shows uploads; `tail fill/dl_watch.log`.
   The Mac relay can hang after a lost master (2026-11-21: silent from 14:26, the Mac buffer filled and its download
   stopped): if relay.log has no new line for 10 minutes while NERSC is up, restart `fill/relay_mac.py loop 2`.
4. The climb retries grading by itself every 10 minutes.
