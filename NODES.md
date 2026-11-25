# Nodes

One section per node of the pipeline, in the order data flows. Each says what the node does, where its code is (under `nodes/`), how it was run, its inputs and outputs (file formats and JSON field names, read from the code), its settings, where it ran, what is hard-coded in it, and what one would change to swap it. The same facts in machine form: `pipeline.yaml`. Everything is the code as it ran on 2026-11-24: absolute paths, cluster logins, SSH control sockets, Globus endpoints and site URLs are written into the code; nothing was refactored.

Conventions used throughout: a *region* is one search area (up to 6 cells of 3 x 3 deg) with every public DECam exposure of 2013-2026; a *unit* is one 5 % of the sky (tens of regions); a *scenario* is a region's detection lists in the form the search program reads, named by an opaque `sid = "s" + sha1(rid)[:8]`; a *lead* is a confident candidate that matches no known body; a *card* (`C###`) is a promising body in the candidate database. `TRKSUB01`..`03` are placeholders for our MPC tracklet names; in comments, `TRKSUB01` also names the agent session that ran the candidate checks.

| # | node | stage | site |
|---|---|---|---|
| 1 | [select-regions](#select-regions) | 1. archive pipeline | Mac (local metadata only) |
| 2 | [download](#download) | 1. archive pipeline | NERSC xfer jobs, Lonestar6 job, Vista one gg node, the Mac (buffer on an external drive) |
| 3 | [relay](#relay) | 1. archive pipeline | Mac (drives Globus CLI and rsync) |
| 4 | [detect](#detect) | 1. archive pipeline | NERSC CPU nodes (account $NERSC_CPU_ACCOUNT; GPU kept for the search), earlier TACC Vista/Lonestar6 |
| 5 | [unit-build](#unit-build) | 1. archive pipeline | NERSC CPU (build jobs, or a held interactive CPU node), earlier Vista gg; known positions on the Mac (JPL Horizons) |
| 6 | [conveyor](#conveyor) | 1. archive pipeline | Mac, driving NERSC and Vista over SSH masters and Globus |
| 7 | [search-worker](#search-worker) | 1. archive pipeline | NERSC gpu_shared / interactive A100 nodes; Vista gh-dev, gh (GH200), gb (GB200) |
| 8 | [grade](#grade) | 2. scoring | programs run on NERSC (A100) or Vista (GB200); scoring on the Mac (numpy) |
| 9 | [rsi-loop](#rsi-loop) | 3. program writing (task 1) | agents on the Mac (sandbox-exec) or the GCP VM; grading on NERSC A100 (also Lonestar6, Mac early on) |
| 10 | [climb](#climb) | 3. program writing (task 2) | Mac (loop), trial host GCP VM (Harbor trials, Docker), NERSC (grading) |
| 11 | [tailor](#tailor) | 3. program writing (task 2) | trial host (agent), optional Vista/Lonestar6 grading |
| 12 | [agents](#agents) | cross-cutting | Mac; trial host = GCP VM with Docker and Harbor (run_trial.sh lives in a separate private repository, not included) |
| 13 | [lead-check](#lead-check) | 4. vetting | Mac, reading scenarios on NERSC or Vista over the SSH master |
| 14 | [arc-refit](#arc-refit) | 4. vetting | Mac + remote reads on NERSC/Vista |
| 15 | [novelty](#novelty) | 4. vetting | Mac (single-threaded, one at a time); SkyBoT and MPC web services |
| 16 | [confirm](#confirm) | 4. vetting | Mac (DECam), NERSC CPU (CFHT) |
| 17 | [image-tests](#image-tests) | 4. vetting | Mac; passes ran as relay agents (launch.sh) |
| 18 | [candidate-db](#candidate-db) | 5. records | Mac |
| 19 | [observe](#observe) | 5. records | Mac |
| 20 | [package](#package) | 5. records | Mac (Drive for Desktop folder) |
| 21 | [pages](#pages) | 5. records | Mac; published to claude.ai artifacts and a Cloudflare site |
| 22 | [archive-plans](#archive-plans) | 1. archive pipeline (planned) | NERSC / Lonestar6 (planned) |
| 23 | [lib-linker](#lib-linker) | library | anywhere (numpy, scipy, astropy; torch/triton for link_gpu) |

## select-regions

Choose the next ~5 % of DECam's sky to search: 3 x 3 deg cells scored by observing seasons, ecliptic latitude, Rubin coverage of where the bodies have moved, DES share and galactic plane; cells merged into regions of <= 6 cells.

**Code** (`nodes/select-regions/`):

- `select-regions/archive5/select_regions.py (first 5 %)`
- `select-regions/ldt5/select_ldt.py (fourth 5 %, LDT-weighted)`
- `select-regions/conveyor/select_units.py (the conveyor's selector since 2026-11-20)`
- `select-regions/linker/footprint/*.py (archive inventory: exposures, cones, windows, seasons, costs)`

**Run as:** `python3 -I archive5/select_regions.py`; `conveyor/unit.py new UID (copies the selector into units/UID/ and runs it)`

**Inputs**

- bulk manifests linker/data_real/decam_bulk/{ls6,vista}.tsv: TSV md5, size, name, tag
- DECam exposure metadata (noirlab_decam_meta) and Rubin visits (rubin_visits): JSON/TSV, from linker/footprint/fetch_*.py
- --present: exposures already at NERSC or cached on Vista; --searched: exposures already searched: stem lists

**Outputs**

- `regions.json`: {asof, rule, frac, margin, cover_deg2, chosen_deg2, n_cells, unique_exposures, unique_tb, regions: [{id, label, ra, dec, deg2, cells, exposures, tb, first_year, last_year, seasons, ...}]}
- `regions_exposures.json`: {rid: [exposure stems]}
- `manifests/<rid>.tsv`: fetch.py manifest: md5, size, name, tag

**Settings:** --frac (sky share, 0.05), --cover (DECam cover, 31,532 deg2), --margin (deg), --ldt weight|off, --ldt-from/--ldt-to/--ldt-min-h

**Ran on:** Mac (local metadata only)

**Hard-coded:** paths 7 (`~/anaconda3/bin/python`, `/global/cfs/cdirs/cosmo/staging/decam`); hosts 5 (`astroarchive.noirlab.edu`); clusters 45 (`NERSC`, `GB200`, `A100`).

**To swap this node:** Another survey: rewrite the inventory (linker/footprint/fetch_*.py against that archive's API) and keep the cell score; another goal (e.g. a follow-up telescope): change the weights in the score function (the LDT weight is the worked example).

**Feeds:** [conveyor](#conveyor)

## download

Bulk, resumable, md5-checked download of DECam InstCal triplets (image, weight, mask) from the NOIRLab Astro Data Archive, several copies per device sharing one manifest, with a device-wide stream budget and cool-down on throttling.

**Code** (`nodes/download/`):

- `download/fill/fetch.py`
- `download/fill/dl_watch.py (monitor and stream controller)`
- `download/fill/make_blank.py (what is still missing)`
- `download/fill/mac_fill.sh, nersc_fill.sbatch, vista_fill.sbatch`
- `download/bulk/*.py, *.sbatch (manifests, first bulk fetch)`
- `download/conveyor/nersc/unit_fetch.sbatch, download/ldt5/ldt5_fetch.sbatch (per-unit downloads)`

**Run as:** `python3 -I fetch.py MANIFEST.tsv DEST [--part k/N] [--streams 32] [--hours 47] [--max-tb 0]`; `python3 -I fill/dl_watch.py loop 10`

**Inputs**

- manifest: TSV: md5, size, name, tag per line, in download order (bulk/make_manifest.py, fill/make_blank.py)
- fetch_ctl.json (optional, beside fetch.py): JSON {total, pause_until, set, note}: device stream budget, written by dl_watch.py

**Outputs**

- `DEST/img/<YYMMDD>/<name>.fits.fz`: DECam InstCal ooi/oow/ood files
- `DEST/log/done_*.txt, fetch_*.log`: names finished (read by the relays); progress lines with MB/s, files, errors
- `dl_watch.json`: per-cycle device rates and control events ('ctl')

**Settings:** --streams, --part, --hours, --max-tb; fetch_ctl.json; STOP files (DEST/log/STOP)

**Ran on:** NERSC xfer jobs, Lonestar6 job, Vista one gg node, the Mac (buffer on an external drive)

**Hard-coded:** paths 57 (`$NERSC_SCRATCH/decam-fill/limits`, `$NERSC_SCRATCH/dbt/env/bin/python`, `$NERSC_SCRATCH/decam-fill`); accounts 79 (`$NERSC_USER`, `$TACC_USER`, `$TACC_UID`); hosts 10 (`astroarchive.noirlab.edu`, `perlmutter.nersc.gov`); clusters 52 (`NERSC`, `Vista`, `regular`); ssh_sockets 4 (`$SSH_CM_DIR/nersc-$NERSC_USER.sock`, `$SSH_CM_DIR/ls6-$TACC_USER.sock`, `$SSH_CM_DIR/vista-$TACC_USER.sock`).

**To swap this node:** Another archive: replace the URL builder and md5 source in fetch.py (it is standard library only); the manifest format stays. Another cluster: copy fetch.py there and add a job script like nersc_fill.sbatch.

**Feeds:** [relay](#relay), [detect](#detect)

## relay

Move finished downloads between sites: Mac buffer -> NERSC (rsync over the SSH master), NERSC -> Vista and Lonestar6 -> Vista/NERSC (Globus tasks, checksum-verified, source deleted only after SUCCEEDED).

**Code** (`nodes/relay/`):

- `relay/fill/relay.py (NERSC -> Vista)`
- `relay/fill/relay_mac.py (Mac -> NERSC)`
- `relay/fill/relay_ls6.py, relay_units_ls6.py`
- `relay/fill/vista_known.py (what is on Vista)`

**Run as:** `python3 -I fill/relay.py loop 30`; `python3 -I fill/relay_mac.py loop 2`

**Inputs**

- each download part's log/done_*.txt: file names finished and md5-checked

**Outputs**

- `relay/tasks/<task id>.json`: Globus task record (files carried, status)
- `relay/relayed.txt`: names now on Vista
- `relay*/relay.log`: one line per cycle

**Settings:** loop minutes; STOP files in each relay folder; Globus endpoint ids and paths in the code

**Ran on:** Mac (drives Globus CLI and rsync)

**Hard-coded:** paths 22 (`$TACC_SCRATCH/decam-fill/img`, `$SSH_CM_DIR/ls6-$TACC_USER.sock`, `$NERSC_SCRATCH/decam-fill`); accounts 33 (`$TACC_USER`, `$NERSC_USER`, `$TACC_UID`); hosts 4 (`perlmutter.nersc.gov`); clusters 74 (`NERSC`, `Vista`, `Lonestar6`); ssh_sockets 5 (`$SSH_CM_DIR/ls6-$TACC_USER.sock`, `$SSH_CM_DIR/nersc-$NERSC_USER.sock`, `$SSH_CM_DIR/vista-$TACC_USER.sock`).

**To swap this node:** Another transfer service: replace the globus calls (submit, task status) in relay.py; the queue logic (done -> in flight -> relayed -> delete) stays.

**Feeds:** [detect](#detect)

## detect

Single-exposure detection: per CCD, block-median background, Gaussian matched filter at the CCD's FWHM, S/N >= 4.5 peaks with sub-pixel centres and RA/Dec from the CCD's TPV WCS. No stacking; static sources are kept.

**Code** (`nodes/detect/`):

- `detect/archive5/tacc/detect.py`
- `detect/archive10/nersc/detect_pad.py (weight map 2 px larger than the image)`
- `detect/conveyor/nersc/unit_detect.sbatch, noweight_detect.py`
- `detect/archive5/tacc/job*.sbatch, detect/archive10/nersc/*.sbatch`
- `detect/ldt5/l5_detect*.sbatch, l5_farm.py`

**Run as:** `python3 -I detect.py IMG OUT [--workers 32] [--snr 4.5] [--watch] [--only STEM,...]`

**Inputs**

- IMG/<YYMMDD>/*_ooi_*, *_oow_*, *_ood_*: DECam InstCal FITS (fpack)

**Outputs**

- `OUT/<YYMMDD>/<stem>.npz`: columns ra, dec (deg), flux, ferr, snr, x, y, ccd, sharp, flag (1 near saturation, 2 masked, 4 edge); exposure values stem, expnum, mjd_mid, band, exptime, magzero, fwhm, airmass, ra0, dec0, wcscal

**Settings:** --snr, --workers; mask handling (CP mask value 7 marks moving bodies: dropped as it ran)

**Ran on:** NERSC CPU nodes (account $NERSC_CPU_ACCOUNT; GPU kept for the search), earlier TACC Vista/Lonestar6

**Hard-coded:** paths 30 (`$NERSC_SCRATCH/archive10`, `$NERSC_SCRATCH/dbt/env/bin/python`, `$NERSC_SCRATCH/ldt5`); accounts 53 (`$NERSC_USER`, `$NERSC_CPU_ACCOUNT`, `$NERSC_GPU_ACCOUNT`); hosts 5 (`astroarchive.noirlab.edu`, `ls6.tacc.utexas.edu`, `vista.tacc.utexas.edu`); clusters 32 (`NERSC`, `regular`, `debug`).

**To swap this node:** Another detector (e.g. SExtractor or a CNN): any program that writes the same .npz columns per exposure plugs in; unit-build reads only those columns.

**Feeds:** [unit-build](#unit-build)

## unit-build

Turn one region's detection lists into a scenario (what a search program reads) and its labels (known bodies placed by JPL Horizons in every exposure, detections within 2" labelled).

**Code** (`nodes/unit-build/`):

- `unit-build/decam-big-tno/bin/build_scen.py (+ build.sbatch, fix_pointing.py)`
- `unit-build/archive5/link/known_pos.py, unit-build/ldt5/known_pos.py`
- `unit-build/conveyor/nersc/unit_build.sbatch, unit_farm.py`
- `unit-build/decam-big-tno/harness/searchunit/{split_region,thin_region}.py`
- `unit-build/decam-big-tno/harness/{open_data,relabel_faint,body_props,body_props_par}.py (the r2 agents' open, labelled regions)`

**Run as:** `python3 -I known_pos.py REG_A,REG_B [--hmax 8]`; `PAIRS=RID:SID,... sbatch build.sbatch (runs build_scen.py next to the data)`

**Inputs**

- detection lists: detect's .npz per exposure
- known/<rid>.json: {rid, bodies: [{name, pdes, H, first_obs, pos: {stem: [ra, dec]}}]}
- regions_exposures.json: {rid: [stems]}

**Outputs**

- `SCEN/<sid>/det_*.npy`: det_ra, det_dec (float64 deg), det_flux, det_ferr, det_snr, det_sharp, det_x, det_y (float32), det_ccd (int16), det_flag (uint8), det_exp (int32 row of the exposure table)
- `SCEN/<sid>/exposures.json`: [{stem, mjd_mid, band, exptime, magzero, fwhm, airmass, ra0, dec0, obs_xyz}]
- `SCEN/<sid>/region.json`: {centre, radius, seasons}
- `TRUTH/<sid>.npz (sealed, never shipped)`: det_body, known table (name, H, weight), nights per body

**Settings:** sid = 's' + sha1(rid)[:8]; --hmax 8; --planted N (0 as it ran)

**Ran on:** NERSC CPU (build jobs, or a held interactive CPU node), earlier Vista gg; known positions on the Mac (JPL Horizons)

**Hard-coded:** paths 28 (`$SSH_CM_DIR/nersc-$NERSC_USER.sock`, `$NERSC_SCRATCH/dbt/scen`, `$TACC_SCRATCH/decam-fill/img`); accounts 48 (`$TACC_UID`, `$TACC_USER`, `$NERSC_USER`); hosts 8 (`perlmutter.nersc.gov`, `ssd.jpl.nasa.gov`, `astroarchive.noirlab.edu`); clusters 40 (`NERSC`, `Vista`, `A100`); ssh_sockets 4 (`$SSH_CM_DIR/nersc-$NERSC_USER.sock`).

**To swap this node:** Another truth source (e.g. a different known-object catalogue): replace known_pos.py's SBDB + Horizons calls, keep its JSON; another detection format: adapt the reader in build_scen.py.

**Feeds:** [search-worker](#search-worker)

## conveyor

The Mac coordinator of continuous discovery: unit after unit (5 % of the sky each), select -> stage -> download -> detect -> build -> one search task per region, mirrored between NERSC and Vista, always with the hill climb's current best program; keeps worker jobs queued; retries failed regions.

**Code** (`nodes/conveyor/`):

- `conveyor/conveyor/conveyor.py (the loop)`
- `conveyor/conveyor/units_side.py (per-unit stages, reloaded every tick)`
- `conveyor/conveyor/unit.py (new / known / stage a unit)`
- `conveyor/conveyor/{watchdog,site_scan}.py`
- `conveyor/conveyor/nersc/*.py`
- `conveyor/PIPELINE.md (the live map as it ran)`

**Run as:** `nohup python3 -I conveyor.py loop 3 >> loop.stdout 2>&1 &`

**Inputs**

- rsi/decam-big-tno/runs/r2/state.json: the climb's best generation and its submission path
- units/<uid>/regions_search.json: [{rid, sid, n_exp, ldt_h}]
- site scans: claims, out/, timing/, worker files at each site

**Outputs**

- `state.json`: per unit: stage, regions, tasks, claims
- `ROOT/queue/<prio>_<unit>_<sid>`: one line: unit sid budget_s class
- `ROOT/best/<gen>/sub, ROOT/best/CURRENT`: the program workers copy
- `conveyor.log, alerts.log, health.json`: text / JSON

**Settings:** KINDS (worker kinds and caps per site), budgets 1,800 s (7,200 s over 1,000 exposures), STOP file

**Ran on:** Mac, driving NERSC and Vista over SSH masters and Globus

**Hard-coded:** paths 34 (`$SSH_CM_DIR/nersc-$NERSC_USER.sock`, `$NERSC_SCRATCH/dbt/env/bin/python`, `$NERSC_SCRATCH/decam-fill`); accounts 65 (`$NERSC_USER`, `$TACC_USER`, `$TACC_UID`); hosts 15 (`perlmutter.nersc.gov`, `vista.tacc.utexas.edu`); clusters 119 (`NERSC`, `Vista`, `debug`); ssh_sockets 10 (`$SSH_CM_DIR/nersc-$NERSC_USER.sock`, `$SSH_CM_DIR/vista-$TACC_USER.sock`, `$SSH_CM_DIR/ls6-$TACC_USER.sock`).

**To swap this node:** Another cluster: add a site entry (host, socket, user, scratch root, Python, worker sbatch) in conveyor.py's SITES and a worker job script; another program source: change where 'best' is read (one function).

**Feeds:** [download](#download), [search-worker](#search-worker)

## search-worker

Run the search program on one region per GPU: claim a task, copy the current best program, run it without network (unshare -rn where allowed) within its budget, write candidates and timings.

**Code** (`nodes/search-worker/`):

- `search-worker/conveyor/worker.sh, nersc_worker.sbatch, vista_worker.sbatch`
- `search-worker/decam-big-tno/harness/search*/ (the campaigns before the conveyor: search38, search10a/b, searchldt5, searcha15)`

**Run as:** `bash worker.sh ROOT SCEN PY SITE LANES BIGOK WALL_S [IDLE_S] (inside an allocation)`

**Inputs**

- ROOT/queue/*: unit sid budget_s class
- SCEN/<sid>/: unit-build's scenario
- ROOT/best/CURRENT: program folder

**Outputs**

- `ROOT/out/<unit>/<sid>/candidates.json`: [{det: [row ids], p}] (the program's output; top 2,000 read)
- `ROOT/timing/<unit>/<sid>`: 'seconds rc gen site device isolation'
- `ROOT/log/<unit>/<sid>.log`: program stdout/stderr

**Settings:** lanes per node, big-region permission (host memory), wall, idle time

**Ran on:** NERSC gpu_shared / interactive A100 nodes; Vista gh-dev, gh (GH200), gb (GB200)

**Hard-coded:** paths 35 (`$NERSC_SCRATCH/dbt`, `~/anaconda3/bin/python`, `$NERSC_SCRATCH/archive10`); accounts 46 (`$NERSC_USER`, `$NERSC_CPU_ACCOUNT`, `$NERSC_GPU_ACCOUNT`); clusters 88 (`NERSC`, `regular`, `A100`).

**To swap this node:** Another program: anything that runs as finder.py --scen DIR --out DIR --budget-s S and writes candidates.json; another cluster: a new sbatch that calls worker.sh with its Python and scratch paths.

**Feeds:** [grade](#grade)

## grade

Score programs (climb rounds) and searched units: global ranking by p over all regions, K = 2 x findable known bodies scored; found = >= 80 % of a candidate's detections on one body and >= 5 of its nights; weight 2 for H <= 7, 1 for 7 < H <= 8. Also per-body feedback for the agents and the leads (confident unlabelled candidates).

**Code** (`nodes/grade/`):

- `grade/decam-big-tno/harness/grade.py`
- `grade/decam-big-tno/harness/oracle.py (perfect 1.0, low-p junk 1.0, high-p junk 0.0)`
- `grade/decam-big-tno/harness/nersc_grade.py + nersc/grade_node.sh (one interactive 4 x A100 node)`
- `grade/decam-big-tno/harness/vista/run_gb.sbatch (GB200, before 2026-11-13)`
- `grade/decam-big-tno/harness/gpu_logs.py`
- `grade/decam-big-tno/harness/searchunit/scoreunit.py (score a conveyor unit, send its leads)`

**Run as:** `python3 -I harness/scoreunit.py score|leads UNIT`; `python3 -I harness/nersc_grade.py JOB TAG=SUBDIR=GRADEDIR [...] [--budget 3600]`

**Inputs**

- OUT/<sid>/candidates.json: [{det, p}]
- timing: seconds per region
- private/truth*/<sid>.npz: labels (sealed)

**Outputs**

- `GRADE/public.json`: {reward, gpu_hours, status, seconds}
- `GRADE/grade.json (private)`: {reward, R_known, weights, gpu_hours, slots, candidates, cut_p, per_scenario, feedback, found, leads}
- `GRADE/bodies.md (private)`: per findable body: found / below_cut / few_nights / mixed / in no candidate

**Settings:** PER_SCEN 2,000; K = 2 x findable; found threshold 80 % and 5 nights; weights by H

**Ran on:** programs run on NERSC (A100) or Vista (GB200); scoring on the Mac (numpy)

**Hard-coded:** paths 9 (`$SSH_CM_DIR/nersc-$NERSC_USER.sock`, `~/anaconda3/bin/python`, `$NERSC_SCRATCH/dbt`); accounts 16 (`$NERSC_USER`, `$TACC_UID`, `$TACC_USER`); hosts 1 (`perlmutter.nersc.gov`); clusters 21 (`A100`, `NERSC`, `Vista`); ssh_sockets 2 (`$SSH_CM_DIR/nersc-$NERSC_USER.sock`).

**To swap this node:** Another grader: keep candidates.json and public.json; change the match rule or weights in grade.py; the climb reads only public.json plus the feedback fields it shows the agents.

**Feeds:** [climb](#climb), [lead-check](#lead-check)

## rsi-loop

Task 1 (planet-search): the first RSI loop. Each generation is one agent session that cannot score anything; it gets every earlier program, its notes and scalar reward, writes a finder for raw DECam images of one patch over days to weeks, and is graded on 10 sealed fields (A100, 900 s each). Its best program (r2 gen015) is the reference solver.

**Code** (`nodes/rsi-loop/`):

- `rsi-loop/planet-search/rsi/harness/rsi_loop.py (outer loop, agent sandbox)`
- `rsi-loop/planet-search/rsi/harness/grade.py (sandboxed grader)`
- `rsi-loop/planet-search/rsi/harness/task/ (instruction, starter, reference linker, tools/check.py, graders.json)`
- `rsi-loop/planet-search/rsi/harness/ls6/ (Lonestar6 and NERSC graders, race.py)`
- `rsi-loop/planet-search/rsi/harness/programs_git.py, campaign_sync.py, gcp_agent.py`
- `rsi-loop/planet-search/rsi/bin/ (scenario selection, fetch, truth)`
- `rsi-loop/planet-search/rsi/baselines/kbmod/ (KBMOD baseline: 0.015)`
- `rsi-loop/planet-search/rsi/REFERENCE_SOLVER.md`

**Run as:** `python3 -I rsi_loop.py gen RUN [--n 1] [--model claude-opus-5-5] [--effort xhigh] [--hours 3]`

**Inputs**

- scenario folder: exposures.csv (exp_id, image, weight, mask, mjd_mid, band, magzero, seeing, obs_x/y/z, ...), exposures/*.fits.fz, meta.json
- history/: every earlier program, NOTES.md, reward.json (reward, per_scenario seconds)

**Outputs**

- `OUT.json (program)`: {candidates: [{epoch_mjd, state [x,y,z,vx,vy,vz], p, mag}]} (<= 200 per scenario)
- `runs/<run>/gens/<gen>/reward.json`: {reward, status, finder_seconds_total, graded_on, budget_s}
- `programs.git`: one commit per generation; tags genNNN, record-<reward>-genNNN

**Settings:** runs/<run>/config.json: eval (sealed sids), k_per_scen, cap_s, start, last_gen, agent_on (mac | gcp)

**Ran on:** agents on the Mac (sandbox-exec) or the GCP VM; grading on NERSC A100 (also Lonestar6, Mac early on)

**Hard-coded:** paths 23 (`~/anaconda3/bin/python`, `$SSH_CM_DIR/ls6-$TACC_USER.sock`, `$SSH_CM_DIR/nersc-$NERSC_USER.sock`); accounts 14 (`$TACC_USER`, `$NERSC_USER`, `$TACC_UID`); hosts 9 (`$VM_HOST`, `ssd.jpl.nasa.gov`, `astroarchive.noirlab.edu`); clusters 110 (`A100`, `LS6`, `Lonestar6`); ssh_sockets 4 (`$SSH_CM_DIR/ls6-$TACC_USER.sock`, `$SSH_CM_DIR/nersc-$NERSC_USER.sock`); models 5 (`Opus 5.5`, `claude-opus-5-5`).

**To swap this node:** Another model: --agent/--model/--effort; another sandbox: sandbox_profile() (macOS) or gcp_agent.py (container); another grader site: graders.json + ls6/*.py.

**Feeds:** [climb](#climb), [novelty](#novelty)

## climb

Task 2 (decam-big-tno): the hill climb whose best program runs the archive search. Each round one or two agents (Claude Fable 5.2 via Claude Code; GPT 6 / 6.2 Astra via Codex) get every earlier program with its rewards, per-body feedback and run logs, write finder.py in a Harbor trial (CPU only, 2 h, no web), and the program is graded on 4 regions (one A100 each, 1,800 s). Rewards: recall and GPU-hours, never mixed.

**Code** (`nodes/climb/`):

- `climb/decam-big-tno/harness/climb.py (run, status, starter)`
- `climb/decam-big-tno/harness/build_task_r2.py, eng_history.py`
- `climb/decam-big-tno/harness/task/, task_r2/ (instructions, starter, tools/check.py, subset.py)`
- `climb/decam-big-tno/harbor/decam-big-tno{,-r2}/ (the Harbor tasks: instruction, task.toml, Dockerfile, tests, solution)`
- `climb/decam-big-tno/progress/wait_climb.py`

**Run as:** `python3 -I harness/climb.py run --run r2 --target N --workers 1|2`; `python3 -I harness/climb.py status --run r2`

**Inputs**

- runs/<run>/config.json: truth, budget_s, vm_base, design, open, agent_override, prior, pairs_from, wait_grade
- history/ (built per round): SUMMARY.md, genNNN/{code, NOTES.md, public.json, grade.json, bodies.md, out/<sid>/candidates.json}, gpu_logs/, engineering/, prior_r1/
- /app/open (r2): all 23 regions' scenarios with labels (name, H, weight, nights), read-only

**Outputs**

- `runs/<run>/state.json`: {target, best: {gen, reward, gpu_hours, hash, submission}, gens: {genNNN: {phase, status, reward, gpu_hours, agent, submission, hash, job}}}
- `runs/<run>/gens/<gen>/{job-aK/, grade/}`: the Harbor job (artifact submission/) and the grade
- `climb.log`: one line per event

**Settings:** PAIR (models per round), AGENT_S 7,200 s, NO_BASE, BUDGET_S, NET and DESIG scans (network code or a designation scores 0), GO and STOP files

**Ran on:** Mac (loop), trial host GCP VM (Harbor trials, Docker), NERSC (grading)

**Hard-coded:** paths 8 (`~/anaconda3/bin/python`, `$VM_WORK`, `$SSH_CM_DIR/vista-$TACC_USER.sock`); accounts 5 (`$TACC_USER`, `$TACC_UID`, `$VISTA_ACCOUNT`); hosts 5 (`$VM_HOST`, `vista.tacc.utexas.edu`); clusters 29 (`Vista`, `NERSC`, `GH200`); ssh_sockets 1 (`$SSH_CM_DIR/vista-$TACC_USER.sock`); models 6 (`Astra`, `claude-fable-5-2`, `Fable 5.2`).

**To swap this node:** Another model: config agent_override {agent, model, name, effort} or PAIR; another grader site: nersc_grade.py is one module (Vista's run_gb.sbatch was the first); another task: build_task_r2.py + task_r2/instruction.md.

**Feeds:** [search-worker](#search-worker), [tailor](#tailor)

## tailor

Port the climb's best program to one GPU device for least GPU time (one 2 h agent trial per device, CPU-only test).

**Code** (`nodes/tailor/`):

- `tailor/decam-big-tno/tailor/tailor.py, stage.py`
- `tailor/decam-big-tno/tailor/targets.json (device facts), agents.json (pluggable agents)`
- `tailor/decam-big-tno/tailor/sites/tacc_grade.sbatch, ls6_env.sh (optional GPU grading)`

**Run as:** `python3 -I tailor/tailor.py port --target gb200-vista --agent astra [--dry]`

**Inputs**

- runs/r2/state.json: best program
- targets.json: per device: host, partition, GPU facts, BUDGET_S

**Outputs**

- `tailor/ports/<target>/`: program + BUDGET_S.txt, PORT.md, port.json
- `tailor/efficiency.json`: table

**Settings:** --target, --agent (agents.json keys: astra, fable)

**Ran on:** trial host (agent), optional Vista/Lonestar6 grading

**Hard-coded:** paths 21 (`$TACC_SCRATCH/dbt/scen`, `$TACC_SCRATCH/dbt/hw`, `$NERSC_SCRATCH/dbt/scen`); accounts 41 (`$TACC_USER`, `$TACC_UID`, `$NERSC_USER`); hosts 3 (`vista.tacc.utexas.edu`, `ls6.tacc.utexas.edu`); clusters 64 (`NERSC`, `Lonestar6`, `A100`); ssh_sockets 3 (`$SSH_CM_DIR/vista-$TACC_USER.sock`, `$SSH_CM_DIR/ls6-$TACC_USER.sock`); models 6 (`Astra`, `openai/gpt-6.2-astra`, `claude-fable-5-2`).

**To swap this node:** Another device: one entry in targets.json; another agent: one entry in agents.json.

## agents

How agents were run. Program-writing agents: Harbor trials on the trial host (climb, tailor) or sandboxed Claude Code sessions on the Mac / GCP VM (rsi-loop). Analysis agents: headless Claude Code through the local model relay (launch.sh pattern: task file in, trace and a <= 25-line hand-back out).

**Code** (`nodes/agents/`):

- `agents/decam-big-tno/harness/agents/launch.sh`
- `agents/planet-search/rsi/candidates/allarch/doublecheck/launch.sh`
- (agent launch code inside climb.py: launch(); rsi_loop.py: sandbox_profile(), run_agent(); tailor.py; gcp_agent.py)

**Run as:** `bash launch.sh <pass>   (MODEL=claude-opus-5-5 EFFORT=xhigh by default)`

**Inputs**

- <pass>.task.md: the brief

**Outputs**

- `<pass>.trace.jsonl`: stream-json trace (not shipped)
- `<pass>.handback.md`: <= 25 lines

**Settings:** model and effort env vars; the relay at $RELAY_ADDR holds the keys ($RELAY_CLI reads them; never in this repo)

**Ran on:** Mac; trial host = GCP VM with Docker and Harbor (run_trial.sh lives in a separate private repository, not included)

**Hard-coded:** paths 4 (`$RELAY_CLI`, `$ALE_HOME/rsi/decam-big-tno`, `$ALE_HOME/planet-search/rsi/candidates/allarch`); models 5 (`claude-opus-5-5`).

**To swap this node:** Another model or harness: climb's PAIR / agent_override and tailor's agents.json name a Harbor agent (claude-code | codex) and a model string; rsi_loop.py takes --agent/--model; launch.sh takes MODEL/EFFORT.

**Feeds:** [climb](#climb), [rsi-loop](#rsi-loop), [image-tests](#image-tests)

## lead-check

Every lead of a graded round or searched unit goes to the novelty queue; then an in-data repeat test: fit one orbit, predict every other exposure of the region where it should be detectable, count detections there against a 30"-shifted null. Plus the fixed-source test (is something at the same sky place at other epochs?).

**Code** (`nodes/lead-check/`):

- `lead-check/decam-big-tno/harness/leads.py`
- `lead-check/decam-big-tno/harness/lead_check.py`
- `lead-check/decam-big-tno/harness/static_test.py`
- `lead-check/decam-big-tno/harness/lead_check_rederive_v5.py`

**Run as:** `python3 -I harness/leads.py RUN GEN [--max 200]`; `DBT_SITE=nersc python -I harness/lead_check.py RUN GEN`

**Inputs**

- grade.json leads: [{sid, rank, p, det}]
- the region's scenario (remote read on NERSC/Vista): det_*.npy, exposures.json

**Outputs**

- `novelty_queue/inbox/<id>.json`: {id, points: [[mjd_utc, ra, dec, sigma, site]], ...}
- `runs/<run>/leads_sent.json`: ledger of sent leads with re-finds ('seen')
- `runs/<run>/lead_check/<tag>.json`: [{id, region, a, e, i, r, H, D_km, expected, found, extra, extra_found, null_hits, rejected_*, hit_nights, verdict, check}]
- `runs/<run>/static/<id>.json`: per arc row: sources at the same place in other-epoch exposures

**Settings:** DBT_SITE (vista | nersc: where the scenarios are read), verdict thresholds in the code

**Ran on:** Mac, reading scenarios on NERSC or Vista over the SSH master

**Hard-coded:** paths 17 (`~/anaconda3/bin/python`, `$SSH_CM_DIR/vista-$TACC_USER.sock`, `$TACC_WORK/vista/dbt/env/bin/python`); accounts 20 (`$TACC_USER`, `$NERSC_USER`, `$TACC_UID`); hosts 4 (`vista.tacc.utexas.edu`, `perlmutter.nersc.gov`); clusters 8 (`Vista`, `NERSC`); ssh_sockets 4 (`$SSH_CM_DIR/vista-$TACC_USER.sock`, `$SSH_CM_DIR/nersc-$NERSC_USER.sock`).

**To swap this node:** Another site for scenarios: DBT_SITE and the host table in lead_check.py; another repeat test: replace check() and keep the result fields.

**Feeds:** [novelty](#novelty), [arc-refit](#arc-refit)

## arc-refit

Extend a lead's arc season by season into the region's other years (2013-2026), predict where it is now, and refit without detections on fixed sources or lone far-season places.

**Code** (`nodes/arc-refit/`):

- `arc-refit/decam-big-tno/harness/arc_extend.py`
- `arc-refit/decam-big-tno/harness/arc_refit.py, arc_refit_nolone.py, arc_circfit.py`

**Run as:** `DBT_SITE=nersc python -I harness/arc_extend.py RUN ID...`; `DBT_SITE=nersc python -I harness/arc_refit.py RUN ID_SUBSTRING ...`

**Inputs**

- lead_check result: orbit fit
- static/<id>.json: fixed-source test per row

**Outputs**

- `runs/<run>/arc/<id>.json`: {id, n, nights, arc_days, arc_years, rms, a, e, i, r, rows, seasons, epoch_tdb, p, cov, now: [{date, ra, dec, sigma_arcsec}]} (positions: private)
- `runs/<run>/arc/<id>.refit.json`: arc format + dropped [{row, mjd, band, reason}], base

**Settings:** season gap 200 d, anchor rules and rms limits in the code

**Ran on:** Mac + remote reads on NERSC/Vista

**Hard-coded:** paths 5 (`~/anaconda3/bin/python`, `$ALE_HOME/planet-search/rsi/candidates/allarch/doublecheck/singlenight.json`); clusters 2 (`Vista`).

**To swap this node:** Another orbit fitter: lib-linker/orbit_fit.py is the one dependency (fit, predict, predict_sigma).

**Feeds:** [image-tests](#image-tests), [candidate-db](#candidate-db)

## novelty

Is a lead already known? Everything catalogued that moves like it (MPCORB, the MPC's ITF of unlinked tracklets along the whole carried orbit, per-night SkyBoT of all classes, observations by the MPC get-obs API) gets one orbit fitted through both data sets (two-body screen, n-body test, shifted-copy power check): one body fits both, two do not.

**Code** (`nodes/novelty/`):

- `novelty/planet-search/rsi/candidates/novelty_check.py, novelty_watch.py, novelty_express.py`
- `novelty/planet-search/rsi/candidates/resolve_hand.py, screen_new.py, mpc_check.py, cand9_orbit.py (n-body fit)`
- `novelty/planet-search/rsi/candidates/allarch/{knownitf_check,itf_path_check,apply_hand,verdicts,nsc_path,conecache}.py`

**Run as:** `nohup python -I novelty_watch.py > novelty_queue/watcher.log 2>&1 &`; `python -I novelty_check.py SPEC.json OUT.json`

**Inputs**

- SPEC.json: {id, points: [[mjd_utc, ra, dec, sigma_arcsec, site]]} or {id, run, gen, sid, p, r}
- rsi/data/mpc/<date>/: MPCORB and the ITF, refreshed daily by the watcher

**Outputs**

- `novelty_queue/done/<id>.result.json`: {verdict, flagged, n_flagged, itf, itf_path_scanned, skybot, skybot_nights, nbody, tests, weak_links, untested, n_untested, seasons, rate, own, seconds, ...}; verdict NOT IN MPC FILES | KNOWN <designation> | KNOWN ITF <trk> | UNSURE: <why>
- `novelty_queue/RESULTS.md`: one line per lead
- `confirm_queue/inbox/`: leads passed on for confirmation

**Settings:** radii, season gap 60 d, time budget TBUDGET, STOP file

**Ran on:** Mac (single-threaded, one at a time); SkyBoT and MPC web services

**Hard-coded:** paths 16 (`~/anaconda3/bin/python`, `$ALE_HOME/planet-search/rsi`, `$ALE_HOME/planet-search/linker`); hosts 9 (`datalab.noirlab.edu`, `minorplanetcenter.net`, `<rubin-science-platform>`); clusters 2 (`NERSC`).

**To swap this node:** Another catalogue: the readers for MPCORB/ITF/obs80 and the SkyBoT call are separate functions; the joint-fit test stays.

**Feeds:** [confirm](#confirm), [candidate-db](#candidate-db)

## confirm

Confirmation in images the program never used: night places by comoving stacks, systematic ranging, public DECam exposures within 60 days fetched and stacked along the ranged orbits with fake-track controls, CFHT MegaCam on NERSC.

**Code** (`nodes/confirm/`):

- `confirm/planet-search/rsi/candidates/lead_confirm.py, confirm_watch.py`
- `confirm/planet-search/rsi/candidates/{confirm,confirm2,confirm_rest,refit,scan,suspects,cand5_f3pos,cfht_fetch}.py`

**Run as:** `python -I lead_confirm.py SPEC.json OUTDIR`; `nohup python -I confirm_watch.py &`

**Inputs**

- SPEC.json (task 1 form): {id, run, gen, sid, p, r}
- archive images: NOIRLab DECam, CADC CFHT (fetched per lead)

**Outputs**

- `OUTDIR/{pos,range,list,...}.json, draws.npz`: stage outputs (resumable)
- verdict: SEEN IN OTHER IMAGES | NOT SEEN | NO DECISIVE DATA

**Settings:** stage thresholds in the docstring (stack S/N >= 5, controls recovered, static test)

**Ran on:** Mac (DECam), NERSC CPU (CFHT)

**Hard-coded:** paths 10 (`$ALE_HOME/planet-search/rsi`, `$SSH_CM_DIR/nersc-$NERSC_USER.sock`, `$NERSC_SCRATCH/planet-cand`); accounts 4 (`$NERSC_USER`, `$NERSC_CPU_ACCOUNT`); hosts 5 (`www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca`, `minorplanetcenter.net`, `perlmutter.nersc.gov`); clusters 13 (`NERSC`, `debug`); ssh_sockets 1 (`$SSH_CM_DIR/nersc-$NERSC_USER.sock`).

**To swap this node:** Another archive: a fetch + measure pair like cfht_fetch.py / confirm_rest.py; the verdict rule reads only the stack S/N per night.

**Feeds:** [candidate-db](#candidate-db)

## image-tests

The record of every image test of every candidate card (passes by analysis agents that measured a card at its predicted place in archive images the search never used: DECam, Pan-STARRS1, HSC, CFHT, ESO), merged into one friend-safe imagetests.json; plus image stamps and videos of a lead.

**Code** (`nodes/image-tests/`):

- `image-tests/planet-search/rsi/candidates/allarch/doublecheck/imgrecord/imagetests_build.py (builder; per-card hand records removed)`
- `image-tests/.../imgrecord/{selftest,numcheck,peek}.py, imgrecord14/{check_record,fpcheck,stages14}.py (checks)`
- `image-tests/decam-big-tno/harness/{cutouts_remote,cutouts_render,lead_video}.py`

**Run as:** `~/anaconda3/bin/python -I imgrecord/imagetests_build.py [--fp]`

**Inputs**

- <pass>.json and <pass>.handback.md per analysis pass: the pass's per-card measurements (not shipped)

**Outputs**

- `imagetests.json`: {about, built, passes, cards: {C###: [{pass, verdict, evidence, nights_seen, nights_missed}]}, counts, disagreements, notes}

**Settings:** PASSES (order), READER/WORDS (verdict mapping), the text rule (no MJD, coordinates, dates or file names in evidence)

**Ran on:** Mac; passes ran as relay agents (launch.sh)

**Hard-coded:** paths 8 (`~/anaconda3/bin/python`, `$ALE_HOME/rsi/decam-big-tno`, `$ALE_HOME/planet-search/linker`); accounts 4 (`$TACC_UID`, `$TACC_USER`); hosts 2 (`astroarchive.noirlab.edu`); clusters 2 (`NERSC`, `Vista`).

**To swap this node:** Another archive pass: add its name to PASSES and a reader for its JSON; the verdict mapping (SEEN AGAIN / NOT SEEN / UNSURE / NO POWER) stays.

**Feeds:** [candidate-db](#candidate-db)

## candidate-db

One folder per promising body with permanent numbers (C###), every identity and every piece of evidence; refreshed by one script that also rebuilds the page and the Drive copy.

**Code** (`nodes/candidate-db/`):

- `candidate-db/decam-big-tno/progress/objects/gather.py (every promising body of every search, re-finds merged)`
- `candidate-db/decam-big-tno/candidates/build.py (registry and cards)`
- `candidate-db/decam-big-tno/candidates/refresh.sh (gather -> build -> observe -> page -> package)`

**Run as:** `zsh candidates/refresh.sh`

**Inputs**

- runs/<run>/{lead_check,arc,static,confirm}/: vetting outputs
- novelty_queue/done/*.result.json: novelty verdicts
- candidates_status.json (task 1): hand-kept status of planet-search bodies

**Outputs**

- `objects.json`: every promising body: ids, run, region, rank, H, orbit summary, stage, verdict texts
- `registry.json`: permanent numbers: lead id / label -> C###
- `C###/card.json`: {cid, name, active, identities, orbit, stage, ladder, history, evidence, ...}
- `C###/README.md, evidence/`: copies of the evidence

**Settings:** stage rules (promising = filed or sent, or seen again + not in the MPC's files + not on fixed sources)

**Ran on:** Mac

**Hard-coded:** paths 2 (`~/anaconda3/bin/python`); hosts 1 (`<private-page-url>`).

**To swap this node:** Another store (e.g. a database table): build.py is the only writer; its card.json fields are the schema.

**Feeds:** [observe](#observe), [package](#package), [pages](#pages)

## observe

Observability of every promising body for the next 14 nights at chosen sites (altitude/azimuth, dark hours, V, 1-sigma); no RA/Dec written.

**Code** (`nodes/observe/`):

- `observe/decam-big-tno/progress/objects/observe.py`
- `observe/decam-big-tno/progress/objects/tonight.py`

**Run as:** `DBT_SITE=nersc ~/anaconda3/bin/python -I progress/objects/observe.py --sites G37`

**Inputs**

- runs/<run>/arc/<id>.json: barycentric state p, cov, epoch_tdb

**Outputs**

- `observe.json`: per site and night: Sun/Moon, dusk/dawn, per body alt/az every STEP min, hours above airmass 2, best altitude, V, sigma

**Settings:** SITES table (code, name, lon, lat, height, UTC offset), STEP

**Ran on:** Mac

**Hard-coded:** paths 3 (`~/anaconda3/bin/python`, `$ALE_HOME/rsi/decam-big-tno`); clusters 2 (`interactive`).

**To swap this node:** Another site: one row in SITES.

## package

Private Drive copy of the candidate database, one folder per object (positions as ADES PSV, orbit, README); and the packaging of a whole hill-climb run with its traces.

**Code** (`nodes/package/`):

- `package/decam-big-tno/candidates/package.py`
- `package/decam-big-tno/harness/observer_batch.py (batches of candidates for an outside observer; helpers used by package.py)`
- `package/decam-big-tno/harness/drive_pack.py (+ README)`

**Run as:** `DBT_SITE=nersc ~/anaconda3/bin/python -I candidates/package.py`

**Inputs**

- candidates/C###/: cards
- allarch/export_*.json: other-archive positions

**Outputs**

- `candidates/_drive/<CID>_<label>/`: README.md, positions.psv (ADES 2022), positions.json, orbit.json
- Drive folder: rsync --checksum copy

**Settings:** DRIVE path; PSV header (submitter, observatory W84)

**Ran on:** Mac (Drive for Desktop folder)

**Hard-coded:** paths 12 (`~/anaconda3/bin/python`, `$DRIVE_FOLDER`, `$ALE_HOME/rsi/decam-big-tno`); hosts 2 (`<private-page-url>`, `$VM_HOST`); clusters 2 (`NERSC`).

**To swap this node:** Another destination: change DRIVE and the final rsync.

## pages

Page builders (builders and templates only; no data): the hill-climb tracker, the Promising New Bodies catalogue, the DECam Data Movement dashboard, the decam-search site, and task 1's tracker.

**Code** (`nodes/pages/`):

- `pages/decam-big-tno/progress/{build_score,docs}.py, index.html`
- `pages/decam-big-tno/progress/objects/{build_page.py, page.html}`
- `pages/planet-search/fill/dash/*.py, index.html`
- `pages/pages/team-site/decam/{build.py, loop.sh, page.html}`
- `pages/planet-search/rsi/progress/*.py`

**Run as:** `python3 -I progress/build_score.py`; `python3 -I progress/objects/build_page.py`; `python3 -I fill/dash/build.py`; `decam/loop.sh`

**Inputs**

- state.json, grade.json, objects.json, observe.json, status of transfers: JSON

**Outputs**

- `score.json, index.html, status.json, public/decam-search/`: page data and HTML

**Settings:** forbidden-word lists (client names, names, Rubin DP2 text) stop a build

**Ran on:** Mac; published to claude.ai artifacts and a Cloudflare site

**Hard-coded:** paths 23 (`$PAGE_PUBLISHER`, `$PAGES_OUT/traces`, `$PAGES_OUT/index.html`); accounts 10 (`$NERSC_USER`, `$TACC_USER`, `$NERSC_CPU_ACCOUNT`); hosts 12 (`<team-site-url>`, `<private-page-url>`, `<private-page-url>`); clusters 177 (`NERSC`, `Vista`, `A100`); ssh_sockets 2 (`$SSH_CM_DIR/nersc-$NERSC_USER.sock`, `$SSH_CM_DIR/vista-$TACC_USER.sock`); models 33 (`Astra`, `Fable 5.2`, `claude-opus-5-5`).

**To swap this node:** Another host: the builders write static HTML/JSON; only the publish step is host-specific.

## archive-plans

Kits for searching the whole archive in task-sized pieces (222 survey-class groups, 3,761 slices) with the reference solver; made and piloted, not the live path (the conveyor replaced them).

**Code** (`nodes/archive-plans/`):

- `archive-plans/planet-search/groups/kit/{make_groups,run_group,score_group,known_bodies}.py, remote_run.sh, sites/*.sbatch, WORKER_BRIEF.md`
- `archive-plans/planet-search/slices/kit/{make_slices,slice_run,slice_score,common}.py`

**Run as:** `python3 -I make_slices.py ...`; `bash slices/<id>/run.sh`

**Inputs**

- groups/windows from linker/footprint: JSON

**Outputs**

- `slices/<id>/`: self-contained folder: run.sh, program, settings, known.json

**Settings:** A100 cost model (0.2276 h per window + 0.0209 h per exposure)

**Ran on:** NERSC / Lonestar6 (planned)

**Hard-coded:** paths 11 (`~/anaconda3/bin/python`, `$NERSC_SCRATCH/rsi-planet/env/bin/python`, `$NERSC_SCRATCH/decam-groups`); accounts 13 (`$NERSC_USER`, `$NERSC_GPU_ACCOUNT`, `$LS6_ACCOUNT`); hosts 6 (`astroarchive.noirlab.edu`, `ssd.jpl.nasa.gov`, `perlmutter.nersc.gov`); clusters 21 (`A100`, `Perlmutter`, `regular`).

**To swap this node:** Another solver: the program folder of a slice.

## lib-linker

Shared orbit and linking code: two-body orbits and frames, the barycentric orbit fitter used by every vetting step, the HelioLinC-style CPU/GPU linkers (starter of both tasks), NSC DR2 fetch.

**Code** (`nodes/lib-linker/`):

- `lib-linker/planet-search/linker/{orbits,orbit_fit,ed138,link_cpu,link_gpu,pairs_triton,nsc_fetch}.py`

**Run as:** `imported`

**Inputs**

- detections / places: numpy arrays

**Outputs**

- orbits, links: Python objects

**Settings:** none

**Ran on:** anywhere (numpy, scipy, astropy; torch/triton for link_gpu)

**Hard-coded:** hosts 1 (`datalab.noirlab.edu`); clusters 1 (`GH200`).

**To swap this node:** Another orbit fitter (e.g. find_orb, layup): wrap it to the fit/predict/predict_sigma functions the vetting code calls.
