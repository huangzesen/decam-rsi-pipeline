# fill: download every DECam exposure NERSC does not hold

Pipeline map (what runs where, logs, stop files, restarts): `../PIPELINE.md` (planet-search/PIPELINE.md).

Started 2026-11-17. TACC Vista scratch holds what NERSC lacks; searches run
on NERSC, and the frames they need are copied back with Globus (NERSC <-> Vista measured ~640 MB/s on 2026-11-17).

| File | What |
|---|---|
| `make_blank.py` | the blank: usable public exposures (`bulk/` manifests) minus the NERSC mirror, archive10's a10n downloads (`a10n_all_stems.txt`) and archive5; the third 5 %'s missing exposures first. Writes `blank.tsv`, `blank.json` |
| `blank_vista.tsv`, `blank_nersc.tsv` | split of `blank.tsv`: Vista downloads the head (115.8 TB), the NERSC relay the tail (38.6 TB) |
| `fetch.py` | (23:25: a full-size file whose md5 differs from the listing is kept when every data HDU's DATASUM verifies, listed in `log/md5diff_*.txt`; the archive edits headers after listing. 23:46: a file damaged in the archive itself, the same wrong bytes twice, is kept once with its bad HDUs in `log/damaged_*.txt`: detection must skip those HDUs) | copy of `bulk/fetch.py` (md5-checked, resumable, `--part k/N`, STOP file) |
| `vista_fill.sbatch` | one Vista gg node, 2 fetch copies x 32 streams; jobs <slurm-job-id>-42 submitted 19:58 PST and cancelled while pending (0 SU) on approval: Vista nodes cost SUs, so all downloading is at NERSC |

Remote: Vista `$TACC_SCRATCH/decam-fill/` (img/<YYMMDD>/, log/). Stop: `touch log/STOP` there.
Vista scratch purges files not accessed for 10 days (touching to avoid it is not allowed).
Plan since 20:40 PST: NERSC's free xfer queue downloads the whole blank.tsv, Globus copies each finished batch to Vista, the NERSC copy is deleted after the Globus task succeeds. Not started yet: waits for NERSC scratch space (archive10's a10n copies are being moved to Vista).

## NERSC leg (from 2026-11-17 21:50 PST)

| File | What |
|---|---|
| `nersc_fill.sbatch` | one job = one part (k/6; since 22:56 parts 4 and 5 run as halves 4/12, 10/12, 5/12, 11/12 in p41, p42, p51, p52) of `blank.tsv` into `$NERSC_SCRATCH/decam-fill/p<k>/img/<YYMMDD>/`, fetch.py in 1-hour chunks; space limits read live from `decam-fill/limits` (since 22:40: start below 19.0 TiB, pause new files at 19.4 TiB; defaults 17.5 / 18.5); `touch decam-fill/STOP_ALL` stops all |
| `relay.py` | on the Mac, `python3 -I fill/relay.py loop 30`: every 30 min one Globus task of newly finished files NERSC -> Vista `decam-fill/img/`; after SUCCEEDED the NERSC copies are deleted (`fill_rm.py` on NERSC, only `p*/img/*/c4d_*`), a15 files kept (`relay/keep_a15.txt`). State in `relay/`; stop with `touch fill/relay/STOP` |
| jobs | free xfer 61516204-07 (parts 0-3); parts 4-5 on 2 regular CPU nodes ($NERSC_CPU_ACCOUNT) once scratch is free |

Space: archive10 a10n (Globus tasks in `archive10/globus_task.txt`) and archive5 n0-3/img (task <globus-task>) are copied to Vista and then deleted at NERSC, keeping the a10n files archive15 links to.
Expected: ~200 MB/s from 4 xfer jobs (measured 2026-11-14), more with the CPU nodes if NOIRLab keeps up; 154 TB in ~6-9 days.

Page refresh (2026-11-17 23:20): a shell loop on the Mac runs `dash/build.py` every 10 minutes (history and `out/status.json`, written atomically), and a background Sonnet agent uploads each new `out/status.json` to <private-page-url> and repoints the 8 status datasets (`dash/wait_new.py`, `dash/docs.py BLOB`). Stop both: `touch fill/dash/STOP`.

2026-11-20 03:35 PST: **Lonestar6 joins the download**. NOIRLab refused most NERSC connections from about 19:00 11-19 (NERSC streams cut 40 -> 12 in `decam-fill/limits` at 03:00 to test). Lonestar6's share = the tail of the remaining list, 24 TB / 116,599 files in `$TACC_SCRATCH/decam-fill-ls6/blank_ls6.tsv` (8 TB at 03:17, 16 TB more at 03:35; the same names are in NERSC's `blank.skip.txt`, so NERSC leaves them; NERSC has 74.9 TB left). `fetch.py` runs on one CPU node with 64 streams (`decam-fill-ls6/node.sbatch`; dev test job <slurm-job-id>, then 24 h job <slurm-job-id>): ~115 MB/s, ~10 TB/day, about 1 SU per hour. `relay_ls6.py loop 30` (Mac) sends finished files by Globus (LS6 collection fc2bdfc1-…) to the same Vista folder as `relay.py` and deletes them on Lonestar6. The login node gave only 7-11 MB/s with 4 streams (1 CPU there).

2026-11-20 03:47 PST: **u06/u16's missing files on Lonestar6.** NOIRLab's refusals left the conveyor units u06 and u16 waiting on 11,236 files (2.2 TB). Lonestar6 job <slurm-job-id> (`$TACC_SCRATCH/units-ls6`, list reversed) fetches them at ~113 MB/s beside the fill node (two LS6 nodes add up); `relay_units_ls6.py loop 10` (Mac) puts each into its NERSC share folder (`relay_units_ls6/dest.tsv`), where the NERSC fetcher skips it and the conveyor detects it. Expected: u06 and u16 searchable within about 6-8 hours.

2026-11-20 17:35 PST: **Vista joins the download again**. NOIRLab's refusals
had cut NERSC to ~50 MB/s (every part at 0-12 MB/s, hundreds of refusals each) and each Lonestar6 node to 13-22 MB/s.
Vista share 2 = the last 16 TB NERSC had left (84,143 files; NERSC `decam-fill/blank_vista2.tsv`, appended to
`blank.skip.txt`, backup `.bak-1120-vista`; copy here `blank_vista2.tsv`). On Vista `decam-fill/vista_fill2.sbatch`
(vista_fill.sbatch on the new list) runs on `gg` CPU nodes (0.33 SU per node-hour): jobs <slurm-job-id>/7 (`dfill2_0`, `dfill2_2`;
2 fetch copies x 32 streams each, parts k/4, 24 h, <= 8 SU each), files straight into `decam-fill/img/` (the cache the
conveyor pulls from), done logs `log/done_*of4.txt`. `fetch.py` there updated to the current one (old: `fetch.py.bak-1120`).
`vista_known.py loop 30` (Mac) appends Vista-downloaded and Lonestar6-relayed names to `relay/relayed.txt` so unit
selection and preparation count them as on Vista (it added 8,158 Lonestar6-relayed names the conveyor had not known).

2026-11-20 17:35 PST: **NERSC downloads paused 12 h**: NOIRLab had cut NERSC to ~50 MB/s with hundreds of refusals per part. All fillx* and cvf-* xfer jobs cancelled
(fetch.py resumes from its done logs). `conveyor/NERSC_DL_PAUSE` holds the end (epoch 1795268124 = 2026-11-21 05:35 PST);
while it lies ahead `unit.py` stage counts a new unit's NERSC shares without submitting them ("paused") and
`units_side.keep_downloads` does nothing, so both resubmit by themselves after it (backups `*.bak-1120-dlpause`).
Lonestar6, Vista and the Mac keep downloading.

2026-11-20 17:40 PST: **the Mac downloads too**. Mac share =
4 TB / 19,881 files carved from NERSC's remaining list (NERSC `decam-fill/blank_mac.tsv`, in `blank.skip.txt`, backup
`.bak-1120-mac`). `fill/mac_fill.sh` runs fetch.py (24 streams, niced) into the Crucial-8TB buffer
`$DATA_DRIVE/decam-fill-mac/` (AGENTS.md there) in 0.1 TB chunks while it holds < 200 GB;
`fill/relay_mac.py loop 2` rsyncs finished files (2 lanes, --remove-source-files, <= 400 per cycle) to NERSC
`decam-fill/pmac/img/` over the NERSC ssh master and appends their names to `pmac/log/done_mac.txt`, so `relay.py` (it walks
every `decam-fill/p*/`) carries them to Vista with Globus and deletes them at NERSC. The first version (17:40-18:15)
pushed straight to Vista over the Vista master at only ~3 MB/s in 4 lanes (one file to NERSC: 47 MB/s); it was stopped,
its temp files on Vista removed (backup `relay_mac.py.bak-1120-vista`). `relay_mac/relayed.txt` = names at NERSC.
First minutes: 103-107 MB/s, no refusals, no bad md5. Stop: `touch .../decam-fill-mac/log/STOP_ALL` (and `log/STOP`),
`touch fill/relay_mac/STOP`.

2026-11-20 19:27 PST: **Lonestar6 downloads stopped**: every fill-ls6 and
units-<uid> job cancelled; `conveyor/LS6_DL_STOP` keeps the conveyor's ls6_feed from submitting new ones (backup
`conveyor/units_side.py.bak-1120-ls6stop`). relay_ls6.py and relay_units_ls6.py keep relaying what was fetched. Mac relay
upload capped at ~86 MB/s (`RELAY_MAC_BWLIMIT`, KiB/s per lane): at full line rate it halved the Mac's download.

2026-11-20 22:35 PST: **active download monitor and rotation**. `fill/dl_watch.py loop 10` (log `dl_watch.log`, history `dl_watch.json`):
per-device NOIRLab download speed, files, bad md5 and errors from every live fetch log (NERSC, Lonestar6, Vista, Mac) and
transfer speeds (Mac -> NERSC rsync, NERSC -> Vista, Lonestar6 -> Vista/NERSC and Vista -> NERSC Globus); a device that
degrades triggers 60 s probes of the idle devices (`rsi/candidates/dltest/dltest.py`; copies in `$SCRATCH/probe` at NERSC
and Lonestar6). It only reports; switching is done by hand. 22:27 probes: NERSC 174 MB/s, Lonestar6 50 MB/s, both without
refusals; the Mac had degraded (8-18 MB/s, most files bad md5), so NERSC's pause ended early (`conveyor/NERSC_DL_PAUSE`
renamed `.ended-20261120-2235`) and the Mac stopped (`STOP_ALL`). `fill_rm.py` now also accepts `pmac/` (backup
`.bak-1120-pmac`, both sides): it had refused, leaving 924 GB of relayed Mac files at NERSC (4,116 deleted at once).

2026-11-20 22:50-23:05 PST: **polite downloads with a per-device stream budget**. NOIRLab refused NERSC again at 22:45 (16 jobs x 24-40 streams: 1 MB/s, 1,753 errors in 10 min).
`fetch.py` (backup `fetch.py.bak-1120-ctl`, same copy on all four devices) now reads `fetch_ctl.json` beside itself every
30 s: `total` = streams for all fetch.py copies of that device together (each registers in `fetch_ctl.json.d/` and takes
its share, at most its --streams), `pause_until` = no file starts before then; it logs `streams N` and `cool-down until`.
`dl_watch.py` (now also the controller) judges each device every 10 min: LIMITED when errors >= 10 and more errors than
files, or bad md5 on > 25 % of >= 8 files; then budget - 1 (not below 1) and a 30 min cool-down; budgets never rise by
themselves (`dl_watch.py set DEVICE N` by hand). Start budgets: NERSC 9 (3 unit fetchers x 3; `decam-fill/limits`
STREAMS=3 caps each job; general fill stopped with `decam-fill/STOP_ALL`), Lonestar6 8 (one job, u19's gap from the end;
`conveyor/units_side.py` LS6_UNITS_MAX 1, STREAMS 8; `LS6_DL_STOP` lifted), Vista 8 (the two queued jobs' four copies, 2
each), Mac 8 (`STREAMS=8 fill/mac_fill.sh`, general fill as before). First readings 22:57-23:00: NERSC 105 MB/s, Lonestar6
47, Mac 51 (together ~200 MB/s), no refusals; the Mac's first minute had 11 bad md5. Probes are now 4 streams for 45 s.

2026-11-21 00:30 PST: the controller also raises. Together had sagged to ~148 MB/s (Vista's jobs queued to 11-21 21:10, the Mac at 7 streams after its 23:45 limit). `dl_watch.py`: a device clean for 3 judged cycles gets +1 stream while all devices together are under 200 MB/s, cap 10 per device, and for 3 h after a limit not back to the budget that was limited (log line `RAISED`). NERSC and Lonestar6 set to 10 by hand; NERSC's per-job cap `decam-fill/limits` STREAMS 3 -> 10 (read at each hourly chunk, backup `limits.bak-1121-0032`), so the device total decides; the next Lonestar6 unit job gets STREAMS=10 (`conveyor/units_side.py`; the running job <slurm-job-id> stays at 8).

2026-11-21 00:45 PST: both Mac "limits" (23:45 bad md5 64 of 126; 00:41 27 of 27) were false. The Mac's list has 16 files (Jan 2013, e.g. c4d_130112_042832_ooi_g_v1) whose served bytes never match the listed md5, and differ on each try (so fetch.py's "damaged at archive" test, the same bytes twice, never fires). Every hourly chunk retried them first, 4 tries each: 64 bad md5 and 16.7 GB wasted per chunk, which dl_watch read as a refusing archive. fetch.py now leaves out files that failed all tries in >= 2 earlier runs (log/failed_*.txt); deployed to NERSC, Lonestar6 (both dirs) and Vista (md5 02c5dbdb...; backups fetch.py.bak-1121-gaveup). Mac budget set back to 8; `dl_watch.py set` now also clears the 3-hour hold after a limit.

2026-11-21 02:55 PST: fetch.py tries files that failed once at the END of each chunk (stable sort), not first: the Mac's 02:52 limit (bad md5 11 of 40) was a new chunk opening on retries of the previous chunk's 5 failures, after a healthy chunk (404 files, 5 % bad). Deployed to all sites; Mac cool-down lifted, budget 8.

2026-11-21 05:15 PST: weighted stream shares. fetch.py splits the device total in proportion to weights from fetch_ctl.json.weights ({"substring of the manifest path": weight}; largest remainder; a process may get 0 and wait; no file = equal split as before), heartbeat files carry each process's weight. Why: at NERSC the 10 streams were split equally over 9 processes, so u17 (1,919 files left, 24 regions waiting) and u19 had 1 stream each beside u20's 7 shares. NERSC weights: /units/u17/ 8, /units/u19/ 6, rest 1 -> u17 4, u19 3, u20 3 streams. Deployed everywhere (md5 cae98418...; backups fetch.py.bak-1121-weights); running copies switch at their next hourly chunk. Update the weights by hand when a unit finishes or a new one starts.

2026-11-21 06:50 PST: dl_watch.py judges the md5 side of LIMITED by files given up after all tries ('failed' in the status line): LIMITED when they are > 25 % of >= 8 files (was: bad md5 tries > 25 %). The Mac's 06:44 limit (bad md5 20 of 71) was 4 April-2013 files tried 4 times each while good files arrived at 50-80 MB/s. Mac cool-down lifted, budget 8; loop restarted (backup dl_watch.py.bak-1121-failedfiles).
2026-11-21 06:47 PST: the 06:44 restart cycle wrongly limited Vista and the Mac (a 1-minute window, and no stored 'failed' count so the whole log's count looked new); both reset (Vista 9, Mac 8). dl_watch.py now never judges a window under 5 min and treats a missing stored count as unchanged.
2026-11-21 12:25 PST: Vista = one gg node, 10 streams, no GPU. Cancelled the second download node 1157827 (parts 2-3 of blank_vista2) and every Vista GPU worker (gh-dev 1159199 running, gb 1158655, gh 1156063/4, gh-dev 1159247 pending); conveyor.py KINDS vista-gh/ghdev/gb max 0 (backup conveyor.py.bak-1121-gganly), loop restarted (pid 53028). Follow-on 1159284 (gg, 24 h, all four parts, afterany:1157826) keeps one node downloading after 1157826 ends. fetch_ctl.json total 10 (dl_watch cap 10), so the node shares 10 streams across its fetch copies. The cancelled worker's claim (u20 scen12) is released by the conveyor to NERSC.
2026-11-21 14:30 PST: NERSC fetch weights now u19 6, u20 2, others (u21) 1 (u17 done; was u17 8, u19 6; backup fetch_ctl.json.weights.bak-1121-1430). 15 fetch processes share the 10-stream budget, so u21's start first gets streams as u19/u20 finish. 14:28 NERSC used 6 of 10.
2026-11-21 15:12 PST: relay_mac.py hung after the 14:06 NERSC master loss (no log line after 14:26, no child process); the Mac buffer reached ~200 GB and mac_fill stopped starting chunks (dl_watch 15:08: mac idle). Restarted (python3 -I fill/relay_mac.py loop 2); uploading again. PIPELINE.md check 3 now says to restart it after 10 silent minutes.
2026-11-22 00:30 PST: **Mac share 2**. Share 1 (4 TB, 19,881 files) was done except April-2013 files, and the Mac had been paused at 23:12 (STOP_ALL). New share = part 3 of Vista's `blank_vista2.tsv` (blocks k/4; parts 2-3 were idle until Vista's follow-on job <slurm-job-id> starts), less what Vista or the relays already hold: 18,201 files, 3.47 TB, written to the Mac's `blank_mac.tsv` (share 1 kept as `blank_mac.tsv.bak-1122-share1`) and, the same names, to Vista `decam-fill/blank_vista2.skip.txt` so Vista's jobs leave them out. Plus 197 share-1 leftovers: 18,398 files.
2026-11-22 00:38 PST: why the Mac failed every 2013 v1 file: `mac_fill.sh` ran fetch.py with Homebrew `python3`, which has no astropy, so `bad_hdus()` always returned None and no stale-md5 or damaged-at-archive file was ever kept on the Mac (the other sites' pythons have astropy). `mac_fill.sh` now uses `PY=${PY:-~/anaconda3/bin/python}` (backup `mac_fill.sh.bak-1122-py`); share 1's failure record moved to `log/failed_0of1.txt.pre-astropy-1122` so those files are tried again. First minutes: 30-66 MB/s, stale-md5 and damaged files kept.

2026-11-22 18:19: dl_watch.py MAX_STREAMS 10 -> 15, COOL_S 1800 -> 600, GOAL_MBS 200 -> 400 (backup dl_watch.py.bak-1122-cap15); every device set to 15 streams by hand; conveyor PREP_MAX 3 -> 4 so a fourth unit's downloads start while u19-u21 finish.

2026-11-22 23:1x: Vista fetch python had no astropy (python3 -I, so no user site), so files whose listed md5 is stale could not be kept by their DATASUMs; they were retried, given up, and cut Vista's streams by the dl_watch rule. Made decam-fill/venv (astropy 6.0.1, numpy 1.26.4) on Vista scratch and pointed vista_fill2.sbatch at it (backup .bak-1122-venv). The running job <slurm-job-id> (started 17:32) keeps the old python until it ends.
