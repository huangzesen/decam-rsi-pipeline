# Brief for a group worker: apply the reference method to one group of DECam windows

2026-11-11.
Your job is to adapt, not to develop: about 20 minutes, at most 2 finder runs. The brief the five pilots worked under
(simulators, injection tests) is `old/WORKER_BRIEF_v1_pilots.md`; do not follow it.

Your group id is in your prompt; your folder is `planet-search/groups/<id>/`. Paths below are relative to
`planet-search/`.

## The project in four lines

We search the whole DECam archive (2012-2026) for trans-Neptunian objects, large distant ones first. The usable
exposures were sorted into windows (one 1 deg sky cell, one 60-day bin, >= 3 nights) and the windows into 222 groups
of similar cadence, sky and observing program. One program, the reference solver, finds moving bodies in one window.
Each group runs its own copy; you make that copy fit the group, from the exposure metadata alone.

## What you have

| path | what |
|---|---|
| `groups/<id>/group.json` | the group's card: family, program, statistics (10th / 50th / 90th percentiles), cost |
| `groups/<id>/windows.json` | every window with its features and exposures (file stamp, MJD, exposure time, band, pointing) |
| `groups/<id>/program/` | your copy of the reference solver (`finder.py`, `detcls.pt`, `rerank.py`, `NOTES.md`) |
| `groups/<id>/settings.json` | run settings: `budget_s`, the seconds the finder gets per window |
| `rsi/REFERENCE_SOLVER.md` | what the reference does, the envelope it was tested in, and what to check per case |
| `groups/kit/mini/s1c351b17_3ccd/` | a real 3-CCD scenario for the smoke test (not from your group) |

The contract stays exactly `python -I finder.py --scenario SCEN_DIR --out OUT.json --budget-s SECONDS`.

## What to do

1. **Read** the card and `rsi/REFERENCE_SOLVER.md`. From `windows.json` get the numbers of the typical window and of
   the extremes: exposures, nights, exposures per night and their spacing, span, bands, exposure times, ecliptic and
   Galactic latitude.
2. **Check the reference's assumptions against those numbers** and set what follows from them:
   - the orbit prior, written for fields near the ecliptic, at this group's latitudes;
   - the switches tied to window size: deep-pair jobs (>= 9 exposures on >= 4 nights), the co-add mode (> 55
     exposures), minimum nights, seed caps;
   - bands and exposure times the reference handles differently (VR, Y, short exposures);
   - `budget_s` in `settings.json`, when the window sizes call for more or less than 900 s;
   - anything that would make the program fail on, or skip, a window of this group.
   Use the program's existing switches and constants where they exist. Change code only for a plain reason from the
   metadata, keep it small, and put it behind a named constant at the top of `finder.py` that records the reference
   value. If nothing needs to change, change nothing.
3. **One smoke test**: your program on the mini scenario at budget 300, to see that it still finishes and writes
   valid output:
   `groups/kit/remote_run.sh groups/<id>/program groups/kit/mini/s1c351b17_3ccd groups/<id>/work/smoke.json 300`
   (a held cluster CPU node; exit code 3 = no node is held, then run it on this Mac with
   `OMP_NUM_THREADS=4 ~/anaconda3/bin/python -I`). The reference gives 165 candidates there in about 210 s.
4. **Deliver**, inside `groups/<id>/`:
   - `program/GROUP.md`: each difference from the reference, its constant, the reason in one line; or "unchanged";
   - `settings.json`;
   - `NOTES.md`, half a page: the group's numbers, what you set and why, what you left alone, anything the run on
     real images should watch.
   Then end with a three-line report.

No simulators, no injected bodies, no parameter sweeps, no comparison campaigns. The check that counts is the group's
run on real images, scored on the known bodies afterwards.

## Rules

- No image of your group exists here and none may be downloaded.
- The known bodies are sealed: do not look up known objects anywhere, and do not read `rsi/private/`,
  `rsi/scen_sealed/`, `rsi/candidates/`, `rsi/runs/*/gens/`, `groups/_private/`, `slices/`. The program must not carry any orbit
  or position of a real body.
- Write only in your own folder (scratch in `groups/<id>/work/`, under 50 MB). `groups/kit/` is read only. No git, no
  cluster access other than `remote_run.sh`, no messages to anyone.
- Say plainly what you did not check.
