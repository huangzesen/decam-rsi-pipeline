# Find distant Solar System bodies in DECam images

Write a program that finds slow-moving bodies beyond 25 au (trans-Neptunian objects) in a set of DECam exposures of
one patch of sky taken over days to weeks. Many of them are too faint to see in a single exposure: they only appear
when exposures are combined along the body's motion. Better methods find more of them with the same compute.

You are one generation in a long sequence of agents. Earlier generations' programs, their notes and the score each
one received are in `history/`; `history/SUMMARY.md` lists every one of them with its method family and score. Improve
the best one, combine ideas, or pivot to a different method (see "Pivot or improve"). `reference/` holds code from an
earlier project that you may reuse (read its README).

## What you deliver

The folder `submission/` (it starts as START_NOTE). It must contain:

- `finder.py`, run as
  `python finder.py --scenario SCENARIO_DIR --out OUT.json --budget-s SECONDS`
- any other modules or files it needs, network weights included (the whole folder at most 500 MB; `.npz`, `.npy` and FITS files are not
  copied; no data tables of known objects);
- `NOTES.md`, starting with three lines: `Family:` the method family in a few words, `Base:` the program you started
  from (or `new`), `Change:` your main change in one sentence. Then what your method does, why, and what you would try
  next. Later generations read it, and the first three lines go into `history/SUMMARY.md`.

The program is run once per scenario on scenarios you never see, each with a wall-clock budget (`--budget-s`,
currently 600 s). It is killed 30 s after the budget, so write `OUT.json` early and update it as you go.

## Input: a scenario folder

- `exposures.csv`, one row per exposure: `exp_id`, `image`, `weight`, `mask` (paths relative to the scenario folder),
  `mjd_start`, `exptime` (s), `mjd_mid` (UTC MJD of mid-exposure), `band` (g, r, i, z, Y or VR), `airmass`, `magzero`
  (may be empty), `seeing_arcsec`, `ra_center`, `dec_center` (deg), `obs_x`, `obs_y`, `obs_z` (the observatory's
  heliocentric position at `mjd_mid`, ICRS equatorial axes, au).
- `exposures/`: the DECam Community Pipeline calibrated files, fpack-compressed FITS, one HDU per CCD (about 60):
  `*_ooi_*` image, `*_oow_*` weight (inverse variance), `*_ood_*` data-quality mask (0 = good). Each CCD header has a
  TPV WCS (astropy reads it), `FWHM` (pixels), `SKYSIGMA`. Pixel scale 0.263 arcsec.
- `meta.json`: centre, time span, number of exposures and nights.

Scenarios span 3 to 60 days, with 4 to 15 nights and 4 to a few hundred exposures, in several bands. Some exposures of
a scenario may point elsewhere within about a degree.

## Output: `OUT.json`

```json
{"candidates": [
  {"epoch_mjd": 60218.88, "state": [x, y, z, vx, vy, vz], "p": 0.93, "mag": 23.4}
]}
```

- `state`: the body's heliocentric position (au) and velocity (au/day) at `epoch_mjd` (UTC MJD), ICRS equatorial
  axes. Positions on the sky are computed from it with two-body motion about the Sun and light time, so the state must
  put the body within 2 arcsec of where it really is on at least 3 different nights of the scenario.
- `p`: your confidence that the candidate is a real body, between 0 and 1. Ranking matters (below).
- `mag`: optional brightness estimate.
- At most 200 candidates per scenario.

## How submissions are scored

**This round your program is graded on GRADER_SHORT** (details under "Machine and rules").

You cannot evaluate your submission: no truth is available to you, and you will not be told per-scenario results.
After your session the program is run on a fixed set of sealed scenarios and gets one number, the reward, which later
generations will see next to your code.

- Over all sealed scenarios together, only the 25 x (number of scenarios) candidates with the highest `p` are kept.
  Junk candidates with high `p` take slots from real ones.
- A kept candidate is a hit when it matches a known body (a body counts once; duplicates score nothing). A body's
  weight comes from its S/N in that scenario's data, measured at its true position:
  - weight 1: S/N >= 5 in a single exposure on at least 3 nights;
  - weight 2: otherwise, S/N >= 5 in the stack of one night's exposures on at least 3 nights;
  - weight 4: otherwise, S/N >= 5 in the stack of all its exposures, with data on at least 3 nights. This is not only
    faint bodies: a body bright on one or two nights and too weak to see on the others also weighs 4.

  Bodies below that are not counted.
- A candidate that matches no known body but is confirmed by exposures held back from you also scores.
- Reward = weighted hits / total weight of the findable known bodies.

## Machine and rules

GRADER_MACHINE
- The program may read only its own folder and the scenario folder, and write only `OUT.json`'s folder and its scratch
  `HOME`/`TMPDIR`.
- Other agents work on this machine at the same time. You can signal (kill) only processes you started: stop your own
  runs by PID (`kill $!`), not with `pkill`/`killall` by name.
- Development scenarios without truth are in `DEV_DIR` (path given below): run your program on them to check that it
  works, how long it takes and what it outputs. There is no way to score it; do not try to obtain truth.
- `python tools/check.py` runs `submission/` on the development scenarios the way the grader will, here on this Mac (a fresh copy of
  the folder, the budget, killed 30 s after it) and validates `OUT.json`. It gives no score. Run it before you finish.
- Do not embed tables of known Solar System objects or try to look them up.

## Pivot or improve

- The run always keeps the best program found so far. Your score affects only your own entry: a low or zero score
  from a new method costs the run nothing, and it tells later generations what does not work yet.
- Bodies not clearly seen on 3 separate nights weigh 4 (see the weights above). A method that cannot find them has a
  low ceiling, however well it is tuned.
- Pivoting to a different method family is welcome. A new method that runs and writes valid output, even if it finds
  little yet, is a useful step for the generations after you; say what is missing in `NOTES.md`.
- Be creative. You are not limited to published methods or to variants of the programs in `history/`: an algorithm of
  your own, built on a new idea about how to tell a slow-moving body from noise, is exactly what this run is looking
  for. Small changes to the best program have often left its score unchanged (see `history/SUMMARY.md`). The choice
  is still yours; if you do try something new, say in `NOTES.md` what the idea is and why it might work.
- Do not limit your imagination. Use any idea, from any field, that you think could work.
- Neural networks that you design yourself are encouraged, for any purpose in the method. Choose the inputs,
  architecture and size to fit this data. Some uses: a small CNN that tells a body from an artifact in cutouts or
  stacks, a network that sets `p` from a candidate's measurements, a learned denoiser or matched filter for faint
  stacks, a network that proposes motions to search, or a nonlinear solver. No earlier program has shipped one, and
  "a small CNN self-trained on injected PSFs" has sat untried in the notes since gen061. No truth is available to
  train on, so make your own training data: for example, inject synthetic bodies with a realistic PSF, motion and
  brightness into the development exposures, and use the real sources and artifacts there as negatives. Train during
  your session, or inside the finder's own time budget on each scenario. `torch` is available on every machine above.
  Weights count toward the 500 MB limit. A `.pt` file is copied with your submission; `.npz` and `.npy` files are not.

## Your session

- Time limit: SESSION_LIMIT. Use it: a change that needs the whole session is fine. By the end, `submission/` must run
  and write valid output (a broken program scores 0 for your entry); run `tools/check.py` before you finish.
- `history/SUMMARY.md`: every earlier program, best first, with its score, where it was graded, its method family,
  base and main change. `history/genNNN/` holds each one's `submission/`, `NOTES.md` and `reward.json` (reward, run
  status, `graded_on`, the machine that reward was measured on, and `budget_s`, the time per scenario it was given;
  earlier entries without `budget_s` had 300 s; `per_scenario`, the seconds your program ran on each hidden scenario and
  how it ended (`ok`, `timeout` or an exit code), with the scenarios labelled A, B, ... in the same order every round;
  there is no per-scenario score), and `trace.md`, that agent's whole session, subagents included: its messages,
  the commands it ran and their (shortened) output. Read the traces to see what earlier agents tried, measured and dropped.
