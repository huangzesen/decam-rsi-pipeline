# Find large distant bodies in thirteen years of DECam single exposures

You write a program that searches one region of sky, observed by the Dark Energy Camera (DECam) on the Blanco 4 m
telescope from 2013 to 2026, for bodies of the outer solar system: trans-Neptunian objects of about 200 km and larger, and
anything bigger and farther (dwarf planets, Sedna-like bodies, a distant planet). It gets every source detected in every
single exposure of the region and returns linkages: sets of detections that it claims are one moving body.

## The purpose, in full

We want one generally applicable, efficient method that finds as many of the known bodies as possible (recall) and, at the
same time, identifies bodies nobody knows yet. The program you leave will be run, unchanged, over the whole DECam archive:
several hundred regions like the ones you have, most of which you have never seen. Report every body you believe is
real, known or not: the known ones measure how good the search is, and the confident ones that match nothing known are
checked by us as possible discoveries.

Nothing is hidden from you. You have every region we have prepared (OPEN_N regions, OPEN_DET detections) with every known
body in them labelled (OPEN_BODIES findable ones), names and absolute magnitudes included. That includes the four regions
the grade is computed on: GRADED_SIDS. You also have every earlier program with its
outputs and its grade in full.

Your result is trusted as it comes, and every generation's program is read after it is graded. What counts is the method. A
program that finds bodies because it carries what you know about these regions (names, positions, orbits, row numbers,
exposure lists, or settings that only make sense for one region) is not a search: it will find nothing in the rest of the
archive, it is easy to see in the code, and it scores 0. Use the labels to understand why bodies are missed and to measure
a change; put what you learn into the method.

## What the program gets

`finder.py --scen DIR --out OUT --budget-s SECONDS`, one region per call. `DIR` holds:

| File | What |
|---|---|
| `det_ra.npy`, `det_dec.npy` | float64, degrees (ICRS), one row per detection |
| `det_flux.npy`, `det_ferr.npy` | float32, matched-filter flux and its error, image units |
| `det_snr.npy` | float32, flux / error; every detection has S/N >= 4.5 |
| `det_sharp.npy` | float32, peak pixel over the PSF peak for the fitted flux: about 1 for a star or a moving body, much more for a cosmic ray |
| `det_flag.npy` | uint8: 1 near saturation, 2 masked pixels within 2 px, 4 within 24 px of the CCD edge |
| `det_x.npy`, `det_y.npy`, `det_ccd.npy` | position on the CCD (pixels) and the CCD number |
| `det_exp.npy` | int32, row of `exposures.json` |
| `exposures.json` | per exposure: `stem`, `mjd_mid` (UTC), `band` (g r i z Y VR), `exptime`, `magzero` (magnitude = magzero - 2.5 log10 flux), `fwhm`, `airmass`, `ra0`, `dec0` (pointing), `m5` (5-sigma point-source magnitude), `obs_xyz` (heliocentric position of the telescope, equatorial, au) |
| `region.json` | region centre, radius, first and last MJD |

Rows are grouped by exposure. Every source is there: stars and galaxies (which sit still), noise peaks, cosmic rays,
artefacts near bright stars, satellites, asteroids, and the bodies you are after. The exposures of a region lie within 4 to
15 degrees of its centre; a region has 80-1,100 exposures and 8-93 million detections (the four graded ones: 530-760
exposures, 55-72 million detections each). Exposures come in bursts: some seasons have a few nights, some have dozens; a
body crosses a region in weeks (at 40 au) or stays for years (beyond 300 au).

## What it returns

`OUT/candidates.json`: a JSON list of `{"det": [row numbers], "p": number}`, one entry per body you believe is real, where
`p` is your confidence that the rows are one real moving body. At most 2,000 entries per region are read (highest `p`
first). Extra keys are ignored.

Optional, never scored: `OUT/trace.json` = `{"stages": [{"name": "...", "groups": [[row numbers], ...]}, ...]}` with the
groups of detections your program held at its own intermediate steps (tracklets, links before a filter, ...; at most 6
stages, 500,000 groups and 5 million row numbers per stage, 100 MB). If it is there, the per-body feedback (below) also
says, for every body, how many of its nights were together in one group at each step, so a body that never reaches a
candidate still shows where it was lost.

## How it is scored

The grade is computed on the four graded regions (GRADED_SIDS; GRADED_BODIES findable known bodies). Each runs on one
NVIDIA A100 with 40 GB of memory (CUDA; Python 3.11 with numpy, scipy, astropy, scikit-learn and PyTorch 2.11 with
Triton), no network, and a fresh copy of your submission folder. The same program runs on every region. `--budget-s` is
BUDGET_S seconds: a ceiling, not a target. The program is killed at the ceiling plus 30 s and whatever `candidates.json`
holds then is scored.

Every program gets two rewards. They are reported separately and never mixed into one number.

1. **Recall** (higher is better).
   - The candidates of all four regions are ranked together by `p`. The number of scored slots is twice the number of
     known bodies the regions hold (findable ones, below): room for every known body and as many new ones, but a wrong
     linkage with a high `p` takes a slot from a real body.
   - A candidate finds a body when at least 80 % of its detections belong to that body and they cover at least 5 of its
     nights. Each body counts once.
   - The bodies scored are the known trans-Neptunian objects, of any size, that have a detection on at least 5 nights in
     a region; weight 2 for absolute magnitude H <= 7 (about 200 km and up), 1 for the fainter ones. Most are brighter
     than H 8; about one in eight is fainter.
   - recall = weighted share of those bodies found.
2. **GPU-hours** (lower is better): the program's run time summed over the four regions, in hours. A program that stops
   when more time would find nothing more costs less than one that fills the ceiling. The whole archive is several
   hundred such regions, so this is the cost of the search itself.

The run keeps the program with the highest recall and, among programs of equal recall, the one that used fewer GPU-hours.
Both numbers of every earlier program are in `history/`.

Candidates in the scored slots that match no known body are the program's new candidates. We check each one (other
archives, orbit fit, images); the real ones are the second goal of the search, so give them the `p` they deserve.

The program must not reach the network (it has none when graded) and must not contain designations of known bodies:
both are also checked automatically and score 0.

## What you have

- `/app/open/`: all OPEN_N regions, the four graded ones among them, with their known bodies labelled (OPEN_DET
  detections, OPEN_BODIES findable bodies). `open/scen/<sid>/` holds the files listed above; `open/labels/<sid>.npz` has
  `det_body` (per detection: the index of the known body it belongs to, -1 for none), `det_night` (night number per
  detection) and `known` (a JSON list, one entry per body index: `name`, `H`, `weight`, `nights`);
  `open/labels/regions.json` lists the regions with their sizes and findable bodies and says which are graded. These are
  the grader's own label files. Read-only.
- `tools/score.py`: the grader's own scoring code. `python3 /app/tools/score.py OUT /app/open/labels GRADE --sids <sid>[,<sid>]
  --report GRADE/report.md` scores `OUT/<sid>/candidates.json` exactly as the grade does (slots, 80 % / 5 nights rule,
  weights) and writes the per-body table. Unlabelled candidates in the scored slots are listed in `GRADE/grade.json`
  under `leads`.
- `tools/subset.py`: cuts a region and its labels to a part of the sky, so a test runs in minutes on CPU:
  `python3 /app/tools/subset.py <sid> WORK --radius 1.5` (a disc) or `--near-bodies 0.5` (everything within 0.5 deg of a
  labelled detection of a findable body). The result has the layout of `/app/open`, so `score.py` works on it. A whole
  region takes a GPU to search in full.
- `/app/data/sample`: one small region in the same format, with no labels, used by `tools/check.py`.
- `submission/`: your program (it starts empty; every earlier program is in `history/`). `submission/finder.py` is the
  entry; any other files may sit next to it (500 MB at most).
- `history/`: every earlier program with its notes, its two rewards (`genNNN/reward.json`) and its outcome for every
  body of the graded regions (`genNNN/bodies.md`): found, or missed and how far the program got with it (in no
  candidate, mixed with other detections, linked on too few nights, linked but ranked below the cut). `genNNN/grade.json`
  is the grader's full output (its `leads` are the unmatched candidates in the scored slots) and
  `genNNN/out/<sid>/candidates.json` is what the program returned for each graded region. `BODIES.md` has one row per
  body over all generations, with its region, its index in that region's label file, name, H, nights per season, S/N and
  fitted orbit; `bodies.json` holds the same.
- `tools/check.py`: runs a fresh copy of `submission/` on the sample exactly as the grader runs it (budget, kill, no
  network) and checks the output format. It gives no score.
- Your machine has 8 CPU cores, 32 GB of memory and no GPU; the program must run on both (use CUDA when
  `torch.cuda.is_available()`). Memory-map the large files (`np.load(..., mmap_mode="r")`).
- Session: SESSION_LIMIT. Leave `submission/NOTES.md` saying what you changed, why, and what you would try next.

## Where to look

Everything is open: how to tell static sources from moving ones, which detections to trust, how to link across years
rather than seasons, how to fit orbits, how to rank linkages, when to stop. The feedback says which bodies every program
missed and at which step; the labels let you see those bodies in the data and measure a change before it is graded. A
change that helps on many regions is worth more than one that helps on the graded four. Do not limit yourself to known
methods; invent where it helps.
