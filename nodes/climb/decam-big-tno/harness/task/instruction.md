# Find large distant bodies in thirteen years of DECam single exposures

You write a program that searches one patch of sky, observed by the Dark Energy Camera (DECam) on the Blanco 4 m
telescope from 2013 to 2026, for bodies of the outer solar system: trans-Neptunian objects of about 200 km and larger, and
anything bigger and farther (dwarf planets, Sedna-like bodies, a distant planet). It gets every source detected in every
single exposure of the patch and returns linkages: sets of detections that it claims are one moving body.

Your program is the search. It runs, unchanged, on patches you never see. Report every body you believe is real, known or
not: the known ones are how the program is scored, and the confident ones that match nothing known are checked by us as
possible discoveries.

## What the program gets

`finder.py --scen DIR --out OUT --budget-s SECONDS`, one patch per call. `DIR` holds:

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
| `region.json` | patch centre, radius, first and last MJD |

Rows are grouped by exposure. Every source is there: stars and galaxies (which sit still), noise peaks, cosmic rays,
artefacts near bright stars, satellites, asteroids, and the bodies you are after. A patch is about 55 square degrees with
400-1,000 exposures and 30-70 million detections. Exposures come in bursts: some seasons have a few nights, some have
dozens; a body crosses a patch in weeks (at 40 au) or stays for years (beyond 300 au).

## What it returns

`OUT/candidates.json`: a JSON list of `{"det": [row numbers], "p": number}`, one entry per body you believe is real, where
`p` is your confidence that the rows are one real moving body. At most 2,000 entries per patch are read (highest `p`
first). Extra keys are ignored.

## How it is scored

Each patch runs on one NVIDIA A100 with 40 GB of memory (CUDA; Python 3.11 with numpy, scipy, astropy, scikit-learn and PyTorch 2.11 with
Triton) with BUDGET_S seconds, no network, and a fresh copy of your submission folder; it is killed at the budget plus
30 s and whatever `candidates.json` holds then is scored. The same program runs on every patch.

- The candidates of all sealed patches are ranked together by `p`. The number of scored slots is twice the number of
  known bodies the patches hold (findable ones, below): room for every known body and as many new ones, but a wrong
  linkage with a high `p` takes a slot from a real body.
- A candidate finds a body when at least 80 % of its detections belong to that body and they cover at least 5 of its
  nights. Each body counts once.
- The bodies scored are the known trans-Neptunian objects with absolute magnitude H <= 8 (about 120 km and up) that
  have a detection on at least 5 nights in a patch; weight 2 for H <= 7 (about 200 km and up), 1 otherwise.
- reward = weighted share of those bodies found.

Your program must not contain lists, orbits or positions of known bodies, nor reach the network: submissions are
checked, and such a program scores 0.

## What you have

- `SAMPLE_DIR`: one small patch in the same format (a cut of a region that is not scored), with no answers. Plant your
  own bodies in it to test.
- `submission/`: the program to improve (it starts as the best program so far). `submission/finder.py` is the entry;
  any other files may sit next to it (500 MB at most).
- `history/`: earlier programs, their notes and rewards. Only the reward of each earlier program is known; no one sees
  per-patch results.
- `tools/check.py`: runs a fresh copy of `submission/` on the sample exactly as the grader runs it (budget, kill, no
  network) and checks the output format. It gives no score.
- Your machine has 8 CPU cores and no GPU; the program must run on both (use CUDA when `torch.cuda.is_available()`).
- Session: SESSION_LIMIT. Leave `submission/NOTES.md` saying what you changed, why, and what you would try next.

## Where to look

The starter is the plain search: a static-source cut, then tracklet-less linking over a grid of distances, one season at
a time inside 80 au and over pairs of seasons beyond. Everything is open: how to tell static sources from moving ones,
which detections to trust, how to link across years rather than seasons, how to fit orbits, how to rank linkages, how to
spend the GPU's time. Do not limit yourself to known methods; invent where it helps.
