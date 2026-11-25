# Reference solver: r2's best program, and what to tailor per DECam case

Decision (2026-11-11). The archive work starts in a new thread; this file is its starting point. The r2 loop keeps running to gen050
as before.

## Which program

- **Reference: `record-0.3910-gen015`** in `rsi/runs/r2/programs.git` (export with
  `git --git-dir=rsi/runs/r2/programs.git archive record-0.3910-gen015 | tar -x -C DEST`). Reward 0.391 (52 of 133
  points) on the 10 sealed fields, 900 s per field on one A100. Its `NOTES.md` explains every stage, newest first.
- `main` (gen020) has the same score and the same 24 bodies, plus three passes that changed nothing on the sealed
  fields: track extension (gen016), weak-third-point triplets (gen017; linked two likely real bodies in dev), a 25-32 au
  prior shoulder (gen020). It also carries the exit-time telemetry of gen018-020: remove it before any production use
  (it holds every run up to 36 s; see gen018's notes).
- Baselines on the same grader: KBMOD stock config 0.015, r1 best (Opus 5.5) 0.271.
- Write-up of the algorithm and its precedents: private page <private-page-url>
  (rounds 1-20).

## What it does (one line per stage)

Per-CCD detection on the GPU into S/N maps (Gaussian matched filter at the header FWHM: within 0.6 % of optimal,
gen022) -> pixel static mask from the other nights, cosmic-ray cleaning, Community Pipeline WCS repair for exposures with
failed astrometry (gen012) -> a 40k-parameter CNN trained on self-labelled stamps drops artefacts -> seeds from single
exposures and night stacks -> heliocentric triplet linker, then seed-pair linking over distance and radial speed with a
bound-orbit pre-cut -> every hypothesis scored by sampling the S/N maps of all exposures, plus an orbit-population prior
(2.2 score units per nat) -> six-element orbit polish on the maps -> two-night candidates, deep two-night pairs in
leftover time (>= 9 exposures on >= 4 nights) -> top 200 per window with a confidence p. Distance 25.5-400 au.

## The envelope it was tested in

The 10 sealed fields: 5-20 exposures, 4-7 nights over 3-45 days, bands g r i z Y VR, exposures 35-600 s, all within 30
deg of the ecliptic (8 within 10 deg) and >= 40 deg from the Galactic plane. Cost 0.57 Perlmutter node-h for the 10
(109 exposures). Linking ends at 45-300 s of a 630 s link window, so time is not the limit at these sizes; 1800 s gave
the same result (gen015 regrade).

## What to tailor per case (2026-11-11 review)

1. **Orbit prior** (`finder.py` around `qr`, `qw`, `g`: distance peaks at 35 and 43 au with a 14 au tail beyond 44,
   inclination Gaussians of 3.5 and 16 deg). A body at 100 au with i = 40 deg loses about 20 score units, twice the
   typical body-to-chance margin. Far from the ecliptic every body is highly inclined, so flatten the prior or make it
   depend on ecliptic latitude; for dwarf-planet searches it must not suppress distant, inclined bodies.
2. **Crowded fields** (Galactic plane): untested. The mask blanks 12-19 % of pixels at high latitude and would blank
   most of a plane field. gen021's static subtraction (recover masked pixels from a template of other nights) gained 2
   faint bodies but lost 3 and is not in the reference. 2026 ED138 is the plane test case.
3. **Large windows**: above 55 exposures a group-coadd mode (gen013) turns on; it never ran on a graded field. DES deep
   fields (215 windows with > 1,000 exposures) need their own design.
4. **Sparse cadence**: it needs >= 3 nights on the same sky within weeks (two-night candidates exist but scored nothing).
   Measured 2026-11-11 (`linker/footprint/window_share.py`): of DECam's 845,585 exposures 655,655 are usable (g r i z
   Y VR, >= 30 s); 358,265 of them (55 %) sit in 27,201 windows of >= 3 nights (1 deg cell, 60-day bin), 280,428 in
   15,292 windows of >= 4. Only 8,499 windows (88,283 exposures, 13 % of usable) are inside the tested envelope (sky
   and <= 55 exposures). About 40 % of usable exposures are in no 3-night window. Plan built on this: `DECAM_PLAN.md`.
5. **Candidate purity**: near the faint limit about 20 of a window's top 25 are chance alignments. At archive scale
   use a stricter cut, or confirm with `rsi/candidates/` (novelty check, lead_confirm, two-month tests).
6. **Classifier** (`detcls.pt`): trained on a few practice fields; it labels its own data, so retraining per season or
   band is cheap.
7. **Brightness reach**: about V 24.5-25 with a window's stacks; a 1,000 km body (H 3) is V 20 / 23 / 24.7 / 26 at 50 /
   100 / 150 / 200 au, so it reaches such bodies to about 150 au, 600 km bodies to about 110 au.

## Validation before scaling (proposed, not run)

- Blind runs on real cases: 2026 ED138's 2016 DECam nights (5 nights, crowded plane) and 2017 OF201 (~90 au).
- Planted bodies in a few windows far from the ecliptic, with high inclinations and 80-200 au distances.
- Grading or HPC runs need approval (transfer tests).
