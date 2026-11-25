# Reference: a GPU tracklet-less linker (from an earlier archive search)

Not a finder: it links detections that are already in a catalogue. You may reuse any part of it.

- `link_cpu.py`: the algorithm on the CPU (read its docstring first). For each assumed distance 1/gamma it transforms
  every detection into the Bernstein-Khushalani-like frame where a body at that distance moves on a nearly straight
  line (light time and radial gravity included), pairs single detections on different nights that a bound orbit can
  join (no same-night tracklets needed), turns each pair into a state (alpha, beta, alpha-dot, beta-dot) at the
  window's mid-time, hashes states into 4-D cells, and verifies cells with >= minpairs pairs by a straight-line fit
  and extension to every exposure (>= minnights nights).
- `link_gpu.py`: the same in PyTorch on a GPU (pairs by exposure-pair batches, keys sorted on the GPU, verification on
  the GPU). Two-stage mode `--gdot-tol 0 --refine --polish` gives the full grid's >= 5-night linkages for ~1/12 of the
  cost. `--fused` uses Triton kernels (CUDA only).
- `orbits.py`: two-body helpers (elements <-> state, Earth position, frames).

Input it expects (`DATA.npz` + `DATA.json`): per exposure `exp_t` (MJD), `exp_obs` (observer heliocentric position,
au, equatorial), per detection `det_ra`, `det_dec` (deg), `det_exp` (exposure index); the JSON holds the frame centre
`ra0`, `dec0` and `seasons_mjd` (list of [start, end] windows). Output: linkages (lists of detection indices) with
their fitted states in the transformed frame.

Measured before (on DECam catalogue detections of one 3 deg^2 field, 127 exposures over ~7 seasons, ~300 thousand
detections): full grid 474 s on an NVIDIA A100, 355 s on a GH200; the two-stage mode 38 s; it recovered a known
~90 au dwarf-planet candidate blind as the top of 534 five-night candidates. Recall on injected bodies 98% with its
index/veto cuts.

Notes for this task: it was written for windows of a whole season with >= 5 nights; the scenarios here are 3-60 days
with 4+ nights. Apple's `mps` backend has no float64, so run it with `--fp32` (or keep the state arithmetic on the
CPU), and `--fused` will not run. It starts from a catalogue: detection (and anything fainter than single-exposure
detection) is up to you.
