# DECam moving-body search

Run one region with `python3 finder.py --scen DIR --out OUT --budget-s 1800`.
The finder uses CUDA when available and also runs on CPU. It reads the single-exposure detection arrays and exposure metadata from `DIR`. Each entry in `OUT/candidates.json` contains the original detection row numbers and a confidence for the linkage.

The search method, validation results and proposed next experiment are in [NOTES.md](NOTES.md).
