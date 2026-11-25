Run one DECam field with Python 3.11.

```sh
python3 finder.py --scen /path/to/region --out /path/to/output --budget-s 1800
```

The search uses the exposure metadata and memory-mapped detection arrays in the input directory. It uses CUDA when available and compiles its native search workers with a local C++17 compiler. It checkpoints ranked detection-row linkages in `candidates.json`. Method and validation details are in `NOTES.md`.
