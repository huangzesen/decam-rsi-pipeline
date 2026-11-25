# DECam moving-body finder

Run one region with Python and a C++17 compiler.

```sh
python3 finder.py --scen /path/to/region --out /path/to/output --budget-s 1800
```

The finder uses CUDA when available and can run on CPU. It reads the exposure
metadata and detection arrays from the specified region. It checkpoints
the confidence-ranked linkages in `candidates.json`. Each `det` array
contains zero-based rows of the original exposure catalogues.

The search, development measurements and remaining uncertainties are
described in [NOTES.md](NOTES.md).
