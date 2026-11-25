# DECam moving-body finder

```sh
python3 finder.py --scen DIR --out OUT --budget-s 1800
```

The finder reads the per-exposure detection arrays and telescope metadata in `DIR`. It checkpoints ranked linkages in `OUT/candidates.json`. Each linkage contains original detection row numbers and a confidence. CUDA is used when available. The native catalogue searches compile locally with a C++17 compiler. Method details and measurements are in `NOTES.md`.
