# DECam moving-body finder

Run one region per invocation.

```bash
python3 finder.py --scen DIR --out OUT --budget-s 1800
```

`OUT/candidates.json` contains confidence-ranked linkages. Each `det` list uses row numbers from the input region's detection arrays. The finder checkpoints that file during the search and can also write proposal groups to `OUT/trace.json`.

The search uses CUDA when available and has a CPU implementation. It builds its native orbit routines from the accompanying C++ sources. [Method notes](NOTES.md) describe the search and local measurements.
