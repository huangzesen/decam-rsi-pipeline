Run one region from this directory.

```bash
python3 finder.py --scen /path/to/region --out /path/to/output --budget-s 1800
```

The finder writes `candidates.json` in the output directory. Each record gives the original single-exposure detection rows in `det` and the linkage confidence in `p`. Available CUDA devices are used automatically. The method and validation measurements are in [NOTES.md](NOTES.md).
