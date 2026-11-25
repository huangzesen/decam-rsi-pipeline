Run the search on one region:

```sh
python3 finder.py --scen /path/to/region --out /path/to/output --budget-s 1800
```

CUDA is used when available; the same method runs on CPU. The detection arrays are memory-mapped. `candidates.json` is
written atomically several times during the run, `trace.json` holds bounded intermediate groups.

`finder.py` is gen012's program; `fifth_night.py` is this generation's addition (the fifth-night check of the sparse
pass's four-detection linkages). See `NOTES.md` for the method, the measurements and what to try next.
