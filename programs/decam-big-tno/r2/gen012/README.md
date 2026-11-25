Run the search with the supplied catalogue directory:

```sh
python3 finder.py --scen /path/to/region --out /path/to/output --budget-s 1800
```

CUDA is used when available; the same method also runs on CPU. Input detection arrays are memory-mapped. Results are written atomically to `candidates.json`, with bounded intermediate groups in `trace.json`.

See `NOTES.md` for the method, experiments, measurements, and remaining work.
