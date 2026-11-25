Run `python3 finder.py --scen DIR --out OUT --budget-s 1800`.

The program selects CUDA when available and also runs on CPU. It writes
`OUT/candidates.json` throughout the search, with detection row numbers and
confidence, and an optional bounded `trace.json`. Input arrays are memory-mapped.
It uses the detection catalogues and exposure metadata; it needs no network,
labels, external ephemerides, or trained catalogue of objects.

`NOTES.md` describes the method, development measurements, and remaining limits.
`FINDER_*` environment variables are optional development switches. Normal runs
need none of them.
