Run `python3 finder.py --scen DIR --out OUT --budget-s SECONDS`.

Uses CUDA when available, with a CPU fallback. Writes `OUT/candidates.json`
and optional intermediate linkage traces. See NOTES.md for the method,
measurements, and limitations.
