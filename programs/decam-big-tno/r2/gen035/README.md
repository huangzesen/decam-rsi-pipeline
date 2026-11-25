Run `python3 finder.py --scen DIR --out OUT --budget-s 1800`.

The program memory-maps the detection catalogue, uses CUDA when available, and writes ranked linkages to
`OUT/candidates.json` and its intermediate groups to `OUT/trace.json`. CPU execution is supported for testing.

See NOTES.md for the method, what changed in this generation, the measurements and what to try next.
