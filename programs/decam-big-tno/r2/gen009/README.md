Run `python3 finder.py --scen DIR --out OUT --budget-s 1800`.

CUDA is used when present; the same code runs on the CPU. The program reads only the region's catalogue and exposure
metadata, writes `OUT/candidates.json` as it goes and `OUT/trace.json` with its intermediate groups.
`NOTES.md` has the method, what was changed against gen008 and what was measured.
