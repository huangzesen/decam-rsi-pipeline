Run `python3 finder.py --scen DIR --out OUT --budget-s SECONDS`.

The program memory-maps the detection catalogue, removes static and unreliable sources, links moving-source hypotheses, fits orbits, and writes ranked linkages to `OUT/candidates.json`. It uses CUDA when available and supports CPU execution. `NOTES.md` describes the changes and development measurements.
