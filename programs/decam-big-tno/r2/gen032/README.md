Run `python3 finder.py --scen DIR --out OUT --budget-s 1800`.

The search filters stationary sources, links admissible orbital arcs within and across observing seasons, fits and extends Keplerian trajectories, and writes ranked detections to `OUT/candidates.json`. It uses CUDA when available and supports CPU development runs. Large input arrays are memory-mapped. Runtime requires only the supplied region; no external catalogue or network access is used.

`NOTES.md` describes this generation's changes and measurements. `FINDER_*` environment variables are optional development switches and are unset during ordinary use.
