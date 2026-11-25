Run `python3 finder.py --scen REGION_DIRECTORY --out OUTPUT_DIRECTORY --budget-s 1800`.

The program selects CUDA when available and otherwise uses CPU workers. Large detection arrays are memory mapped. It writes `candidates.json` atomically throughout the search and records intermediate linkages in `trace.json`.

The search combines static-source rejection, parallax and seasonal seeds, bound Kepler-orbit fits, and uncertainty-guided continuation across exposures. Three-night completion uses curvature evidence to order its work, preserves uncertainty in short arcs, and measures accidental-association rates in the catalogue actually queried. See NOTES.md for changes and validation.
