Run `python3 finder.py --scen DIR --out OUT --budget-s SECONDS`.

The search reads the supplied single-exposure detections and exposure metadata, selects CUDA when available, and checkpoints `OUT/candidates.json` throughout the search. See NOTES.md for the method and validation.
