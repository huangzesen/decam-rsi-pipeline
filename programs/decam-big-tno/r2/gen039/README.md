Run `python3 finder.py --scen DIR --out OUT --budget-s 1800` to write `OUT/candidates.json`.

The continuous-distance search builds its C++17 worker in a temporary directory. The finder uses CUDA when available and also runs on a CPU. The method and validation results are in `NOTES.md`.
