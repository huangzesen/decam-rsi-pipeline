Run `python3 finder.py --scen DIR --out OUT --budget-s 1800`.

The search uses CUDA when available and otherwise runs on CPU. It writes `OUT/candidates.json` and optional diagnostic traces. See `NOTES.md` for the method, measured comparisons and remaining work.
