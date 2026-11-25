Run `python3 finder.py --scen DIR --out OUT --budget-s 1800`.

The search selects CUDA when available and also runs on CPU. It writes
`OUT/candidates.json` and optional diagnostic linkage stages. See NOTES.md for
the method, validation, and remaining limitations.
