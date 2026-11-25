Run `python3 finder.py --scen DIR --out OUT --budget-s 1800`.

The program selects CUDA automatically and also runs on CPU. It reads only the
region catalogue and exposure metadata. Results are written incrementally to
`OUT/candidates.json`; `OUT/trace.json` records intermediate linkage groups.

See `NOTES.md` for the method, development measurements, and limitations.
