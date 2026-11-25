Run the archive search with:

```bash
python3 finder.py --scen DIR --out OUT --budget-s 1800
```

The program selects CUDA when available and also supports CPU execution. It memory-maps the input detection arrays, writes checkpoints to `OUT/candidates.json`, and emits optional linkage traces. It requires no network or external catalogue. See `NOTES.md` for the method, development measurements, and limitations.
