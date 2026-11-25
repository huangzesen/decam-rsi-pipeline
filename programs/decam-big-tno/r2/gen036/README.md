Run one region with:

```sh
python3 finder.py --scen DIR --out OUT --budget-s 1800
```

The program uses CUDA when available and otherwise runs on the CPU. It reads
the supplied exposure catalogues, writes atomic checkpoints to
`OUT/candidates.json`, and reports intermediate linkage groups in
`OUT/trace.json`. It needs no network connection or external catalogue.

This version adds a range-marginalized orbital-motion priority for detection
pairs and reuses faint-ended seasonal pairs in local continuation. See
[NOTES.md](NOTES.md) for the method, measured comparisons, limitations, and
suggested next experiments.
