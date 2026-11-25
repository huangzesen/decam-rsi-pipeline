Run the search with Python 3.11 and a C++17 compiler on the path.

```sh
python3 finder.py --scen REGION --out OUTPUT --budget-s 1800
```

The program checkpoints ranked linkages in `OUTPUT/candidates.json`. Each
`det` entry indexes the original input arrays for that region. See
`NOTES.md` for the search method and measurements.
