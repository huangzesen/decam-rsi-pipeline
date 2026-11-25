Run the catalogue search with:

    python3 finder.py --scen DIR --out OUT --budget-s 1800

The program selects CUDA when available and otherwise uses CPU. It writes
OUT/candidates.json atomically during the search and keeps its last completed
result if the time budget runs out. See NOTES.md for the method and validation.
