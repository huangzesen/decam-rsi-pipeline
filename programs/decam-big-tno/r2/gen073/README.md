# DECam moving-object finder

```sh
python3 finder.py --scen DIR --out OUT --budget-s 1800
```

The finder reads single-exposure catalogues and observing geometry from the supplied region and writes ranked linkages to `OUT/candidates.json`. Detection indices refer to the original input rows. It uses CUDA when available and includes a CPU implementation. Search and validation details are in `NOTES.md`.
