Run the finder on one DECam region with Python 3.11, NumPy, SciPy, PyTorch and a C++17 compiler.

```sh
python3 finder.py --scen /path/to/region --out /path/to/output --budget-s 1800
```

The finder writes ranked detection-row linkages to `candidates.json` in the output directory and replaces that file at checkpoints. CUDA is selected when available. Development measurements and the method are in [NOTES.md](NOTES.md).
