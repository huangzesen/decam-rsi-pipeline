Run `python3 finder.py --scen DIR --out OUT --budget-s 1800` for one region.
The program writes ranked original detection rows and their linkage confidences to
`OUT/candidates.json`. It uses CUDA when available and also runs on CPU.
The native search and verification use a local C++17 compiler.
