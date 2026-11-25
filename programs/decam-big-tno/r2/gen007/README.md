Run `python3 finder.py --scen DIR --out OUT --budget-s 1800`.

CUDA is selected automatically; CPU execution is supported. The program reads
only the supplied exposure catalogues and metadata. It writes candidates
atomically during the search and records bounded intermediate linkage traces.
No external catalogue, labels, object list, or network service is needed.

The optional fused CUDA implementation validates itself against the reference
orbit model before use and falls back to PyTorch on failure. Development
`FINDER_*` switches are unnecessary for normal execution. See NOTES.md.
