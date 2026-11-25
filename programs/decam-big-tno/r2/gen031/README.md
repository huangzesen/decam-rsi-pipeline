finder.py is the entry: `python3 finder.py --scen DIR --out OUT --budget-s SECONDS`.

The search is gen029's (seasonal orbit search, growth, carry, cross-season bridge, completion; its modules are next to
finder.py, unchanged). New in this generation: static_shadow.py, an evidence term of the confidence that asks the
complete catalogue whether the places of a linkage's detections light up again on other epochs. See NOTES.md.

Source code only: no stored detections, identities, positions, orbits or region-specific settings; no network.
