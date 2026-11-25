Run `python3 finder.py --scen DIR --out OUT --budget-s 1800`.

The search filters stationary sources, links admissible orbital arcs within and across observing seasons, fits and
extends Keplerian trajectories, and writes ranked detections to `OUT/candidates.json`. It uses CUDA when available and
runs on the CPU for development. Large input arrays are memory-mapped. Runtime needs only the supplied region: no
catalogue, no network, no stored identities, positions, orbits or region-specific settings.

This generation joins three earlier, independent pieces (the population-ranked local confirmation queue,
`orbital_priority.py`; the perihelion factor, `orbit_plausibility.py`; the static-shadow evidence, `static_shadow.py`)
and sizes the local confirmation queue to the device instead of to a fixed count. See `NOTES.md`.

`FINDER_*` environment variables are optional development switches and are unset in ordinary use.
