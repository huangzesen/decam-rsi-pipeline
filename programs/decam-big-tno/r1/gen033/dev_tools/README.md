Dev tools of gen033 (not used by finder.py; the grader never runs them).

- plant_far.py SRC DST NB SEED RMIN RMAX : gen030's plant_colour.py with hot orbits (transverse speed 0.65-1.35 of the
  circular speed, radial speed +-0.4 of it, both uniform): the planted sets of the far tests (63-71 au, 86-98 au) and the
  "hot" control at 39.5-41.5 au.
- cls.py CAND.json DST [found.npy] : gen030's scorer (80 % of the detections, five nights), found share by night pattern.
- evalr.py CAND.json DST : found share by number of nights.
- xtab.py found.npy DST : found share by (nights, pattern), seasons, and nights with a detection of S/N >= 5.
