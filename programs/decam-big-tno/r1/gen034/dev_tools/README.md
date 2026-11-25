Dev tools of gen034 (not used by finder.py; the grader never runs them). Paths inside are those of the session.

- plant_h.py SRC DST NB SEED : gen033's plant_far.py (hot orbits, one redness per body) with a distance mix (35 % 29-38 au,
  55 % 38-48 au, 10 % 48-65 au) and r magnitudes with pdf ~ 10^(0.6 m) on 20.5-24.5, so that the planted bodies have a
  spread of absolute magnitudes: `hv = mr + 0.4 - 5 log10(r (r - 0.8))` from truth.npz says which ones would score.
- eval_h.py CAND.json DST [MULT] : weighted found share of the planted bodies that would score (hv <= 8.4, weight 2 for
  hv <= 7.4), by distance, and a slot simulation (random subsets of the planted bodies at real-sky density).
- sim2.py PLANTED_CAND.json DST UNPLANTED_CAND.json : the slot simulation of NOTES.md - true candidates of a planted run
  against the chance candidates of an unplanted run (x1 / x3 / x10 / x30), for variants of the H factor recomputed from
  the keys of candidates.json (hv, arc, seasons, hfac).
- audit2.py CAND.json SCEN PMIN x RADIUS : gen009's audit.py with a search radius: every catalogue detection within
  RADIUS arcsec of the two-body track of each candidate with p >= PMIN.
- cls.py : gen030's scorer by night pattern.
