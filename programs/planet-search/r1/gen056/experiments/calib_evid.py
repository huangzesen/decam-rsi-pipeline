"""Evidence-based (look-elsewhere) cross-scenario calibration of stack-only p.

Stack p is a logistic in z = (score - median_null) / MAD with slope 0.8 (stackd to_cands).  The null is built from
refined window seeds, but the CEM refinement + CV re-rank searches a trial space that grows with the number of
exposures: the expected max of noise over T trials rises ~ log T, so a rich scenario (13 exp) lifts junk to a
higher z than a short one (7 exp) at equal truth.  We remove that shift:
    dz = BETA * ln(n_exp / N_REF)          (0 when n_exp <= N_REF: never boosts short scenarios)
    logit(p') = logit(p) - dz / 0.8        for src == "S" only (L and LS keep their p)
The list is then re-sorted by p (rank_balance re-sorts anyway).  numpy only, stand-alone.
"""
import math
import numpy as np

BETA = 1.2      # z-units of look-elsewhere shift per e-fold of exposures over N_REF (0 = off)
N_REF = 7       # exposure count of the shortest typical scenario (no discount at or below it)
Z_SLOPE = 0.8   # fixed: logistic width used by stackd's p(z); not a tunable


def _logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def calib(cs, scen):
    n_exp = int(scen.get("n_exp", N_REF) or N_REF)
    dz = BETA * max(0.0, math.log(max(n_exp, 1) / float(N_REF)))
    if dz <= 0 or not cs:
        return sorted(cs, key=lambda c: -float(c["p"]))
    shift = dz / Z_SLOPE
    out = []
    for c in cs:
        if c.get("src") == "S":
            p = float(c["p"])
            p2 = 1.0 / (1.0 + math.exp(min(60.0, shift - _logit(p))))
            out.append(dict(c, p=float(np.clip(p2, 1e-4, 0.999)), p_precal=p))
        else:
            out.append(c)
    return sorted(out, key=lambda c: -float(c["p"]))


if __name__ == "__main__":
    demo = [dict(p=q, src=s) for q, s in [(0.95, "L"), (0.82, "S"), (0.6, "S"), (0.45, "S"), (0.3, "L")]]
    for ne in (7, 10, 13):
        print(ne, [(c["src"], round(c["p"], 3)) for c in calib(demo, dict(n_exp=ne, n_night=4))])
