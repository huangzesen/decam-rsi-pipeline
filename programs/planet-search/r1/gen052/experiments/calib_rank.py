"""Cross-scenario calibration of stack-only p by a per-scenario quantile anchor (down-only logit shift).

Use inside finder.merge():  cs = calib(phys_demote(...), dict(n_exp=..., n_night=...))  -> then rank_balance in write().

Idea: the stack stage's p = sigmoid((z - 4.5) / 0.8), with z from a median/MAD null of refined window seeds.  In rich
scenarios (many exposures) the post-refinement noise maximum is larger than that null predicts, so the whole stack-only
population sits higher (INJ1 junk at p 0.45-0.82).  The bulk of a scenario's stack-only list is junk, so its quantile
CAL_Q is a scale-robust measure of where junk lands in THIS scenario.  If that anchor is above the reference background
CAL_PBG, every stack-only candidate's logit is shifted down by the excess (order among stack-only kept; strong ones
barely move in p because the sigmoid saturates).  If the anchor is at/below CAL_PBG the scenario is untouched: the rule
never raises any p (no gen007b-style forcing of a poor scenario's tail).  Linker (L) and matched (LS) p are unchanged.
"""
import math
import numpy as np

CAL_Q = 0.5      # quantile (by rank, from the top) of the scenario's stack-only list used as the junk anchor
CAL_PBG = 0.05   # p that the junk anchor is mapped to (reference background level); 0 or >= 1 disables
CAL_MIN_S = 20   # fewer stack-only candidates than this -> no change (anchor not reliable)


def _logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def calib(cs, scen=None):
    if not (0 < CAL_PBG < 1) or not cs:
        return cs
    try:
        S = [c for c in cs if c.get("src") == "S"]
        if len(S) < CAL_MIN_S:
            return cs
        ps = np.sort(np.array([float(c["p"]) for c in S]))[::-1]          # descending
        anchor = float(ps[min(len(ps) - 1, int(round(CAL_Q * (len(ps) - 1))))])
        shift = max(0.0, _logit(anchor) - _logit(CAL_PBG))
        if shift <= 0:
            return cs
        out = []
        for c in cs:
            if c.get("src") == "S":
                x = _logit(c["p"]) - shift
                out.append(dict(c, p=1 / (1 + math.exp(-x))))
            else:
                out.append(c)
        return sorted(out, key=lambda c: -float(c["p"]))
    except Exception:
        return cs
