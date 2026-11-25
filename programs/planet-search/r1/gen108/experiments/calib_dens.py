"""Cross-scenario p calibration by per-scenario stack-only junk density (gen040 experiment, stand-alone).

Use in finder.merge():  cs = calib(phys_demote(...), dict(n_exp=..., n_night=...))  -> then rank_balance in write().

Idea: real stack-only (src "S") bodies are the faint end of the same population whose brighter members the linker
finds, so a scenario should hold about CAL_R real high-p stack-only candidates per high-p linker-found (L/LS) one.
High-p stack-only candidates beyond that expectation are junk (the stack null is too generous in that scenario), so
every S candidate's p is multiplied by the estimated purity f = min(1, (CAL_R * n_conf + N0) / n_S_hi).  A scenario
with few high-p stack-only candidates (short arcs, or a clean stack) is untouched (f = 1).  The scenario's own order
is kept: after the shrink, p is made non-increasing along the incoming order (running minimum), so only the
cross-scenario calibration (who wins the global top-25N cut) changes, not the tuned interleave/phys order.
No orbit-shape tests (gen039 showed real S hits often have odd fits).
"""
import numpy as np

CAL_R = 0.5    # expected real high-p stack-only candidates per high-p linker-found (L / LS) candidate (0 = off)
CAL_PHI = 0.4  # p threshold defining "high-p" for the counts
N0 = 3.0       # fixed: free allowance of high-p stack-only candidates per scenario (protects linker-poor scenarios)
F_MIN = 0.25   # fixed: floor on the shrink factor


def calib(cs, scen=None):
    if CAL_R <= 0 or len(cs) < 2:
        return cs
    try:
        order = sorted(range(len(cs)), key=lambda i: -float(cs[i]["p"]))   # incoming order (already p-sorted)
        cs = [cs[i] for i in order]
        p = np.clip(np.array([float(c["p"]) for c in cs]), 0.0, 1.0)
        isS = np.array([c.get("src") == "S" for c in cs])
        hi = p >= CAL_PHI
        n_conf = int((hi & ~isS).sum())
        n_s = int((hi & isS).sum())
        f = 1.0 if n_s <= 0 else float(np.clip((CAL_R * n_conf + N0) / n_s, F_MIN, 1.0))
        if f >= 1.0:
            return cs
        q = np.where(isS, p * f, p)
        q = np.minimum.accumulate(q)          # keep the scenario's own order (non-increasing p)
        return [dict(c, p=float(x)) for c, x in zip(cs, q)]
    except Exception:
        return cs


if __name__ == "__main__":
    # toy: rich scenario (18 confirmed, 22 high-p S junk) vs short one (6 confirmed, 2 S)
    rich = [dict(p=0.99 - 0.01 * k, src="LS" if k % 3 == 0 else "L") for k in range(18)] + \
           [dict(p=0.82 - 0.017 * k, src="S") for k in range(22)]
    short = [dict(p=0.9 - 0.1 * k, src="L") for k in range(6)] + [dict(p=0.32, src="S"), dict(p=0.28, src="S")]
    for name, cs, sc in (("rich", rich, dict(n_exp=13, n_night=6)), ("short", short, dict(n_exp=7, n_night=4))):
        out = calib(cs, sc)
        print(name, [round(c["p"], 2) for c in out][16:26])
