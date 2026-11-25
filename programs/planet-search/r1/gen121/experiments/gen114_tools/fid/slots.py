import sys, numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen114/tmp/exp")
import sim
for vs in ["", "RANK_Q0=0.9", "RANK_Q0=0", "STACK_W=0.5", "CAL=0"]:
    F = sim.load_finder(); P = sim.prepare(F, sim.parse_variant(vs))
    pool = sorted([(p, k, int(h), i) for k, s in enumerate(P) for i, (p, h) in enumerate(zip(s["p"], s["hit"]))], key=lambda x: -x[0])[:75]
    per = [sum(1 for x in pool if x[1] == k) for k in range(len(P))]
    maxr = [max([x[3] for x in pool if x[1] == k] or [-1]) for k in range(len(P))]
    hits = [len({x[2] for x in pool if x[1] == k and x[2] >= 0}) for k in range(len(P))]
    dups = sum(1 for k in range(len(P)) for h in set(x[2] for x in pool if x[1]==k and x[2]>=0) if sum(1 for x in pool if x[1]==k and x[2]==h)>1)
    t25 = [len({h for h in s["hit"][:25] if h >= 0}) for s in P]
    print(f"{vs or 'base':12s} slots/scen {per} deepest rank {maxr} bodies-in-cut {hits} bodies-in-own-top25 {t25} bodies-with-dup-in-cut {dups} score {sim.global_score(P)[0]:.4f}")
