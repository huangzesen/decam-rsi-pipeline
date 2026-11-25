from sweep import *
V = [("baseline", {}, pp_none)]
for w in (0.4, 0.5, 0.55, 0.6, 0.65):
    V.append((f"STACK_W={w}", {"STACK_W": w}, pp_none))
for w in (0.6, 0.65):
  for q0, r in ((0.9, 40), (0.8, 30), (0.7, 40)):
    V.append((f"SW={w} Q0={q0} R={r}", {"STACK_W": w, "RANK_Q0": q0, "RANK_R": r}, pp_none))
V.append(("Q0=0.8 R=30", {"RANK_Q0": 0.8, "RANK_R": 30}, pp_none))
import sys
rs = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
report(run(V, rs))
