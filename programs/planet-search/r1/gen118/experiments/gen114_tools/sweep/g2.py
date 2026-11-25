from sweep import *
V = [("baseline", {}, pp_none)]
for q0 in (0.85, 1.0):
    V.append((f"Q0={q0}", {"RANK_Q0": q0}, pp_none))
V.append(("Q0=0.9 R=40", {"RANK_Q0": 0.9, "RANK_R": 40}, pp_none))
V.append(("STACK_W=0.6", {"STACK_W": 0.6}, pp_none))
V.append(("CAL_PHI=0.2", {"CAL_PHI": 0.2}, pp_none))
V.append(("CAL_N0=0", {"CAL_N0": 0}, pp_none))
V.append(("ILEAVE=2", {"ILEAVE": 2}, pp_none))
for a in (0.3, 1.0):
    V.append((f"pp_rank a={a}", {}, pp_rank(a)))
for a in (0.5, 1.0):
    V.append((f"pp_quant a={a}", {}, pp_quant(a)))
V.append(("STACK_W=0.6 Q0=0.9 R=40", {"STACK_W": 0.6, "RANK_Q0": 0.9, "RANK_R": 40}, pp_none))
import sys
rs = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
report(run(V, rs))
