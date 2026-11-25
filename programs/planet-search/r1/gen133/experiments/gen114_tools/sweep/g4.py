from sweep import *
V = [("baseline", {}, pp_none)]
for q0 in (0.5, 0.85, 1.0):
    V.append((f"Q0={q0}", {"RANK_Q0": q0}, pp_none))
for q0, r in ((0.7, 15), (0.7, 40), (0.9, 40), (0.8, 30), (0.9, 15)):
    V.append((f"Q0={q0} R={r}", {"RANK_Q0": q0, "RANK_R": r}, pp_none))
for w in (0.5, 0.6, 0.8, 0.9):
    V.append((f"STACK_W={w}", {"STACK_W": w}, pp_none))
V.append(("CAL=0", {"CAL": 0}, pp_none))
for k, vals in (("CAL_R", (0.1, 0.5)), ("CAL_FMIN", (0.3, 0.8)), ("CAL_PHI", (0.2, 0.6)), ("CAL_N0", (0, 6)), ("CAL_PMAX", (0.7, 1.01))):
    for x in vals:
        V.append((f"{k}={x}", {k: x}, pp_none))
for ps in (0.6, 0.95):
    V.append((f"P_STRONG={ps}", {"P_STRONG": ps}, pp_none))
V.append(("ILEAVE=2", {"ILEAVE": 2}, pp_none))
for a in (0.3, 1.0):
    V.append((f"pp_rank a={a}", {}, pp_rank(a)))
for a in (0.5, 1.0):
    V.append((f"pp_quant a={a}", {}, pp_quant(a)))
V.append(("SW=0.6 Q0=0.9 R=40", {"STACK_W": 0.6, "RANK_Q0": 0.9, "RANK_R": 40}, pp_none))
import sys
rs = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
report(run(V, rs))
