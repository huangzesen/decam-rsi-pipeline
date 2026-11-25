from sweep import *
V = [("baseline", {}, pp_none)]
for q0 in (0, 0.5, 0.85, 0.95, 1.0):
    V.append((f"Q0={q0}", {"RANK_Q0": q0}, pp_none))
for q0 in (0.7, 0.9):
    for r in (10, 15, 40, 80):
        V.append((f"Q0={q0} R={r}", {"RANK_Q0": q0, "RANK_R": r}, pp_none))
for w in (0.5, 0.6, 0.8, 0.9, 1.0):
    V.append((f"STACK_W={w}", {"STACK_W": w}, pp_none))
for ps in (0.6, 0.7, 0.9, 0.95):
    V.append((f"P_STRONG={ps}", {"P_STRONG": ps}, pp_none))
V.append(("CAL=0", {"CAL": 0}, pp_none))
for k, vals in (("CAL_R", (0.1, 0.5, 1.0)), ("CAL_FMIN", (0.3, 0.8)), ("CAL_PHI", (0.2, 0.6)), ("CAL_N0", (0, 6)), ("CAL_PMAX", (0.7, 1.01))):
    for x in vals:
        V.append((f"{k}={x}", {k: x}, pp_none))
V.append(("ILEAVE=0", {"ILEAVE": 0}, pp_none)); V.append(("ILEAVE=2", {"ILEAVE": 2}, pp_none))
for a in (0.3, 1.0):
    V.append((f"pp_rank a={a}", {}, pp_rank(a)))
for a in (0.5, 1.0):
    V.append((f"pp_quant a={a}", {}, pp_quant(a)))
for k in (0, 5):
    V.append((f"pp_conf k={k}", {}, pp_conf(k)))
import sys
rs = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
report(run(V, rs))
