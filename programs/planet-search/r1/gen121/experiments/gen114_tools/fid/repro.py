import sys, json, numpy as np
sys.path.insert(0, "$RSI_SANDBOX/r1/gen114/tmp/exp")
import sim
F = sim.load_finder()
for name, scen, run, truth in sim.SCEN[:3]:
    rows, nt, no = sim.nights(scen)
    A = json.load(open(run + ".linker.json"))["candidates"]
    S = json.load(open(run + ".stack.json"))["candidates"]
    cs = sim.final_list(F, A, S, nt, no, {})
    O = json.load(open(run))["candidates"]
    dp = max(abs(float(a["p"]) - b["p"]) for a, b in zip(cs, O))
    ds = max(max(abs(float(x) - y) for x, y in zip(a["state"], b["state"])) for a, b in zip(cs, O))
    de = max(abs(float(a["epoch_mjd"]) - b["epoch_mjd"]) for a, b in zip(cs, O))
    print(name, len(cs), len(O), "max|dp|", dp, "max|dstate|", ds, "max|depoch|", de)
