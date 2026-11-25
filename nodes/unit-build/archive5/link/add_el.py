#!/usr/bin/env python3
"""Add each known body's SBDB elements to archive5/link/known/<rid>.json as "el" = [a, e, inc, Om, om, M0, t0]
(ecliptic, radians, M0 at t0 = epoch in MJD), the layout link_cpu.evaluate() reads (obj_* arrays).
Usage: python3 -I add_el.py REG_A REG_B ..."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PS = os.path.dirname(os.path.dirname(HERE))
S = json.load(open(os.path.join(PS, "linker", "data_real", "sbdb", "tno_20261031.json")))
by = {r[S["fields"].index("pdes")]: dict(zip(S["fields"], r)) for r in S["data"]}
for rid in sys.argv[1:]:
    p = os.path.join(HERE, "known", rid + ".json"); K = json.load(open(p))
    for b in K["bodies"]:
        r = by[b["pdes"]]
        b["el"] = [float(r["a"]), float(r["e"]), float(np.radians(float(r["i"]))), float(np.radians(float(r["om"]))),
                   float(np.radians(float(r["w"]))), float(np.radians(float(r["ma"]))), float(r["epoch"]) - 2400000.5]
    json.dump(K, open(p, "w"))
    print(rid, len(K["bodies"]), "bodies with elements")
