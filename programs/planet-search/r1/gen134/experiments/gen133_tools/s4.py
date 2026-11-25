import sys, pickle, io, contextlib
from sim import *
data = pickle.load(open(T+"/runs/r1data.pkl","rb"))
sys.path.insert(0, "$RSI_SANDBOX/r1/gen133/workspace/submission")
import finder as F
for s, (rows_, labs, nb) in data.items():
    F.set_expt(list(csv.DictReader(open(f"{T}/scen/{s}/exposures.csv"))))
    cs = [dict(c, _i=i) for i, c in enumerate(rows_)][:200]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = F.vet_adjust(cs, [c["f"] for c in cs])
    print(s, buf.getvalue().strip().splitlines()[-1])
    for r, c in enumerate(out[:30]):
        l = labs[c["_i"]]; f = c["f"]
        print("   new %2d old %3d p %.2f %-2s %-2s zlb %.1f" % (r, c["_i"], c["p"], c["src"], l[1] if l else "-", f["zlbno"]))
