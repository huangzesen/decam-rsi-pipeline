import sys, json, os
sys.path.insert(0, "$RSI_SANDBOX/r1/gen149/workspace/submission")
os.chdir("$RSI_SANDBOX/r1/gen149/workspace/submission")
import numpy as np, vet
scen = "$RSI_SANDBOX/r1/gen149/tmp/scen_injR"
cs = json.load(open("$RSI_SANDBOX/r1/gen149/tmp/runs/out_baseR.json"))["candidates"][:60]
full = vet.measure(scen, cs, None, workers=1, log=lambda *a: None)
orig = vet._one_exposure_
cnt = {"n": 0}
def broken(*a):
    o, ok, st = orig(*a)
    cnt["n"] += 1
    return (o[: len(o) // 2], False, st) if cnt["n"] in (2, 5) else (o, ok, st)
vet._one_exposure_ = broken
part = vet.measure(scen, cs, None, workers=1, log=print)
os.environ["TNO_VET_PART"] = "0"; cnt["n"] = 0
old = vet.measure(scen, cs, None, workers=1, log=print)
nf = sum(f is not None for f in full); npt = sum(f is not None for f in part); no = sum(f is not None for f in old)
print("features full/part/old:", nf, npt, no)
for i in range(8):
    a, b = full[i], part[i]
    if a and b:
        print(i, "zlbno full %.2f part %.2f  nexp %d/%d npos %d/%d" % (a["zlbno"], b["zlbno"], a["nexp"], b["nexp"], a["npos"], b["npos"]))
