import sys, pickle, itertools
from sim import *
d1 = pickle.load(open(T+"/runs/r1data.pkl","rb")); d2 = pickle.load(open(T+"/runs/r2data.pkl","rb"))
src = open("s3.py").read().split('print("gen129')[0].replace('data = pickle.load(open(T+"/runs/r1data.pkl","rb"))', '')
data = dict(d1); data.update(d2)
EXR = None
exec(src.replace("EXR = {s:", "EXR = {s:"))
base = dict(VET=1, PROM=1, PROM_Z0=6, PROM_Z1=12, PROM_P0=0.45, PROM_P1=0.8, PROM_CHI=0.25, PROM_NP=3, PROM_DZL=2)
def ev(extra):
    global data
    out = []
    for dd in (d1, d2):
        data = dd
        out.append(run(dict(base, **extra), False, 200)[0])
    return out, sum(out)
print("gen132", [None])
print("default", ev(dict(VET_LO=1.5, VET_HI=4.5, VET_FL=0.5, VET_ST=0.3)))
for lo, hi, fl, st in itertools.product([1.5, 2.5, 3.5], [4.5, 6, 8], [0.2, 0.35, 0.5], [0.3]):
    if hi <= lo: continue
    print((lo, hi, fl, st), ev(dict(VET_LO=lo, VET_HI=hi, VET_FL=fl, VET_ST=st)))
