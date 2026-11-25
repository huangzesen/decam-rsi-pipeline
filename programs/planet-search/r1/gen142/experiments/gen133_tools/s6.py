import sys, pickle, io, contextlib, itertools, os
from sim import *
tag = sys.argv[1]; scens = sys.argv[2].split(",")
pk = f"{T}/runs/{tag}data.pkl"
if os.path.exists(pk): data = pickle.load(open(pk, "rb"))
else:
    data = {s: load(tag, s) for s in scens}; pickle.dump(data, open(pk, "wb"))
src = open("s3.py").read().split('print("gen129')[0].replace('data = pickle.load(open(T+"/runs/r1data.pkl","rb"))', '')
exec(src)
print("gen129", run(dict(VET=0, PROM=0), True, 200))
print("gen132", run(dict(VET=1, PROM=0), True, 200))
print("gen133 default", run(dict(VET=1, PROM=1, PROM_Z0=6, PROM_Z1=12, PROM_P0=0.45, PROM_P1=0.8, PROM_CHI=0.25, PROM_NP=3, PROM_DZL=2), True, 200))
for dzl, z0, p0, chi in itertools.product([0, 1, 2, 3], [5, 6, 7], [0.3, 0.45], [0.12, 0.25]):
    env = dict(VET=1, PROM=1, PROM_Z0=z0, PROM_Z1=12, PROM_P0=p0, PROM_P1=0.8, PROM_CHI=chi, PROM_NP=3, PROM_DZL=dzl)
    print((dzl, z0, p0, chi), run(env, False, 200))
