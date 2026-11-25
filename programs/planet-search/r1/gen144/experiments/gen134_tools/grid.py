import sys, itertools, ana
scens = sys.argv[1:]
data = ana.load("base", scens)
print("base", ana.evaluate(data, "old", env=dict(VET_WF=0)))
print("gen133-suggested", ana.evaluate(data, "old", env=dict(VET_WF=0, VET_LO=2.5, VET_HI=8, VET_FL=0.2), verbose=True))
for lo, hi, fl in itertools.product([1.5, 2.5, 3.5], [4.5, 6, 8], [0.5, 0.35, 0.2, 0.1]):
    print((lo, hi, fl), ana.evaluate(data, "old", env=dict(VET_WF=0, VET_LO=lo, VET_HI=hi, VET_FL=fl)))
