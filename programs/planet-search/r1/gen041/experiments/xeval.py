"""Cross-scenario offline evaluation of the wrapper merge on kept intermediates.

python exp/xeval.py SUBDIR RUNDIR [VAR=VAL ...]
RUNDIR holds injK.json.linker.json / injK.json.stack.json for INJ1..3 in tmp.  VAR=VAL set wrapper module globals.
Prints per-scenario top-25 hits and the GLOBAL top-25*N weighted hits (A=1, B=2, C=4) - the grader's cut.
"""
import sys, os, json, csv, math, importlib.util, io, contextlib
import numpy as np
SUB = os.path.abspath(sys.argv[1]); RUN = os.path.abspath(sys.argv[2])
T = "$RSI_SANDBOX/r1/gen041/tmp"
sys.path.insert(0, os.path.join(SUB, "linker"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_inject as SI
spec = importlib.util.spec_from_file_location("wrap", os.path.join(SUB, "finder.py"))
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
CAL = None
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    if k == "CALIB":
        sp = importlib.util.spec_from_file_location("cal", v); CAL = importlib.util.module_from_spec(sp); sp.loader.exec_module(CAL)
        continue
    if k.startswith("C."):
        setattr(CAL, k[2:], type(getattr(CAL, k[2:]))(float(v))); continue
    setattr(w, k, type(getattr(w, k))(float(v)) if not isinstance(getattr(w, k), str) else v)
WT = {"A": 1, "B": 2, "C": 4}
pool = []
tot = 0
for i in (1, 2, 3, 4):
    scen = f"{T}/INJ{i}"
    lj, sj = f"{RUN}/inj{i}.json.linker.json", f"{RUN}/inj{i}.json.stack.json"
    if not os.path.exists(lj):
        continue
    rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
    nt = {}
    for r in rows:
        t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
        nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
    night_t = np.array([v[0] for v in nt.values()]); night_o = np.array([v[1] for v in nt.values()])
    with contextlib.redirect_stdout(io.StringIO()):
        if hasattr(w, "SCEN_INFO"):
            w.SCEN_INFO = dict(n_exp=len(rows), n_night=len(night_t))
        m = w.merge(w.load(lj), w.load(sj), night_t, night_o)
        if CAL is not None:
            m = CAL.calib(m, dict(n_exp=len(rows), n_night=len(night_t)))
        fin = w.rank_balance(m)[:200]
    truth = json.load(open(f"{scen}/truth.json"))
    cands, res, hr, dup = SI.score(truth, fin, rows)
    cls = [b.get("cls", b.get("class", "?")) for b in truth["bodies"]]
    tot += sum(WT.get(c, 0) for c in cls)
    t25 = {res[k][0] for k in range(min(25, len(res))) if res[k] is not None}
    junk25 = sum(1 for k in range(min(25, len(res))) if res[k] is None)
    print(f"INJ{i}: top25 bodies {len(t25)} (" + " ".join(f"{c}{sum(1 for j in t25 if cls[j]==c)}" for c in "ABC")
          + f") junk25 {junk25}; hits " + " ".join(f"{cls[j]}{r}:{p:.2f}" for j, (r, p) in sorted(hr.items(), key=lambda x: x[1][0])[:40]))
    for k, c in enumerate(cands):
        pool.append((float(c["p"]), i, res[k][0] if res[k] is not None else None, cls[res[k][0]] if res[k] is not None else None, c.get("src", "?")))
pool.sort(key=lambda x: -x[0])
N = len({x[1] for x in pool})
seen, wh = set(), 0
for p, i, j, c, _ in pool[:25 * N]:
    if j is not None and (i, j) not in seen:
        seen.add((i, j)); wh += WT[c]
print(f"GLOBAL top{25*N}: weighted hits {wh} / {tot}; per scen " +
      str({i: sum(1 for s in seen if s[0] == i) for i in (1, 2, 3, 4)}) + f"; p at cut {pool[25*N-1][0]:.3f}")
if os.environ.get("XV"):
    for k, (p, i, j, c, sr) in enumerate(pool[:95]):
        print(k, f"{p:.3f}", i, (c or "-") + sr, end=" | ")
    print()
