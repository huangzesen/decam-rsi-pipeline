"""validate_runs.py RUNDIR... : the check.py test on finished runs (run_node.py / finder_node.py output folders).

For each <run>/info.json + candidates.json: status and seconds (the run was made the grader's way: fresh copy, stripped
environment, one A100, killed 30 s after the budget), then the same validation as harness/task/tools/check.py: a dict
with a "candidates" list of at most 200, each with a finite epoch_mjd, a 6-number finite state and p in [0, 1]; states
the grader would drop (unbound, or closer than 25 au) are counted."""
import json, math, os, sys
GM = 2.959122082855911e-4
ok_all = True
for d in sys.argv[1:]:
    info = json.load(open(os.path.join(d, "info.json")))
    msgs, n, n_drop = [], 0, 0
    f = os.path.join(d, "candidates.json")
    if not os.path.exists(f):
        msgs.append("no candidates.json")
    else:
        try:
            c = json.load(open(f))["candidates"]; n = len(c)
            if n > 200:
                msgs.append(f"{n} candidates (> 200)")
            for k, x in enumerate(c):
                s = [float(v) for v in x["state"]]; e = float(x["epoch_mjd"]); p = float(x["p"])
                if len(s) != 6 or not all(map(math.isfinite, s + [e, p])) or not 0 <= p <= 1:
                    msgs.append(f"candidate {k}: bad state/epoch/p"); continue
                r = math.sqrt(sum(v * v for v in s[:3])); v2 = sum(v * v for v in s[3:])
                if r < 25 or v2 / 2 - GM / r >= 0:
                    n_drop += 1
        except Exception as ex:
            msgs.append(f"invalid: {type(ex).__name__}: {ex}")
    good = info["status"] == "ok" and info["seconds"] <= 930 and not msgs
    ok_all &= good
    print(f"{os.path.basename(os.path.normpath(d))}: {info['status']}, {info['seconds']:.0f} s on {info.get('host')} GPU {info.get('gpu')}, "
          f"{n} candidates, {n_drop} would be dropped" + "".join("\n   " + m for m in msgs))
print("ALL OK" if ok_all else "PROBLEMS FOUND")
