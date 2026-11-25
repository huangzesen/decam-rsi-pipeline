"""z3q table: for a stack.json from an N3CV run, list top rows with z3q/zq/zc and whether they hit a truth body."""
import sys, json, csv, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_inject as S
truth = json.load(open(sys.argv[1])); d = json.load(open(sys.argv[2])); rows = list(csv.DictReader(open(sys.argv[3])))
N = int(sys.argv[4]) if len(sys.argv) > 4 else 60
cands, res, hr, dup = S.score(truth, d["candidates"], rows)
cls = [b["cls"] for b in truth["bodies"]]
print("rank   p     kind  z3q    zq    zc    hit")
for r, (c, h) in enumerate(zip(cands, res)):
    if r > N:
        break
    hs = "" if h is None else f"{cls[h[0]]}{h[0]}"
    print(f"{r:4d} {c['p']:.3f} {str(c.get('kind', '-')):>4} {c.get('z3q', float('nan')):6.2f} "
          f"{c.get('zq', float('nan')):5.2f} {c.get('zc', float('nan')):5.2f} {hs}")
