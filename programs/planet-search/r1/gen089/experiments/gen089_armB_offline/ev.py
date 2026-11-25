import sys, json, glob, os, csv, numpy as np
sys.path.insert(0, '$RSI_SANDBOX/r1/gen089/workspace/submission/experiments')
sys.path.insert(0, '$RSI_SANDBOX/r1/gen089/workspace/submission/stackd')
import score_inject as si
D = sys.argv[1] if len(sys.argv) > 1 else '$RSI_SANDBOX/r1/gen089/tmp/armB/dump'
T = '$RSI_SANDBOX/r1/gen089/tmp/inj7/'
truth = json.load(open(T + 'truth.json')); rows = list(csv.DictReader(open(T + 'exposures.csv')))
def ev(cands):
    cs, res, hr, dup = si.score(truth, cands, rows, top=25)
    top = sum(1 for j, (r, p) in hr.items() if r < 25)
    byc = {}
    for j, (r, p) in hr.items():
        byc.setdefault(truth['bodies'][j]['cls'], []).append(r)
    return top, {k: sorted(v) for k, v in sorted(byc.items())}, dup
if __name__ == '__main__':
  for f in sorted(glob.glob(D + '/*.json')):
    c = json.load(open(f)); c = c.get('candidates', c) if isinstance(c, dict) else c
    print(os.path.basename(f), *ev(c))
