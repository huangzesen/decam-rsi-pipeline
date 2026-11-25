"""Compare sim.py's post-hoc thinning with 'proper' thinning (drop bodies' raw linker/stack candidates BEFORE merge,
so rank_balance ranks, interleave p reassignment and dens_calib n_conf all respond)."""
import sys, json, time, numpy as np, io, contextlib
sys.path.insert(0, "$RSI_SANDBOX/r1/gen114/tmp/exp")
import sim, score_inject as si
variant = sim.parse_variant(sys.argv[1] if len(sys.argv) > 1 else "")
q = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
F = sim.load_finder()
for k, v in variant.items(): setattr(F, k, v)
data = []
for name, scen, run, truth in sim.SCEN[:3]:
    rows, nt, no = sim.nights(scen)
    T = json.load(open(truth))
    A = json.load(open(run + ".linker.json"))["candidates"]
    S = json.load(open(run + ".stack.json"))["candidates"]
    # tag raw candidates with body hit
    for X in (A, S):
        cs, res, _, _ = si.score(T, X, rows)
        for c, h in zip(cs, res):
            c["_b"] = -1 if h is None else h[0]
    w = np.array([sim.W[b["cls"]] if len(b["nights"]) >= 3 else 0 for b in T["bodies"]], float)
    data.append((name, rows, nt, no, T, A, S, w))
rng = np.random.default_rng(0)
t0 = time.time()
R_proper, R_post = [], []
for r in range(reps):
    pool_pr, pool_po, den = [], [], 0.0
    for k, (name, rows, nt, no, T, A, S, w) in enumerate(data):
        kb = rng.random(len(w)) < q
        den += (w * kb).sum()
        keep = lambda c: c["_b"] < 0 or kb[c["_b"]]
        # proper
        with contextlib.redirect_stdout(io.StringIO()):
            cs = F.rank_balance(F.merge([dict(c) for c in A if keep(c)], [dict(c) for c in S if keep(c)], nt, no))[:200]
        cs, res, _, _ = si.score(T, cs, rows)
        for c, h in zip(cs, res):
            pool_pr.append((float(c["p"]), k, -1 if h is None else h[0]))
        # post-hoc (sim.py)
        with contextlib.redirect_stdout(io.StringIO()):
            cs = F.rank_balance(F.merge([dict(c) for c in A], [dict(c) for c in S], nt, no))[:200]
        cs, res, _, _ = si.score(T, cs, rows)
        for c, h in zip(cs, res):
            hb = -1 if h is None else h[0]
            if hb < 0 or kb[hb]:
                pool_po.append((float(c["p"]), k, hb))
    for pool, R in ((pool_pr, R_proper), (pool_po, R_post)):
        pool.sort(key=lambda x: -x[0]); got = {(k, h) for p, k, h in pool[:75] if h >= 0}
        R.append(sum(data[k][7][h] for k, h in got) / den)
print(sys.argv[1:], f"proper {np.mean(R_proper):.4f}  posthoc {np.mean(R_post):.4f}  diff {np.mean(np.array(R_proper)-R_post):+.4f} +- {np.std(np.array(R_proper)-R_post)/np.sqrt(reps):.4f}  ({time.time()-t0:.0f}s)", flush=True)
