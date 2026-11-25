import sys, itertools, numpy as np, json
sys.path.insert(0, "$RSI_SANDBOX/r1/gen114/tmp/sweep")
import ssim

INJ = ["inj7d", "injF", "injD"]


def junkclone(s):
    m = s["hit"] < 0
    return dict(s, name=s["name"] + "_J", p=s["p"][m], praw=s["praw"][m], src=[x for x, k in zip(s["src"], m) if k],
                hit=s["hit"][m], truth=False, w=np.zeros(0), cls=[])


def settings(P):
    names = [s["name"] for s in P]
    inj = [n for n in INJ if n in names]
    out = []
    for q in (1.0, 0.3, 0.1):
        out.append((f"all q={q}", P, {n: q for n in inj}, 1 if q == 1 else 200))
    for perm in itertools.permutations((0.05, 0.3, 1.0)):
        out.append(("het " + "/".join(map(str, perm)), P, dict(zip(inj, perm)), 200))
    for d in inj:
        sub = [s for s in P if s["name"] != d]
        out.append((f"loso-{d} q=1", sub, {}, 1))
        out.append((f"loso-{d} q=.3", sub, {n: 0.3 for n in inj}, 200))
    dev = [s for s in P if not s["truth"]]
    if dev:
        for s in P:
            if s["truth"]:
                out.append((f"{s['name']}+dev q=1", [s] + dev, {}, 1))
                out.append((f"{s['name']}+dev q=.3", [s] + dev, {s["name"]: 0.3}, 200))
    jc = P + [junkclone(s) for s in P if s["truth"]]
    out.append(("+junkclones q=1", jc, {}, 1))
    out.append(("+junkclones q=.3", jc, {n: 0.3 for n in inj}, 200))
    out.append(("+junkclones het", jc, dict(zip(inj, (0.05, 0.3, 1.0))), 200))
    return out


def rich_settings(R):
    jc = R + [junkclone(s) for s in R if s["truth"]]
    out = []
    dev = [s for s in R if not s["truth"] and not s["name"].endswith("_J")]
    if dev:
        return [("rich+dev q=1", R, {}, 1), ("rich+dev q=.5", R, {"rich": 0.5, "injF": 0.5}, 200)]
    return [("rich q=1", R, {}, 1), ("rich q=.5", R, {"rich": 0.5, "injF": 0.5}, 200),
            ("rich+jc q=1", jc, {}, 1), ("rich+jc q=.5", jc, {"rich": 0.5, "injF": 0.5}, 200),
            ("rich+jc rich1 F.3", jc, {"injF": 0.3}, 200)]


# ---- custom post-process calibrations (operate on per-scenario dict, return new p array) ----
def pp_none(s, P):
    return s["p"]


def pp_rank(a):  # pure rank share: p' = 1 - rank/200 (+tiny p tiebreak)
    def f(s, P):
        r = np.arange(len(s["p"]))
        return (1 - r / 200.) * a + (1 - a) * s["p"]
    return f


def pp_quant(alpha):
    """per-scenario quantile mapping: map each scenario's praw to its rank-quantile in the pooled reference
    distribution of all scenarios' praw at the same rank, blended: p' = alpha*ref(rank) + (1-alpha)*p."""
    def f(s, P):
        L = max(len(x["p"]) for x in P)
        M = np.array([np.pad(x["p"], (0, L - len(x["p"])), constant_values=0) for x in P])
        ref = np.median(M, 0)[:len(s["p"])]
        return alpha * ref + (1 - alpha) * s["p"]
    return f


def pp_conf(k):
    """slot allocation by estimated real count: n_est = #(praw>=0.5) ; boost ranks < n_est + k by +1"""
    def f(s, P):
        n = int((s["praw"] >= 0.5).sum()) + k
        p = s["p"].copy()
        p[:n] += 1.0
        return p
    return f


def combine(a, b, name):
    f = ssim.F()
    pr = np.concatenate([a["praw"], b["praw"]]); o = np.argsort(-pr, kind="stable")[:200]
    hit = np.concatenate([a["hit"], np.where(b["hit"] >= 0, b["hit"] + len(a["w"]), -1)])[o]
    r = np.arange(len(o)); q = f.RANK_Q0 * np.exp(-r / f.RANK_R) if f.RANK_Q0 > 0 else 0 * r
    p = 1 - (1 - pr[o]) * (1 - q)
    src = [(a["src"] + b["src"])[i] for i in o]
    return dict(name=name, p=p, praw=pr[o], src=src, hit=hit, w=np.concatenate([a["w"], b["w"]]),
                cls=a["cls"] + b["cls"], truth=True)


def apply_pp(P, fn):
    return [dict(s, p=fn(s, P)) for s in P]


def run(variants, reps_scale=1.0, quiet=False):
    base = None
    res = {}
    for lab, v, pp in variants:
        P0 = ssim.prepare(v)
        d = {x["name"]: x for x in P0}
        R0 = [combine(d["injD"], d["inj7d"], "rich")] + [x for x in P0 if x["name"] not in ("injD", "inj7d")]
        P = apply_pp(P0, pp)
        R = apply_pp(R0, pp)
        row = []
        for sl, PP, qs, reps in settings(P) + rich_settings(R):
            wu, r, cl = ssim.gscore(PP, qs, reps=max(1, int(reps * reps_scale)), seed=1)
            row.append((sl, wu, r, cl))
        res[lab] = row
    ssim.save()
    return res


def report(res, baselab="baseline"):
    b = res[baselab]
    print("settings:", " | ".join(f"{i}:{x[0]}" for i, x in enumerate(b)))
    print(f"{'baseline':34s}", " ".join(f"{x[1]:6.2f}" for x in b))
    for lab, row in res.items():
        if lab == baselab:
            continue
        d = [x[1] - y[1] for x, y in zip(row, b)]
        print(f"{lab:34s}", " ".join(f"{x:+6.2f}" for x in d), f" min {min(d):+.2f} sum {sum(d):+.2f}")
