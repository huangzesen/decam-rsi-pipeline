#!/usr/bin/env python3
"""searcha15: score the third 5 % (planet-search/archive15, conveyor unit a15) as searched by the conveyor, and send its leads on.
A copy of harness/searchldt5/scoreldt5.py (2026-11-19 15:30): outputs runs/conveyor/a15/{out,timing}/<sid> (timing
"seconds rc gen site device isolation"; every region ran the climb's best program at its start), pseudo-run runs/sa15,
labels private/truth_a15 (written by the conveyor build).
  scoreldt5.py score   grade every region with a timing (harness/grade.py; all regions ranked together, top K = 2 x findable)
                       -> runs/sa15/gens/a15-<gens>/grade/grade.json; prints recall per region
  scoreldt5.py leads   that grade's leads through harness/leads.py sa15 <tag> with DBT_SITE=nersc (novelty check and lead check follow)
2026-11-18, 2026-11-19."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG = [dict(r, n_exp=int(r["n_exp"])) for r in json.load(open(os.path.join(HERE, "..", "..", "planet-search", "units", "a15", "regions_search.json")))]
SRC = os.path.join(HERE, "runs", "conveyor", "a15")
RUN = os.path.join(HERE, "runs", "sa15")
TRUTH = os.path.join(HERE, "private", "truth_a15")
PY = os.path.expanduser("~/anaconda3/bin/python")


def config():
    os.makedirs(RUN, exist_ok=True)
    cfg = dict(sids=[r["sid"] for r in REG], names={r["sid"]: r["rid"] for r in REG},
               note="searcha15: the climb's best program at each region's start over the 62 scenarios of the third 5 % (archive15); conveyor runs on NERSC A100 and Vista GH200 (not sealed)")
    json.dump(cfg, open(os.path.join(RUN, "config.json"), "w"), indent=1)


def tag_of():
    gens = sorted({open(os.path.join(SRC, "timing", s)).read().split()[2] for s in os.listdir(os.path.join(SRC, "timing"))})
    return "a15-" + "+".join(gens)


def score():
    config(); by = {r["sid"]: r for r in REG}; tag = tag_of()
    G = os.path.join(RUN, "gens", tag); out = os.path.join(G, "out"); os.makedirs(out, exist_ok=True)
    timing, where = {}, {}
    for sid in os.listdir(os.path.join(SRC, "timing")):
        f = open(os.path.join(SRC, "timing", sid)).read().split()
        timing[sid] = float(f[0]); where[sid] = f"{f[3]} {f[2]}"
        dst = os.path.join(out, sid)
        if not os.path.islink(dst):
            os.symlink(os.path.join(SRC, "out", sid), dst)
    sids = sorted(timing, key=lambda s: by[s]["rid"])
    tj = os.path.join(G, "timing.json"); json.dump(timing, open(tj, "w"))
    subprocess.run([PY, "-I", os.path.join(HERE, "harness", "grade.py"), out, TRUTH, os.path.join(G, "grade"),
                    "--sids", ",".join(sids), "--timing", tj], check=True)
    g = json.load(open(os.path.join(G, "grade", "grade.json")))
    print(f"{tag}: {len(sids)} regions, reward {g['reward']:.4f}, slots {g['slots']}, cut p {g['cut_p']}, leads {len(g['leads'])}, GPU-h {sum(timing.values()) / 3600:.2f}")
    print("region  n_exp  findable found  leads  seconds  where")
    fa = fo = 0
    for s in sids:
        x = g["per_scenario"][s]; fa += x["known_findable"]; fo += x["known_found"]
        print(f"{by[s]['rid']:6s} {by[s]['n_exp']:5d}  {x['known_findable']:8d} {x['known_found']:5d}  {x['leads']:5d}  {x['seconds']:7.0f}  {where[s]}")
    print(f"total findable {fa}, found {fo} ({fo / max(fa, 1):.1%} unweighted)")


def leads():
    env = dict(os.environ, DBT_SITE="nersc")
    subprocess.run([PY, "-I", os.path.join(HERE, "harness", "leads.py"), "sa15", tag_of(), "--max", "2000"], env=env, check=True)


if __name__ == "__main__":
    {"score": score, "leads": leads}[sys.argv[1]]()
