#!/usr/bin/env python3
"""search38: score a finished search over the 5 % search regions and send its leads on.

  score38.py score NAME   fetch NAME_normal / NAME_big from NERSC (search38.py fetch), grade every region that has a
                          timing with harness/grade.py against private/truth38 (all regions ranked together, top
                          K = 2 x findable, as in the hill climb), print recall per region, and file the result as the
                          pseudo-run runs/s38/gens/NAME-GEN/ (grade/grade.json private)
  score38.py leads NAME   send that grade's leads through harness/leads.py s38 NAME-GEN with DBT_SITE=nersc (places
                          read from the scenarios on NERSC scratch; the novelty check and lead_check.py follow)
runs/s38/config.json lists every region's sid and its name (R004 ...), so lead ids carry the region name.
2026-11-13.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG = json.load(open(os.path.join(HERE, "harness", "search38", "regions38.json")))
REG = sorted(REG + [dict(rid="REG_C", sid="scen07", n_exp=787)], key=lambda r: r["rid"])   # 2026-11-15: r2b also runs REG_C
LOCAL = os.path.join(HERE, "runs", "search38")
RUN = os.path.join(HERE, "runs", "s38")
TRUTH = os.path.join(HERE, "private", "truth38x")     # links: truth38/*.npz plus truth/scen07.npz (REG_C), all H <= 8
PY = os.path.expanduser("~/anaconda3/bin/python")


def config():
    os.makedirs(RUN, exist_ok=True)
    cfg = dict(sids=[r["sid"] for r in REG], names={r["sid"]: r["rid"] for r in REG},
               note="search38: the hill climb's best program over the 5 % search's other 38 regions (not sealed)")
    json.dump(cfg, open(os.path.join(RUN, "config.json"), "w"), indent=1)
    return cfg


def score(name):
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "harness", "search38", "search38.py"), "fetch"], check=True)
    cfg = config(); by = {r["sid"]: r for r in REG}
    roots = [os.path.join(LOCAL, f"{name}_{t}") for t in ("normal", "big") if os.path.isdir(os.path.join(LOCAL, f"{name}_{t}"))]
    gens = {g for r in roots for g in os.listdir(r) if g.startswith("gen")}
    assert len(gens) == 1, gens
    gen = gens.pop(); tag = f"{name}-{gen}"
    G = os.path.join(RUN, "gens", tag); out = os.path.join(G, "out"); os.makedirs(out, exist_ok=True)
    timing = {}
    for r in roots:
        td = os.path.join(r, gen, "timing")
        for sid in (os.listdir(td) if os.path.isdir(td) else []):
            sec, rc = open(os.path.join(td, sid)).read().split()
            timing[sid] = float(sec)
            dst = os.path.join(out, sid)
            if not os.path.islink(dst):
                os.symlink(os.path.join(r, gen, "out", sid), dst)
    sids = sorted(timing, key=lambda s: by[s]["rid"])
    tj = os.path.join(G, "timing.json"); json.dump(timing, open(tj, "w"))
    subprocess.run([PY, "-I", os.path.join(HERE, "harness", "grade.py"), out, TRUTH, os.path.join(G, "grade"),
                    "--sids", ",".join(sids), "--timing", tj], check=True)
    g = json.load(open(os.path.join(G, "grade", "grade.json")))
    print(f"{tag}: {len(sids)} regions, reward {g['reward']:.4f}, slots {g['slots']}, cut p {g['cut_p']}, leads {len(g['leads'])}")
    print("region  n_exp  findable found  leads  seconds  status")
    fa = fo = 0
    for s in sids:
        x = g["per_scenario"][s]; fa += x["known_findable"]; fo += x["known_found"]
        print(f"{by[s]['rid']}  {by[s]['n_exp']:5d}  {x['known_findable']:8d} {x['known_found']:5d}  {x['leads']:5d}  "
              f"{x['seconds']:7.0f}  {x['status']}")
    print(f"total findable {fa}, found {fo} ({fo / max(fa, 1):.1%} unweighted)")


def leads(name):
    gs = [d for d in os.listdir(os.path.join(RUN, "gens")) if d.startswith(name + "-")]
    assert len(gs) == 1, gs
    env = dict(os.environ, DBT_SITE="nersc")
    subprocess.run([PY, "-I", os.path.join(HERE, "harness", "leads.py"), "s38", gs[0], "--max", "2000"], env=env, check=True)


if __name__ == "__main__":
    {"score": score, "leads": leads}[sys.argv[1]](sys.argv[2])
