#!/usr/bin/env python3
"""searchldt5: the fourth 5 % of DECam's sky, chosen for what LDT can observe now (planet-search/ldt5; user, 2026-11-18:
"lets try to run another 5% region on nersc? Focus on the region where LDT can observe recently", "Let's use the gpu for
detection. If cpu not available then", "Prioritize the download the required data for this region").
A copy of search10b.py with its own folders: NERSC $D/searchldt5, known bodies planet-search/ldt5/known, labels
private/truthldt5, outputs runs/searchldt5. Because NERSC's CPU and regular GPU nodes are all waiting for a rolling
reboot, the build and the search run as shared GPU jobs (buildldt5.sbatch, runldt5.sbatch).

  stage              copy build_scen.py, the two job scripts, regions_exposures.json and the known-body files to NERSC
  build RIDS         submit buildldt5.sbatch for the regions RIDS (comma list, or "all")
  truth              copy the known-body labels to private/truthldt5 here, then delete each one that arrived whole on NERSC
  launch NAME [RUN B_NORMAL B_BIG JOBS]   (defaults r2 1800 7200 6) the best program of RUN; regions <= 1,000 exposures
                     at B_NORMAL in JOBS one-GPU jobs, larger ones at B_BIG in two-GPU jobs with one lane each; only built
                     regions run (a region whose scenario is missing is skipped, so a later launch picks it up)
  status | fetch     as search10b
Scores and leads: harness/grade.py OUT private/truthldt5 GRADE --sids SID.
"""
import json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "harness"))
import nersc_grade as NG                                   # noqa: E402

D = "$NERSC_SCRATCH/dbt"; S = f"{D}/searchldt5"
L5 = os.path.join(HERE, "..", "..", "planet-search", "ldt5")
REG = json.load(open(os.path.join(HERE, "harness", "searchldt5", "regionsldt5.json")))
BIG = 1000
LOCAL = os.path.join(HERE, "runs", "searchldt5")


def stage():
    NG.ssh(f"mkdir -p {S}/in/known {S}/log {S}/truth")
    for f in ("bin/build_scen.py", "harness/searchldt5/buildldt5.sbatch", "harness/searchldt5/runldt5.sbatch"):
        NG.rsync(os.path.join(HERE, f), f"{NG.HOST}:{S}/")
    NG.rsync(os.path.join(L5, "search_in", "regions_exposures.json"), f"{NG.HOST}:{S}/in/")   # regions > 2,000 exposures in parts (ldt5/parts.json)
    NG.rsync(os.path.join(L5, "search_in", "known") + "/", f"{NG.HOST}:{S}/in/known/")
    lst = "\n".join(f"{r['rid']} {r['sid']} {r['n_exp']}" for r in REG) + "\n"
    os.makedirs(LOCAL, exist_ok=True); p = os.path.join(LOCAL, "list.txt"); open(p, "w").write(lst)
    NG.rsync(p, f"{NG.HOST}:{S}/list.txt")
    print(NG.ssh(f"ls {S}; ls {S}/in/known | wc -l; wc -l {S}/list.txt")[1])


def build(rids):
    want = None if rids == "all" else set(rids.split(","))
    sel = [r for r in REG if want is None or r["rid"] in want]
    name = f"build_{len(sel)}_{sel[0]['rid']}.txt"
    os.makedirs(LOCAL, exist_ok=True); p = os.path.join(LOCAL, name)
    open(p, "w").write("".join(f"{r['rid']} {r['sid']} {r['n_exp']}\n" for r in sel))
    NG.rsync(p, f"{NG.HOST}:{S}/{name}")
    print(NG.ssh(f"cd {S} && sbatch --export=ALL,LIST={S}/{name} buildldt5.sbatch")[1])


def truth():
    dst = os.path.join(HERE, "private", "truthldt5"); os.makedirs(dst, exist_ok=True)
    NG.rsync(f"{NG.HOST}:{S}/truth/", dst + "/")
    rem = dict(l.split()[::-1] for l in NG.ssh(f"cd {S}/truth && stat -c '%s %n' *.npz")[1].split("\n") if l.strip())
    ok = [f for f, n in rem.items() if os.path.exists(os.path.join(dst, f)) and os.path.getsize(os.path.join(dst, f)) == int(n)]
    print(len(ok), "of", len(rem), "truth files here whole;", len([f for f in os.listdir(dst) if f.endswith(".npz")]), "in", dst)
    if ok:
        print(NG.ssh(f"cd {S}/truth && rm -f " + " ".join(ok) + " && echo deleted " + str(len(ok)) + " on NERSC; ls | wc -l")[1])


def launch(name, run="r2", b_normal="1800", b_big="7200", jobs="6", prebuilt="no"):
    # prebuilt=yes: submit for every region now; the jobs wait for $S/BUILT (written by buildldt5.sbatch) before searching
    st = json.load(open(os.path.join(HERE, "runs", run, "state.json")))
    best = st["best"]; sub = best["submission"]; gen = best["gen"]
    built = set(NG.ssh(f"for s in {' '.join(r['sid'] for r in REG)}; do [ -f {D}/scen/$s/exposures.json ] && echo $s; done")[1].split())
    print("program:", gen, best["reward"], "| built", len(built), "of", len(REG))
    if prebuilt == "yes":
        built = set(r["sid"] for r in REG)
    NG.rsync(os.path.join(HERE, "harness/searchldt5/runldt5.sbatch"), f"{NG.HOST}:{S}/")
    for tag, budget in (("normal", int(b_normal)), ("big", int(b_big))):
        sel = [r for r in REG if r["sid"] in built and (r["n_exp"] <= BIG) == (tag == "normal")]
        if not sel:
            continue
        root = f"{S}/{name}_{tag}"
        NG.ssh(f"mkdir -p {root}/{gen}/sub")
        NG.rsync(sub + "/", f"{NG.HOST}:{root}/{gen}/sub/")
        os.makedirs(LOCAL, exist_ok=True); p = os.path.join(LOCAL, f"list_{name}_{tag}.txt")
        # best LDT hours first, so the regions LDT sees longest are searched first
        open(p, "w").write("".join(f"{gen} {r['sid']}\n" for r in sorted(sel, key=lambda r: -r["ldt_h"])))
        NG.rsync(p, f"{NG.HOST}:{root}/list.txt")
        n = min(int(jobs), len(sel)) if tag == "normal" else len(sel)
        per = -(-len(sel) // n)
        wall_s = per * (budget + 120) + 900; wall = f"{wall_s // 3600:02d}:{wall_s % 3600 // 60 + 1:02d}:00"
        g = "--gpus 1 -c 32" if tag == "normal" else "--gpus 2 -c 64"
        lanes = "" if tag == "normal" else ",LANES=1"
        print(f"{tag}: {len(sel)} regions, {n} jobs ({g}), budget {budget} s, wall {wall}")
        for k in range(n):
            print(" ", NG.ssh(f"cd {S} && sbatch {g} --export=ALL,ROOT={root},BUDGET={budget},WALL_S={wall_s}{lanes} -t {wall} "
                              f"-J dbt-l5{tag[0]}{k} runldt5.sbatch")[1].strip())


def status():
    print(NG.ssh(f"squeue -u $NERSC_USER -h -o '%i %j %T %M %S' | grep -E 'dbt-l5|dbt-build|l5d'; cd {S} && for t in *_normal *_big; do "
                 f"[ -d $t ] && echo $t $(ls $t/*/timing 2>/dev/null | wc -l) of $(wc -l < $t/list.txt) done; done")[1])


def fetch():
    for t in NG.ssh(f"cd {S} && ls -d *_normal *_big 2>/dev/null")[1].split():
        NG.rsync(f"{NG.HOST}:{S}/{t}/", os.path.join(LOCAL, t) + "/")
    print("fetched to", LOCAL)


if __name__ == "__main__":
    {"stage": stage, "build": build, "truth": truth, "launch": launch, "status": status, "fetch": fetch}[sys.argv[1]](*sys.argv[2:])
