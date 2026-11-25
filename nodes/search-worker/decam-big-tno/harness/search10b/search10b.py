#!/usr/bin/env python3
"""search10b: step B of the second 5 % (planet-search/archive10). User, 2026-11-16, in reply to "Step B (the other 20
regions, 24,653 exposures): not started, and it waits for your go": "Let's go".
The hill climb's best program (gen034) over the other 20 archive10 regions (R029 ... R077 of archive10's own numbering).
A copy of harness/search10a/search10a.py with its own folders, so nothing of search10a, search38 or their runs is
touched: NERSC $D/search10b, known bodies planet-search/archive10/stepB/known, labels private/truth10b, outputs
runs/search10b, scores runs/s10b (score10b.py). Detection lists: archive10/sb_0..7/det for the 20,096 exposures detected
in step B, and the first 5 % search's lists (archive5/*/det) for the 3,983 exposures those regions share with it.
Region R061 (9,591 exposures, the DES supernova fields X1, X2, X3) is too large for one scenario, so it is built as
several parts (rids R061a ...), each a list of stems in regions_exposures.json; regions10b.json names every part.
It reuses search38's run38.sbatch and node38.sh unchanged. The commands are search38's:

search38: the best hill-climb program run over the 5 % search's other 38 regions on NERSC, for new candidates.

2026-11-13. Steps (each its own command, run from the Mac through the user's NERSC master):

  stage    copy build_scen.py, the job scripts, regions_exposures.json, the known-body files and list.txt to NERSC
  build    submit build10b.sbatch (one CPU node, $NERSC_CPU_ACCOUNT): one scenario per region from the detection lists on NERSC
  truth    copy the known-body labels (search10b/truth/*.npz) to private/truth10b/ here, then delete on NERSC each one
           that arrived whole (same size)
  launch NAME LIST [RUN B_NORMAL B_BIG]   (defaults r1 3600 7200) copy the best graded program and submit run38.sbatch for the built regions of LIST ("RID SID NEXP"
           lines): regions <= 1,000 exposures at 3,600 s, larger ones at 7,200 s, one round per node (4 regions per
           node), $NERSC_GPU_ACCOUNT regular; roots search10b/NAME_normal, NAME_big
  status   squeue + how many regions have output
  fetch    copy outputs and timings back (runs/search10b/<root>/<gen>/out/<sid>/candidates.json)
Scores and leads per region: harness/grade.py OUT private/truth10b GRADE --sids SID, then score10b.py.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "harness"))
import nersc_grade as NG                                   # noqa: E402  (ssh/rsync through the user's master)

D = "$NERSC_SCRATCH/dbt"; S = f"{D}/search10b"
A5L = os.path.join(HERE, "..", "..", "planet-search", "archive10", "stepB")
REG = json.load(open(os.path.join(HERE, "harness", "search10b", "regions10b.json")))
BIG = 1000
LOCAL = os.path.join(HERE, "runs", "search10b")


def stage():
    NG.ssh(f"mkdir -p {S}/in/known {S}/log {S}/truth")
    for f in ("bin/build_scen.py", "harness/search10b/build10b.sbatch", "harness/search38/run38.sbatch"):
        NG.rsync(os.path.join(HERE, f), f"{NG.HOST}:{S}/")
    NG.rsync(os.path.join(A5L, "regions_exposures.json"), f"{NG.HOST}:{S}/in/")
    NG.rsync(os.path.join(A5L, "known") + "/", f"{NG.HOST}:{S}/in/known/")
    lst = "\n".join(f"{r['rid']} {r['sid']} {r['n_exp']}" for r in REG) + "\n"
    os.makedirs(LOCAL, exist_ok=True); p = os.path.join(LOCAL, "list10b.txt"); open(p, "w").write(lst)
    NG.rsync(p, f"{NG.HOST}:{S}/list.txt")
    print(NG.ssh(f"ls {S} {S}/in/known | head -60; wc -l {S}/list.txt")[1])


def build():
    print(NG.ssh(f"cd {S} && sbatch build10b.sbatch")[1])


def truth():
    dst = os.path.join(HERE, "private", "truth10b"); os.makedirs(dst, exist_ok=True)
    NG.rsync(f"{NG.HOST}:{S}/truth/", dst + "/")
    rem = dict(l.split()[::-1] for l in NG.ssh(f"cd {S}/truth && stat -c '%s %n' *.npz")[1].split("\n") if l.strip())
    ok = [f for f, n in rem.items() if os.path.exists(os.path.join(dst, f)) and os.path.getsize(os.path.join(dst, f)) == int(n)]
    print(len(ok), "of", len(rem), "truth files here whole;", len([f for f in os.listdir(dst) if f.endswith(".npz")]), "in", dst)
    if ok:
        print(NG.ssh(f"cd {S}/truth && rm -f " + " ".join(ok) + " && echo deleted " + str(len(ok)) + " on NERSC; ls | wc -l")[1])


def launch(name, listfile, run="r1", b_normal="3600", b_big="7200"):
    # run: whose best program (runs/<run>/state.json); budgets in seconds for regions <= / > 1,000 exposures.
    # 2026-11-15: launch r2b LIST r2 1800 7200
    st = json.load(open(os.path.join(HERE, "runs", run, "state.json")))
    best = st["best"]; sub = best["submission"]; gen = best["gen"]
    rows = [l.split() for l in open(listfile) if l.strip()]
    built = set(NG.ssh(f"for s in {' '.join(r[1] for r in rows)}; do [ -f {D}/scen/$s/exposures.json ] && echo $s; done")[1].split())
    miss = [r[0] for r in rows if r[1] not in built]
    print("program:", gen, best["reward"], "| built", len(built), "of", len(rows), "| not built:", " ".join(miss) or "none")
    for f in ("harness/search38/node38.sh", "harness/search38/run38.sbatch"):
        NG.rsync(os.path.join(HERE, f), f"{NG.HOST}:{S}/")
    for tag, budget in (("normal", int(b_normal)), ("big", int(b_big))):
        sel = [r for r in rows if r[1] in built and (int(r[2]) <= BIG) == (tag == "normal")]
        if not sel:
            continue
        root = f"{S}/{name}_{tag}"; nodes = (len(sel) + 3) // 4
        wall_s = budget + 30 + 900; wall = f"{wall_s // 3600:02d}:{wall_s % 3600 // 60 + 1:02d}:00"
        NG.ssh(f"mkdir -p {root}/{gen}/sub")
        NG.rsync(sub + "/", f"{NG.HOST}:{root}/{gen}/sub/")
        os.makedirs(LOCAL, exist_ok=True); p = os.path.join(LOCAL, f"list_{name}_{tag}.txt")
        open(p, "w").write("".join(f"{gen} {r[1]}\n" for r in sorted(sel, key=lambda r: -int(r[2]))))
        NG.rsync(p, f"{NG.HOST}:{root}/list.txt")
        print(f"{tag}: {len(sel)} regions, {nodes} nodes, budget {budget} s, wall {wall}")
        for k in range(nodes):
            print(" ", NG.ssh(f"cd {S} && sbatch --export=ALL,ROOT={root},BUDGET={budget},WALL_S={wall_s} -t {wall} "
                              f"-J dbt-{name}{tag[0]}{k} run38.sbatch")[1].strip())


def status():
    print(NG.ssh(f"squeue -u $NERSC_USER -h -o '%i %j %T %M %S' | grep dbt-; cd {S} && for t in *_normal *_big; do "
                 f"[ -d $t ] && echo $t $(ls $t/*/timing 2>/dev/null | wc -l) of $(wc -l < $t/list.txt) done; done")[1])


def fetch():
    for t in NG.ssh(f"cd {S} && ls -d *_normal *_big 2>/dev/null")[1].split():
        NG.rsync(f"{NG.HOST}:{S}/{t}/", os.path.join(LOCAL, t) + "/")
    print("fetched to", LOCAL)


if __name__ == "__main__":
    {"stage": stage, "build": build, "truth": truth, "launch": launch, "status": status, "fetch": fetch}[sys.argv[1]](*sys.argv[2:])
