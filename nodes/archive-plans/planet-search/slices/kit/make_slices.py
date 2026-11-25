"""Write self-contained slice folders, from metadata alone (nothing is downloaded, nothing runs).

A slice is the run unit of the DECam archive search: about 10 windows, the size of the RSI task (groups/README.md,
DECAM_PLAN.md). This script copies everything one slice needs out of its group into slices/<id>/, so that the folder
runs on its own: ./run.sh WORKDIR does download -> run -> reward -> candidates.

  slices/<id>/run.sh            the slice's script
  slices/<id>/README.md         what the slice is and what each step does
  slices/<id>/slice.json        the card: numbers, program, estimated cost
  slices/<id>/windows.json      its windows and their exposures
  slices/<id>/files.json        exposure -> its three archive files (name, md5, size)
  slices/<id>/settings.json     seconds per window, download settings
  slices/<id>/program/          the finder it runs (finder.py, detcls.pt, rerank.py) and PROGRAM.md
  slices/<id>/private/known.json   every known body in its windows, for the reward only (never shown to a worker)
  slices/<id>/lib/              slice_run.py, slice_score.py, common.py, orbits.py (copies of slices/kit/, linker/)

Usage, from planet-search/:  ~/anaconda3/bin/python -I slices/kit/make_slices.py g099-017 g121-009 ...
       [--program group|reference]   group (default): the group's adapted program; reference: r2 gen015 unchanged
       [--force]                     overwrite an existing slice folder (its results/ are kept)
       --test                        write slices/kit/test/tslice from groups/kit/test/gtest (3 CCDs, runs on the Mac)
"""
import argparse, hashlib, json, os, shutil, sys, time

KIT = os.path.dirname(os.path.abspath(__file__))
SLICES = os.path.dirname(KIT)
ROOT = os.path.dirname(SLICES)
GROUPS = os.path.join(ROOT, "groups")
REFERENCE = "r2 record-0.3985-gen015 (rsi/runs/r2/programs.git), the reference solver"
A100_PER_WINDOW, A100_PER_EXP = 0.2276, 0.0209        # as groups/kit/make_groups.py: <= 20 exposures, above


def md5(f):
    return hashlib.md5(open(f, "rb").read()).hexdigest()


def rng(v, fmt="{:g}"):
    v = sorted(v)
    return fmt.format(v[0]) if v[0] == v[-1] else fmt.format(v[0]) + " to " + fmt.format(v[-1])


def build(sid, gdir, windows, known_path, prog_dir, prog_name, out, force, group_note=""):
    card_g = json.load(open(os.path.join(gdir, "group.json")))
    F = json.load(open(os.path.join(gdir, "files.json")))
    S = json.load(open(os.path.join(gdir, "settings.json")))
    if os.path.exists(out):
        if not force:
            print(f"{sid}: {out} exists (use --force)"); return False
        for x in os.listdir(out):
            if x != "results":
                p = os.path.join(out, x)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    for d in ("program", "private", "lib"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    # windows and files
    W = [dict(w) for w in windows]
    for w in W:
        w.pop("role", None)
    files = {e["stamp"]: F[e["stamp"]] for w in W for e in w["exposures"] if e["stamp"] in F}
    for w in W:
        w["exposures"] = [e for e in w["exposures"] if e["stamp"] in files]
        w["n_exp"] = len(w["exposures"])
        w["n_nights"] = len({int((e["mjd_start"] - 0.5) // 1) for e in w["exposures"]})
    json.dump(dict(id=sid, windows=W), open(os.path.join(out, "windows.json"), "w"), indent=0)
    json.dump(files, open(os.path.join(out, "files.json"), "w"), indent=0)
    nexp = sum(w["n_exp"] for w in W)
    gb = sum(f[p][2] for f in files.values() for p in "iwd") / 1e9
    budget = S.get("budget_s", 900)
    settings = dict(budget_s=budget, fetch_ahead_windows=S.get("fetch_ahead_windows", 1), fetch_streams=S.get("fetch_streams", 16),
                    note="budget_s: seconds the finder gets per window (a number, or a list of [max_exposures, seconds] pairs)")
    json.dump(settings, open(os.path.join(out, "settings.json"), "w"), indent=1)

    # program: only what the finder runs
    for f in sorted(os.listdir(prog_dir)):
        p = os.path.join(prog_dir, f)
        if os.path.isfile(p) and f != "NOTES.md" and not f.startswith("."):
            shutil.copy(p, os.path.join(out, "program", f))
    fm = md5(os.path.join(out, "program", "finder.py"))
    ref_md5 = md5(os.path.join(GROUPS, "kit", "reference", "finder.py"))
    with open(os.path.join(out, "program", "PROGRAM.md"), "w") as fh:
        fh.write(f"# The program slice {sid} runs\n\n{prog_name}.\n\n"
                 f"- base: {REFERENCE}\n- finder.py md5 `{fm}`"
                 + (" (identical to the reference)" if fm == ref_md5 else f" (the reference's is `{ref_md5}`; the differences and their reasons are in GROUP.md)")
                 + "\n- contract: `python -I finder.py --scenario SCEN_DIR --out OUT.json --budget-s SECONDS`\n"
                 "- needs: numpy, astropy, torch (CUDA when a GPU is there); no network, no other files\n")

    # known bodies of these windows
    kn = dict(tier1=0, tier2=0, tier3=0, too_faint=0, uncertain_orbit=0)
    have_known = bool(known_path) and os.path.exists(known_path)
    if have_known:
        K = json.load(open(known_path))
        kw = {str(w["key"]): K["windows"].get(str(w["key"]), []) for w in W}
        for v in kw.values():
            for b in v:
                t = b["tier_pred"]
                kn["uncertain_orbit" if t is None else "too_faint" if t == 0 else f"tier{t}"] += 1
        json.dump(dict(id=sid, made=K.get("made"), source=K.get("source"), max_sigma=K.get("max_sigma"), windows=kw),
                  open(os.path.join(out, "private", "known.json"), "w"))

    # code
    for f in ("slice_run.py", "slice_score.py", "common.py"):
        shutil.copy(os.path.join(KIT, f), os.path.join(out, "lib", f))
    shutil.copy(os.path.join(ROOT, "linker", "orbits.py"), os.path.join(out, "lib", "orbits.py"))

    def bsec(n):
        if isinstance(budget, (int, float)):
            return float(budget)
        return next((float(s) for m, s in budget if n <= m), float(budget[-1][1]))
    gpu_h = sum(bsec(w["n_exp"]) for w in W) / 3600
    card = dict(id=sid, group=card_g["id"], group_name=card_g["group"], family=card_g.get("family"), made=time.strftime("%Y-%m-%d"),
                windows=len(W), exposures=nexp, gb=round(gb, 1), first=min(str(w["first"]) for w in W), last=max(str(w["last"]) for w in W),
                exposures_per_window=[min(w["n_exp"] for w in W), max(w["n_exp"] for w in W)],
                nights_per_window=[min(w["n_nights"] for w in W), max(w["n_nights"] for w in W)],
                bands=" ".join(sorted({e["band"] for w in W for e in w["exposures"]})),
                budget_s=budget, gpu_hours_at_most=round(gpu_h, 2),
                a100_hours_estimate=round(sum(A100_PER_WINDOW if w["n_exp"] <= 20 else A100_PER_EXP * w["n_exp"] for w in W), 2),
                slots=25 * len(W), program=dict(name=prog_name, base=REFERENCE, finder_md5=fm, same_as_reference=fm == ref_md5),
                known_bodies_predicted=kn if have_known else None)
    json.dump(card, open(os.path.join(out, "slice.json"), "w"), indent=1)

    rounds = -(-len(W) // 4)
    minutes = rounds * (bsec(20) + 120) / 60 + 5
    t_nexp, t_nights = rng([w["n_exp"] for w in W]), rng([w["n_nights"] for w in W])
    t_exptime, t_beta = rng([w["exptime"] for w in W]), rng([w["beta"] for w in W], "{:.0f}")
    t_budget = "%g" % budget if isinstance(budget, (int, float)) else str(budget)
    t_b20 = "%g" % bsec(20)
    disk_gb = max(sum(files[e["stamp"]][p][2] for e in w["exposures"] for p in "iwd") for w in W) / 1e9 * 2 * 4
    t_groupmd = ", `GROUP.md` (differences from the reference)" if os.path.exists(os.path.join(out, "program", "GROUP.md")) else ""
    gname = card_g["group"].replace(" | ", ", ")
    title = f"Slice {sid}: {len(W)} windows, {nexp} exposures, {gb:.1f} GB  ({gname})"
    with open(os.path.join(out, "run.sh"), "w") as fh:
        fh.write(f"""#!/bin/bash
# {title}
# One self-contained run:  1 download -> 2 run -> 3 reward -> 4 candidates.   Read README.md first.
#
#   ./run.sh WORKDIR                      the whole slice; needs GPUs and a node that reaches the NOIRLab archive
#   ./run.sh WORKDIR --dry-run            print what would be fetched and run; fetches nothing, runs nothing
#   ./run.sh WORKDIR --local-index FILE   link images already on disk (one path per line) instead of downloading
#   more options: --gpus N  --keep-images  --redo  --windows KEY,KEY  --program DIR  --threads N
#   PYTHON=/path/to/python picks the interpreter (numpy, astropy, torch). A rerun resumes where it stopped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${{PYTHON:-python}}"
[ $# -ge 1 ] || {{ sed -n 2,10p "$0"; exit 2; }}
WORK="$1"; shift
mkdir -p "$WORK"
case " $* " in *" --dry-run "*) exec "$PY" -I "$HERE/lib/slice_run.py" "$HERE" "$WORK" "$@";; esac
exec > >(tee -a "$WORK/slice.log") 2>&1

# steps 1 and 2: window by window, download the images, run the finder, measure, delete the images
"$PY" -I "$HERE/lib/slice_run.py" "$HERE" "$WORK" "$@"

# steps 3 and 4: the reward on the known bodies, then the candidate list and the leads
"$PY" -I "$HERE/lib/slice_score.py" "$HERE" "$WORK"

echo "$(date '+%F %T') slice {sid} done on $(hostname)" | tee "$WORK/DONE"
""")
    os.chmod(os.path.join(out, "run.sh"), 0o755)

    kline = ("no list of known bodies was made for this group: the slice gives candidates but no reward" if not have_known else
             f"{kn['tier1'] + kn['tier2'] + kn['tier3']} predicted findable (visible in single exposures {kn['tier1']}, in a night's stack "
             f"{kn['tier2']}, only in the stack of all nights {kn['tier3']}), {kn['too_faint']} too faint, {kn['uncertain_orbit']} with an "
             f"orbit too uncertain to score. The run measures them on the images and the reward uses the measured classes")
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(f"""# {title}

Made {card['made']} by `slices/kit/make_slices.py` from group {card_g['id']}, from archive metadata alone. This folder
holds everything the slice needs. One command runs it from start to finish:

```
./run.sh WORKDIR
```

## The slice

| | |
|---|---|
| survey class (group {card_g['id']}) | {gname}{group_note} |
| windows (one 1-degree sky cell in one 60-day bin) | {len(W)} |
| exposures | {nexp}; {t_nexp} per window on {t_nights} nights, band {card['bands']}, {t_exptime} s |
| dates | {card['first']} to {card['last']} |
| ecliptic latitude | {t_beta} deg |
| download | {gb:.1f} GB ({3 * nexp} files: image, weight map and mask per exposure) |
| finder time | {t_budget} s per window at most on one GPU: {gpu_h:.2f} GPU-hours at most for the slice |
| on one Perlmutter GPU node (4 A100) | {rounds} rounds of 4 windows: about {minutes:.0f} minutes, {minutes / 60:.1f} node-hours |
| program | {prog_name} (`program/PROGRAM.md`) |
| known bodies in these windows | {kline} |

## What `run.sh` does

1. **Download.** For each window it fetches the exposures' files from the NOIRLab archive
   (`files.json`: name, md5, size), checks every md5 while the file streams in, and retries a failed file up to 4
   times. It stays one window ahead of each GPU, so at most about {disk_gb:.0f} GB
   sit on disk at once on a 4-GPU node. With `--local-index FILE` a file found on disk is linked instead.
   It then writes the window's scenario folder in the RSI task's format (`exposures.csv` with times and the
   observatory's position, `meta.json`, links to the files). Every exposure is given to the finder.
2. **Run.** `program/finder.py --scenario S --out OUT.json --budget-s {t_b20}` runs on one GPU per window, in a
   fresh copy with a clean environment and no network (when the node allows it), and is killed 30 s after its
   budget. It writes up to 200 candidates: an orbit (position and velocity at an epoch), a confidence p and a
   magnitude each. While the images are still on disk the script measures, by forced photometry, the known bodies
   at their predicted places and the window's 50 best candidates along their tracks, plus the same sky spots on
   other nights. Then it deletes the window's images.
3. **Reward.** The RSI task grader's rules: over the slice the {25 * len(W)} candidates with the highest p are kept
   (25 per window on average). A kept candidate is a hit when its orbit lies within 2 arcsec of a known body (or
   the body's 3-sigma) on at least 3 nights. A body weighs 1 if single exposures show it, 2 if only a night's
   stack does, 4 if only the stack of all nights does. reward = weight found / weight of all findable bodies.
4. **Candidates.** Every kept candidate is labelled: a known body, a lead, or unmatched. A lead matches no known
   body, has p >= 0.5, no known body within 10 arcsec, and passes the screen measured in step 2: at least 3
   nights with stack S/N >= 4.5 on its track, the same sky spot on other nights below S/N 3 (so not a star or
   galaxy), and S/N >= 5 without its brightest exposure. Each lead is written in the input format of the novelty
   check; from there the existing pipeline takes over (catalogues and joint orbit fit, then confirmation in
   images the program never used).

## What comes out, in WORKDIR

| file | what |
|---|---|
| `reward.json` | the reward, recall, bodies found by class, window status, GPU time; numbers only |
| `candidates.json` | every kept candidate: window, rank, p, distance, orbit, label, screen numbers |
| `leads/<id>.json` | one file per lead, ready for `rsi/candidates/novelty_queue/inbox/` |
| `private/reward_bodies.json` | every known body: measured class, found or not, rank |
| `out/<window>/` | per window: `candidates.json` (the finder's own output), `finder.log`, `info.json` (status, seconds, download speed), `exposures.csv`, `known_phot.json`, `cand_phot.json` |
| `run.json`, `slice.log`, `DONE` | totals of steps 1-2, the log, the end marker |

## In this folder

| path | what |
|---|---|
| `run.sh` | the script above |
| `slice.json` | the card: these numbers, the program's checksum |
| `windows.json` | the {len(W)} windows and their exposures (time, band, pointing) |
| `files.json` | exposure -> its three archive files (name, md5, size) |
| `settings.json` | seconds per window, download streams |
| `program/` | the finder: `finder.py`, `detcls.pt` (its small detection network), `rerank.py`; `PROGRAM.md`{t_groupmd} |
| `private/known.json` | the known bodies of these windows with predicted positions. Used only by steps 2 (photometry) and 3. The finder is never given it. Not for workers who adapt programs |
| `lib/` | `slice_run.py` (steps 1-2), `slice_score.py` (steps 3-4), `common.py`, `orbits.py` |

Needs: Python with numpy, astropy and torch; GPUs (it also runs on a CPU, slowly); a node that reaches
astroarchive.noirlab.edu unless `--local-index` covers every file. `./run.sh WORKDIR --dry-run` checks the folder
and the disk without fetching or running anything.
""")
    print(f"{sid}: {len(W)} windows, {nexp} exposures, {gb:.1f} GB, {gpu_h:.2f} GPU-h at most, program {prog_name}, "
          f"known findable {kn['tier1'] + kn['tier2'] + kn['tier3'] if have_known else 'no list'} -> {out}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*"); ap.add_argument("--program", default="group", choices=["group", "reference"])
    ap.add_argument("--force", action="store_true"); ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    ref = os.path.join(GROUPS, "kit", "reference")
    if a.test:
        g = os.path.join(GROUPS, "kit", "test", "gtest")
        W = json.load(open(os.path.join(g, "windows.json")))["windows"]
        build("tslice", g, W, os.path.join(g, "known_test.json"), ref, "the reference, unchanged", os.path.join(KIT, "test", "tslice"), True)
        shutil.copy(os.path.join(g, "local_index.txt"), os.path.join(KIT, "test", "local_index.txt"))
        return
    for sid in a.ids:
        gid = sid.split("-")[0]
        g = os.path.join(GROUPS, gid)
        S = {s["id"]: s for s in json.load(open(os.path.join(g, "slices.json")))["slices"]}
        if sid not in S:
            print(f"{sid}: not in {gid}/slices.json"); continue
        byk = {w["key"]: w for w in json.load(open(os.path.join(g, "windows.json")))["windows"]}
        W = [byk[k] for k in S[sid]["windows"]]
        if a.program == "group":
            prog, name = os.path.join(g, "program"), f"group {gid}'s adapted copy of the reference"
        else:
            prog, name = ref, "the reference, unchanged"
        build(sid, g, W, os.path.join(GROUPS, "_private", gid, "known.json"), prog, name, os.path.join(SLICES, sid), a.force)


if __name__ == "__main__":
    main()
