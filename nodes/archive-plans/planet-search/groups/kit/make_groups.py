"""Write one folder per group from the exposure metadata alone (no image is read or downloaded).

  groups/<id>/group.json     the card: family, parent, part, statistics, cost, roles, what the reference was tested on
  groups/<id>/windows.json   every window: key, sid, role (practice / validation / production), its features, and its
                             exposures (stamp, exposure number, MJD of the start, exposure time, band, RA, Dec, proposal)
  groups/<id>/files.json     stamp -> the three InstCal files of the exposure (image i, weight w, mask d: name, md5,
                             size), when the archive file list is there (--instcal); an exposure without a complete
                             set is left out here and skipped by the group job
  groups/<id>/program/       the starting program (kit/reference unless --start), copied once; the worker tailors it
  groups/<id>/settings.json  run settings the worker may change (time cap per window, ...), written once
  groups/<id>/run.sh         the group's runnable script (calls kit/run_group.py), rewritten every time
  groups/<id>/slices.json    the group's runnable windows cut into slices of about the RSI task's size (--slice-cost,
                             2.5 A100-hours = 10 windows of <= 20 exposures; a larger window is a slice of its own),
                             in time order then right ascension; each window in windows.json carries its slice id.
                             A slice is the unit a job runs (run_group.py --slices); the group only holds the program,
                             the settings and the known bodies its slices share

Roles come from a fixed hash of the window key: the first 4 are practice, the next 10 validation (6 in dense families and
in later parts of a parent, 4 in later dense parts), the rest production. With the file list, a window that keeps fewer
than 3 nights with files gets the role "no data" and takes no part.

Usage: ~/anaconda3/bin/python -I groups/kit/make_groups.py g099 g121 ...   (or --all)
"""
import argparse, hashlib, json, os, re, shutil, glob


def dump(obj, path, **kw):
    """Atomic, and only when the content changes: workers may be reading these files."""
    txt = json.dumps(obj, **kw)
    if os.path.exists(path) and open(path).read() == txt:
        return
    open(path + ".tmp", "w").write(txt)
    os.replace(path + ".tmp", path)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                               # planet-search/groups
FOOT = os.path.join(os.path.dirname(ROOT), "linker", "footprint")
EXP_FIELDS = ["stamp", "expnum", "mjd_start", "exptime", "band", "ra", "dec", "proposal"]


def frac(key):
    return int(hashlib.sha1(str(key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def sid_of(key):
    return "s" + hashlib.sha1(str(key).encode()).hexdigest()[:8]


def load_instcal(d, stamps):
    """stamp -> {"i": [name, md5, size], "w": ..., "d": ...} for the newest complete Community Pipeline version."""
    vers = {}
    for f in sorted(glob.glob(os.path.join(d, "instcal_*.json"))):
        for name, md5, size, rel, *_ in json.load(open(f))["rows"]:
            m = re.match(r"(c4d_\d{6}_\d{6})_oo([iwd])_[^_]+_(.+)\.fits\.fz$", name)
            if m and m.group(1) in stamps:
                vers.setdefault(m.group(1), {}).setdefault(m.group(3), {})[m.group(2)] = [name, md5, size, rel]
    out = {}
    for st, vs in vers.items():
        full = {v: x for v, x in vs.items() if set(x) == {"i", "w", "d"}}
        if not full:
            continue
        cp = sorted((v for v in full if re.fullmatch(r"v\d+", v)), key=lambda v: int(v[1:]))
        out[st] = full[cp[-1] if cp else sorted(full)[0]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--instcal", default=os.path.join(os.path.dirname(ROOT), "linker", "data_real", "noirlab_decam_instcal"))
    ap.add_argument("--start", default=os.path.join(HERE, "reference"), help="starting program copied into a new group")
    ap.add_argument("--slice-cost", type=float, default=2.5, help="A100-hours per slice (the RSI task: 10 windows x 900 s)")
    ap.add_argument("--count", action="store_true", help="print the numbers only, write nothing")
    a = ap.parse_args()
    G = json.load(open(os.path.join(FOOT, "groups.json")))
    W = json.load(open(os.path.join(FOOT, "groups_windows.json")))
    E = json.load(open(os.path.join(FOOT, "groups_exposures.json")))["windows"]
    cards = {g["id"]: dict(g, family=f["family"]) for f in G["families"] for g in f["groups"]}
    ids = sorted(cards) if a.all else a.ids
    byg = {}
    for w in W:
        if w["id"] in ids:
            byg.setdefault(w["id"], []).append(w)
    stamps = {e[0] for i in ids for w in byg[i] for e in E[str(w["key"])]}
    inst = load_instcal(a.instcal, stamps) if os.path.isdir(a.instcal) else None
    tot_s = []
    for i in ids:
        c, ws = cards[i], sorted(byg[i], key=lambda w: frac(w["key"]))
        dense = c["family"].startswith(("dense", "deep"))
        lead = c["part"] == 1
        nval = (6 if dense else 10) if lead else (4 if dense else 6)
        nprac = min(4, max(0, len(ws) - nval))
        out, lost, files = [], 0, {}
        if inst is not None:                      # windows that cannot run go last and get no role
            runs = {w["key"]: len({int((e[2] - 0.5) // 1) for e in E[str(w["key"])] if e[0] in inst}) >= 3 for w in ws}
            ws = [w for w in ws if runs[w["key"]]] + [w for w in ws if not runs[w["key"]]]
            nrun = sum(runs.values())
        else:
            nrun = len(ws)
        nprac = min(4, max(0, nrun - nval))
        for k, w in enumerate(ws):
            role = "practice" if k < nprac else "validation" if k < nprac + nval else "production" if k < nrun else "no data"
            exps = []
            for e in E[str(w["key"])]:
                x = dict(zip(EXP_FIELDS, e))
                if inst is not None:
                    if e[0] in inst:
                        files[e[0]] = {p: inst[e[0]][p][:3] for p in "iwd"}
                    else:
                        lost += 1
                exps.append(x)
            out.append(dict(key=w["key"], sid=sid_of(w["key"]), role=role,
                            **{f: w[f] for f in ("n_exp", "n_nights", "span", "bands", "exptime", "rep_frac", "gap_min", "depth1",
                                                 "depth", "beta", "b", "ra", "dec", "first", "last", "prop", "prop_frac")},
                            exposures=exps))
        # slices: runnable windows in time order then right ascension, packed up to --slice-cost
        slices, cur, cc = [], [], 0.0
        def wcost(w):
            n = sum(1 for e in w["exposures"] if inst is None or e["stamp"] in inst)
            return (0.2276 if n <= 20 else 0.0209 * n), n
        for w in sorted((w for w in out if w["role"] != "no data"), key=lambda w: (str(w["first"]), w["ra"])):
            c1, n1 = wcost(w)
            if cur and cc + c1 > a.slice_cost + 1e-9:
                slices.append(cur); cur, cc = [], 0.0
            cur.append((w, c1, n1)); cc += c1
        if cur:
            slices.append(cur)
        srows = []
        for k, sl in enumerate(slices):
            sidk = f"{i}-{k + 1:03d}"
            gb = sum(f[p][2] for w, _, _ in sl for e in w["exposures"] if e["stamp"] in files for f in [files[e["stamp"]]]
                     for p in "iwd") / 1e9 if inst is not None else None
            for w, _, _ in sl:
                w["slice"] = sidk
            srows.append(dict(id=sidk, windows=[w["key"] for w, _, _ in sl], n_windows=len(sl),
                              n_exposures=sum(n for _, _, n in sl), gb=round(gb, 1) if gb is not None else None,
                              a100_h=round(sum(c1 for _, c1, _ in sl), 2), first=str(sl[0][0]["first"]), last=str(sl[-1][0]["last"])))
        for w in out:
            w.setdefault("slice", None)
        tot_s.append((i, c["family"], srows))
        if a.count:
            print(f"{i} {c['group']}: {nrun} runnable windows in {len(srows)} slices")
            continue
        d = os.path.join(ROOT, i)
        os.makedirs(d, exist_ok=True)
        card = dict(id=i, group=c["group"], family=c["family"], parent=c["parent"], part=c["part"], parts=c["parts"],
                    lead=lead, asof=G.get("asof"), roles=dict(practice=nprac, validation=min(nval, nrun - nprac),
                                                               production=max(0, nrun - nprac - nval), no_data=len(ws) - nrun),
                    slices=len(srows), slice_cost_a100_h=a.slice_cost,
                    file_lists=inst is not None, exposures_without_instcal=lost if inst is not None else None,
                    gb=round(sum(f[p][2] for f in files.values() for p in "iwd") / 1e9, 1) if inst is not None else None,
                    stats={k: v for k, v in c.items() if k not in ("id", "group", "family", "parent", "part", "parts", "sample")},
                    stats_note="n_exp, n_nights, span (days), exptime (s), depth1 / depth (V a TNO must beat at 5 sigma in one "
                               "exposure / in the stack), abs_beta (deg), gap_min (longest same-night gap, minutes) are 10th / "
                               "50th / 90th percentiles over the windows; cost in A100-hours at the reference's 900 s per window",
                    reference_tested_on="10 windows of 5-20 exposures on 4-7 nights over 3-45 days, within 30 deg of the ecliptic, "
                                        ">= 40 deg from the Galactic plane, 900 s per window on one A100")
        dump(card, os.path.join(d, "group.json"), indent=1)
        dump(dict(id=i, exposure_fields=EXP_FIELDS, windows=out), os.path.join(d, "windows.json"))
        dump(dict(id=i, slice_cost_a100_h=a.slice_cost, slices=srows), os.path.join(d, "slices.json"), indent=1)
        if inst is not None:
            dump(files, os.path.join(d, "files.json"))
        if not os.path.exists(os.path.join(d, "program")):
            shutil.copytree(a.start, os.path.join(d, "program"))
        sp = os.path.join(d, "settings.json")
        if not os.path.exists(sp):
            json.dump(dict(budget_s=900, fetch_ahead_windows=1, fetch_streams=16,
                           note="budget_s: seconds the finder gets per window (a number, or a list of [max_exposures, seconds] "
                                "pairs in rising order)"), open(sp, "w"), indent=1)
        open(os.path.join(d, "run.sh"), "w").write(
            f"#!/bin/bash\n# Group {i}: {c['group']}\n# Everything this group needs: fetch its images, run its program, "
            f"measure the known bodies, keep the results, delete the images.\n# Usage: ./run.sh WORKDIR [practice|validation|"
            f"production|all] [--dry-run] [more run_group.py options]\nset -euo pipefail\nHERE=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
            f"exec \"${{PYTHON:-python}}\" -I \"$HERE/../kit/run_group.py\" \"$HERE\" \"$@\"\n")
        os.chmod(os.path.join(d, "run.sh"), 0o755)
        print(f"{i} {c['group']}: {len(ws)} windows ({card['roles']}), {len(srows)} slices, "
              f"{sum(len(w['exposures']) for w in out)} exposures"
              + (f", {lost} without InstCal files, {card['gb']} GB" if inst is not None else ", no file lists yet"))
    import statistics
    R = [r for _, _, rows in tot_s for r in rows]
    if R:
        fam = {}
        for _, f, rows in tot_s:
            x = fam.setdefault(f, [0, 0, 0.0]); x[0] += len(rows); x[1] += sum(r["n_windows"] for r in rows); x[2] += sum(r["a100_h"] for r in rows)
        print(f"{len(R)} slices in {len(tot_s)} groups: {sum(r['n_windows'] for r in R)} windows, {sum(r['n_exposures'] for r in R)} "
              f"exposures, {sum(r['a100_h'] for r in R):.0f} A100-h; per slice median {statistics.median(r['n_windows'] for r in R):.0f} "
              f"windows, {statistics.median(r['n_exposures'] for r in R):.0f} exposures, {statistics.median(r['a100_h'] for r in R):.2f} "
              f"A100-h" + (f", {statistics.median(r['gb'] for r in R):.0f} GB (max {max(r['gb'] for r in R):.0f} GB, "
                           f"{max(r['a100_h'] for r in R):.1f} A100-h)" if R[0]["gb"] is not None else ""))
        for f, x in sorted(fam.items(), key=lambda t: -t[1][0]):
            print(f"  {f:28s} {x[0]:5d} slices {x[1]:6d} windows {x[2]:7.0f} A100-h")


if __name__ == "__main__":
    main()
