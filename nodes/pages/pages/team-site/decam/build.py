#!/usr/bin/env python3
"""<team-site-url> the DECam distant-body search, everything in one place.

2026-11-19new" objects".
The password is only in the Worker secret DECAM_PASSWORD (username var DECAM_USERNAME); src/worker.js gates /decam-search/
with it and the rest of the site with ALE_PASSWORD.

Writes public/decam-search/:
  index.html          whole-sky scan progress (every 3 deg x 3 deg DECam cell, coloured by the 5 % slice that searched it),
                      the slices table, the leading new objects, the hill climb, what we have done
  catalogue/index.html  a copy of the Promising New Bodies page (rsi/decam-big-tno/progress/objects/index.html)
Reads (all local): planet-search/linker/data_real (every usable public DECam exposure; cached as decam/footprint.json),
each slice's regions.json, planet-search/conveyor/state.json, the search runs' grade.json and timing.json,
rsi/decam-big-tno/progress/objects/db_public.json (the catalogue's friend-safe cards), runs/r1|r2/state.json.
Friend-safe like the catalogue: no positions or dates of unreported bodies, nothing seen only in Rubin data, no names of
people. Usage: python3 -I decam/build.py [--deploy]"""
import glob, json, math, os, re, shutil, subprocess, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); SITE = os.path.dirname(HERE)
ALE = os.path.dirname(os.path.dirname(SITE)); PS = os.path.join(ALE, "planet-search"); DBT = os.path.join(ALE, "rsi", "decam-big-tno")
OUT = os.path.join(SITE, "public", "decam-search")
R = math.pi / 180


def cell_of(r, d):                                       # planet-search/archive5/select_regions.py, unchanged
    band = min(59, max(0, math.floor((d + 90) / 3)))
    dc = -90 + 3 * band + 1.5
    n = max(1, math.floor(360 * math.cos(dc * R) / 3))
    return band * 1000 + (math.floor(r / 360 * n) % n), n


def footprint():
    f = os.path.join(HERE, "footprint.json")
    if os.path.exists(f):
        return json.load(open(f))
    DR = os.path.join(PS, "linker", "data_real"); files = set()
    for site in ("ls6", "vista"):
        for line in open(os.path.join(DR, "decam_bulk", site + ".tsv")):
            files.add(line.split("\t")[2][:17])
    cells = defaultdict(lambda: [0, 0.0])                 # n_exp, exposure-hours
    for fn in sorted(glob.glob(os.path.join(DR, "noirlab_decam_meta", "decam_meta_*.json"))):
        d = json.load(open(fn)); ix = {k: i for i, k in enumerate(d["fields"])}
        for r in d["rows"]:
            if r[ix["stamp"]] in files and r[ix["ra"]] is not None:
                k, n = cell_of(float(r[ix["ra"]]), float(r[ix["dec"]])); c = cells[k]; c[0] += 1; c[1] += float(r[ix["exptime"]] or 0) / 3600
                files.discard(r[ix["stamp"]])            # one row per exposure
    out = {}
    for k, (n_exp, h) in cells.items():
        b, col = divmod(k, 1000); dc = -90 + 3 * b + 1.5; n = max(1, math.floor(360 * math.cos(dc * R) / 3))
        area = (math.sin((dc + 1.5) * R) - math.sin((dc - 1.5) * R)) * (360 / n) * (180 / math.pi)
        out[str(k)] = dict(b=b, c=col, n=n, e=n_exp, h=round(h, 1), a=round(area, 2))
    json.dump(out, open(f, "w"))
    return out


def run_stats(run):
    """Union over a search run's graded program runs: for each region the latest grade; recall, leads, GPU-hours."""
    per, leads, gpu = {}, 0, 0.0
    gens = sorted(glob.glob(os.path.join(DBT, "runs", run, "gens", "*", "grade", "grade.json")), key=os.path.getmtime)
    best = None
    for g in gens:
        G = json.load(open(g))
        if "per_scenario" not in G:
            continue
        T = os.path.join(os.path.dirname(os.path.dirname(g)), "timing.json"); T = json.load(open(T)) if os.path.exists(T) else {}
        for sid, x in G["per_scenario"].items():
            per[sid] = (x["known_found"], x["known_findable"], x.get("leads", 0), T.get(sid, 0))
        prog = os.path.basename(os.path.dirname(os.path.dirname(g))).split("-", 1)[-1]
        best = prog if best is None else best
    if not per:
        return None
    fo = sum(v[0] for v in per.values()); fa = sum(v[1] for v in per.values())
    return dict(regions=len(per), found=fo, findable=fa, leads=sum(v[2] for v in per.values()), gpu_h=round(sum(v[3] for v in per.values()) / 3600, 1),
                programs=sorted({os.path.basename(os.path.dirname(os.path.dirname(g))).split("-", 1)[-1] for g in gens}))


SLICES = [  # order of search; cells come from each slice's regions.json (archive10: only the regions searched in steps A and B)
    dict(id="s1", name="1st 5 %", src="archive5", run="s38", note="first search of 5 % of DECam's sky, chosen for seasons of coverage near the ecliptic"),
    dict(id="s2", name="2nd 5 %", src="archive10", run=["s10a", "s10b"], note="regions of the 10 % selection searched in two steps"),
    dict(id="s3", name="3rd 5 %", src="archive15", run="sa15", unit="a15", note="sky whose exposures were already at NERSC"),
    dict(id="s4", name="4th 5 %", src="ldt5", run="sldt5", unit="ldt5", note="weighted to sky the LDT can observe this winter"),
    dict(id="s5", name="5th 5 %", src="units/u05", run="su05", unit="u05", note="LDT-weighted; the continuous pipeline's first unit"),
    dict(id="s6", name="6th 5 %", src="units/u06", run="su06", unit="u06", note="LDT-weighted"),
    dict(id="s7", name="7th 5 %", src="units/u16", run="su16", unit="u16", note="LDT-weighted"),
    # 2026-11-22: the units after u16 were missing from the map
    dict(id="s8", name="8th 5 %", src="units/u17", run="su17", unit="u17", note="LDT-weighted"),
    dict(id="s9", name="9th (part)", src="units/u18", run="su18", unit="u18", note="the rest of the LDT-weighted sky, 23 regions"),
    dict(id="s10", name="10th 5 %", src="units/u19", run="su19", unit="u19", note="continuous pipeline"),
    dict(id="s11", name="11th 5 %", src="units/u20", run="su20", unit="u20", note="continuous pipeline"),
    dict(id="s12", name="12th 5 %", src="units/u21", run="su21", unit="u21", note="continuous pipeline"),
    dict(id="s13", name="13th 5 %", src="units/u22", run="su22", unit="u22", note="continuous pipeline"),
    dict(id="s14", name="14th 5 %", src="units/u23", run="su23", unit="u23", note="continuous pipeline"),
]
STAGE = {"searched": "searched", "search": "searching", "prep": "preparing (download, detection)", None: "searched"}


def slices(fp):
    st = {u["id"]: u for u in json.load(open(os.path.join(PS, "conveyor", "state.json")))["units"]}
    log = open(os.path.join(PS, "conveyor", "loop.out")).read()
    out = []
    for s in SLICES:
        d = json.load(open(os.path.join(PS, s["src"], "regions.json"))); regs = d["regions"]
        if s["src"] == "archive10":
            keep = set()
            for step in ("stepA", "stepB"):
                keep |= {k[:4] for k in json.load(open(os.path.join(PS, "archive10", step, "regions_exposures.json")))}
            regs = [r for r in regs if r["id"] in keep]
        cells = sorted({int(c) for r in regs for c in r["cells"]})
        u = st.get(s.get("unit"), {}); stage = u.get("stage") if s.get("unit") else "searched"
        runs = s["run"] if isinstance(s["run"], list) else [s["run"]]
        stats = [x for x in (run_stats(r) for r in runs) if x]
        S = None
        if stats:
            S = dict(regions=sum(x["regions"] for x in stats), found=sum(x["found"] for x in stats), findable=sum(x["findable"] for x in stats),
                     leads=sum(x["leads"] for x in stats), gpu_h=round(sum(x["gpu_h"] for x in stats), 1), programs=sorted({p for x in stats for p in x["programs"]}))
        prog = None
        if s.get("unit") and stage in ("search", "prep"):
            m = re.findall(rf'"{s["unit"]}": \{{"regions": (\d+), "built": (\d+), "done": (\d+), "running": (\d+)', log)
            if m:
                a, b, c, e = map(int, m[-1]); prog = dict(regions=a, built=b, done=c, running=e)
            hp = json.load(open(os.path.join(PS, "conveyor", "health.json"))).get("progress", {}).get(s["unit"])
            if hp and stage == "prep":
                prog = dict(prog or {}, detect_left=hp["left"])
        when = u.get("searched_at") or ""
        out.append(dict(id=s["id"], name=s["name"], note=s["note"], stage=STAGE.get(stage, stage), state=stage or "searched",
                        regions=len(regs), cells=cells, deg2=round(sum(fp.get(str(c), {}).get("a", 0) for c in cells)),
                        exposures=sum(int(r.get("exposures", 0)) for r in regs), tb=round(sum(float(r.get("tb", 0) or 0) for r in regs), 1),
                        stats=S, progress=prog, searched_at=when[:10]))
    return out


def bodies():
    db = json.load(open(os.path.join(DBT, "progress", "objects", "db_public.json")))
    out = []
    for c in db["cards"]:
        lad = {s["step"]: s for s in c["ladder"]}
        oa = lad.get("other_archives", {}).get("state"); rep = lad.get("reported", {}).get("state")
        filed = c.get("stage") == "filed"
        if not (filed or oa == "pass"):
            continue                                      # leading = reported, or seen by another telescope on 2+ nights
        out.append((0 if filed else 1, c.get("H") or 99, c))
    out.sort(key=lambda t: (t[0], t[1]))                 # reported first, then the largest
    W = ("cid", "label", "name", "stage", "stageTxt", "H", "D", "a", "e", "i", "r", "nights", "arc", "rms", "search", "summary", "group", "facts")
    slim = lambda c: dict({k: c.get(k) for k in W}, ladder=[dict(step=x["step"], state=x["state"]) for x in c["ladder"]])
    keep = [slim(t[2]) for t in out][:18]
    return db["asof"], keep, len(db["cards"]), [slim(c) for c in db["cards"]]


def climb():
    S = json.load(open(os.path.join(DBT, "runs", "r2", "state.json")))
    pts = []
    for g, v in sorted(S["gens"].items()):
        r = v.get("reward") if isinstance(v, dict) else None
        if r is not None:
            pts.append(dict(g=g, r=round(float(r), 4), h=v.get("gpu_hours")))
    return dict(best=S["best"]["gen"], reward=S["best"]["reward"], gpu_h=S["best"].get("gpu_hours"), pts=pts)


DONE = [  # what we have done, newest first; dates of our own work only
    ("2026-11-19", "Third and fourth 5 % searched and scored (61-63 % of findable known bodies recovered); every candidate checked against MPC files night by night and against other telescopes' archives; 7 turned out to be known bodies and were dropped."),
    ("2026-11-19", "Candidate database: one folder per body with every identity, position, exposure and check; the catalogue and this site are built from it."),
    ("2026-11-19", "Hill climb ended at generation 53: the program finds 83 % of findable known bodies in its four graded regions at 1.9 GPU-hours."),
    ("2026-11-18", "Continuous pipeline: 5 % of the sky at a time, downloads and detection at NERSC, searches on NERSC A100s and Vista GH200s, always with the climb's latest best program."),
    ("2026-11-17", "Whole-archive data movement: NOIRLab downloads, Globus transfers between NERSC and Vista, scratch management."),
    ("2026-11-15", "Candidates seen in other archives (Subaru HSC, CFHT, NSC, other DECam programs) started going to an observer for follow-up."),
    ("2026-11-14", "Second hill climb: two agents per round, everything open to them; 53 generations by 11-19."),
    ("2026-11-12", "The search task built: an agent writes the program that finds large distant bodies in one region's DECam detections; first hill climb, 36 rounds."),
    ("2026-11-09", "Second candidate from the earlier ten-field search reported to the Minor Planet Center."),
    ("2026-11-03", "First candidate from the ten-field search reported to the Minor Planet Center."),
]


def mask(h):
    """Rubin DP2 detections (the user's data rights) become 'a survey whose data we cannot share here', and the follow-up
    observer is not named. Rubin's own public MPC reports (ITF) stay."""
    h = re.sub(r"(SEEN )?in Rubin DP2 \((\d+) nights?; data under our data rights, not for OBSERVER_NAME or shared pages without approval\)",
               lambda m: (m.group(1) or "") + f"in a survey whose data we cannot share here ({m.group(2)} night{'s' if m.group(2) != '1' else ''})", h)
    h = h.replace("seen a second time only in Rubin data", "seen a second time only in data we cannot share here")
    return h.replace("OBSERVER_NAME", "an observer").replace("OBSERVER_NAME", "the observer")


def site_catalogue(h):
    """The catalogue page as shared here (mask() plus two labels of the page itself)."""
    h = mask(h)
    h = re.sub(r"(SEEN )?in Rubin DP2 \((\d+) nights?; data under our data rights, not for OBSERVER_NAME or shared pages without approval\)",
               lambda m: (m.group(1) or "") + f"in a survey whose data we cannot share here ({m.group(2)} night{'s' if m.group(2) != '1' else ''})", h)
    h = h.replace("seen a second time only in Rubin data", "seen a second time only in data we cannot share here")
    h = h.replace("(Subaru, CFHT, ESO, Rubin)", "(Subaru, CFHT, ESO, other DECam programs)")
    h = h.replace('["Rubin data", r.rubin', '["Unshared survey", r.rubin')
    h = h.replace("OBSERVER_NAME", "an observer").replace("OBSERVER_NAME", "the observer")
    left = [m.group(0) for m in re.finditer(r".{40}(Rubin DP2|DP2|OBSERVER_NAME|our data rights|approval).{20}", h)]
    if left:
        sys.exit("refusing to copy the catalogue: " + " | ".join(left[:3]))
    return h


def main():
    fp = footprint(); sl = slices(fp); asof, top, ncards, allc = bodies(); cl = climb()
    cover = round(sum(c["a"] for c in fp.values())); n_exp = sum(c["e"] for c in fp.values())
    owner = {}
    for s in sl:
        for c in s["cells"]:
            owner.setdefault(str(c), s["id"])
    D = dict(asof=time.strftime("%Y-%m-%d %H:%M"), catalogue_asof=asof, cover_deg2=cover, n_exp=n_exp, ncards=ncards,
             cells=[[int(k), v["e"], owner.get(k, "")] for k, v in fp.items()], slices=[{k: v for k, v in s.items() if k != "cells"} for s in sl],
             top=top, all=allc, climb=cl, done=DONE)
    txt = mask(json.dumps(D, separators=(",", ":"), ensure_ascii=False))
    for bad in ("Rubin DP2", "DP2", "OBSERVER_NAME", "our data rights", "observer@", "CLIENT_NAME", "CLIENT_NAME_ZH"):
        if bad in txt:
            sys.exit(f"refusing: '{bad}' in the page data")
    os.makedirs(os.path.join(OUT, "catalogue"), exist_ok=True)
    page = open(os.path.join(HERE, "page.html")).read().replace("/*DATA*/null", txt.replace("</", "<\\/"))
    open(os.path.join(OUT, "index.html"), "w").write(page)
    cat = site_catalogue(open(os.path.join(DBT, "progress", "objects", "index.html")).read())
    open(os.path.join(OUT, "catalogue", "index.html"), "w").write(cat)
    print(f"wrote {OUT}/index.html ({len(page) // 1024} kB): {len(fp)} cells, {cover} deg2, {len(top)} leading bodies; catalogue copied")
    if "--deploy" in sys.argv:
        env = dict(os.environ, CLOUDFLARE_AUTH_USE_KEYRING="false")
        subprocess.run(["wrangler", "deploy"], cwd=SITE, env=env, check=True)


if __name__ == "__main__":
    main()
