#!/usr/bin/env python3
"""The grader's scoring code, for the labelled regions in /app/open (or cuts of them made by tools/subset.py).

    python3 /app/tools/score.py OUT LABELS GRADE --sids sid1[,sid2] [--report GRADE/report.md] [--timing OUT/timing.json]

OUT/<sid>/candidates.json = [{"det": [row numbers], "p": number}, ...] is your program's output for region <sid>
(finder.py --scen .../scen/<sid> --out OUT/<sid>); LABELS/<sid>.npz is that region's labels (/app/open/labels, or the
labels folder subset.py wrote). At most 2,000 candidates per region are read (highest p first). The candidates of all
regions given are ranked by p together and the top K = 2 x (findable labelled bodies in those regions) are scored.
Found: >= 80 % of a candidate's detections belong to one body and cover >= 5 of its nights; each body counts once.
Findable: a labelled body with a detection on >= 5 nights in the region. reward (recall) = found weight / findable weight.
Per body (GRADE/grade.json "feedback", and the --report table), over all valid candidates of its region:
  found       a candidate in the scored top K is >= 80 % this body and covers >= 5 of its nights
  below_cut   such a candidate exists but is outside the scored top K (or beyond the 2,000 read per region)
  few_nights  its purest link (>= 80 % this body) covers only 2-4 of its nights
  mixed       a candidate holds it on >= 2 nights but under 80 % of that candidate's detections are this body
  none        no candidate holds it on 2 nights
If OUT/<sid>/trace.json = {"stages": [{"name": str, "groups": [[row numbers], ...]}, ...]} exists (at most 6 stages,
500,000 groups and 5 M row numbers per stage, 100 MB), the feedback also gives, per stage, the most of the body's nights in
one group that is >= 80 % this body (pure) and in any group (any). Never scored.
"leads" in grade.json: candidates in the scored top K whose detections are >= 80 % unlabelled, on >= 5 nights (false
linkages, known bodies that are missing from the labels, or bodies nobody knows).
gpu_hours is filled only with --timing (a JSON {sid: seconds} you write yourself); the grade measures it.
Writes GRADE/grade.json and GRADE/public.json and prints one summary line.
"""
import argparse, json, math, os
import numpy as np

PER_SCEN, FRAC, NMIN = 2000, 0.8, 5
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("truth"); ap.add_argument("grade")
ap.add_argument("--sids", default=""); ap.add_argument("--timing", default=""); ap.add_argument("--report", default="")
a = ap.parse_args()
sids = a.sids.split(",") if a.sids else sorted(f[:-4] for f in os.listdir(a.truth) if f.endswith(".npz"))
timing = json.load(open(a.timing)) if a.timing and os.path.exists(a.timing) else {}
os.makedirs(a.grade, exist_ok=True)

T, cands, status, ALLC = {}, [], {}, {}
for sid in sids:
    z = np.load(os.path.join(a.truth, sid + ".npz"))
    T[sid] = dict(body=z["det_body"], known=json.loads(str(z["known"])), planted=json.loads(str(z["planted"])))
    nrow = len(T[sid]["body"])
    p = os.path.join(a.out, sid, "candidates.json")
    try:
        C = json.load(open(p))
        assert isinstance(C, list)
        good = []
        for c in C:
            det = np.asarray(c["det"], dtype=np.int64); pv = float(c["p"])
            if det.ndim != 1 or len(det) == 0 or det.min() < 0 or det.max() >= nrow or not math.isfinite(pv):
                continue
            good.append((pv, np.unique(det)))
        good.sort(key=lambda x: -x[0])
        ALLC[sid] = good
        good = good[:PER_SCEN]
        cands += [(pv, sid, det) for pv, det in good]
        status[sid] = "ok" if C else "empty"
    except FileNotFoundError:
        status[sid] = "missing"
    except Exception as e:
        status[sid] = f"invalid: {type(e).__name__}"

cands.sort(key=lambda x: -x[0])
NIGHT = {sid: np.load(os.path.join(a.truth, sid + ".npz"))["det_night"] for sid in sids}
def findable_known(sid):
    body, night = T[sid]["body"], NIGHT[sid]
    m = np.flatnonzero((body >= 0) & (body < 100000))
    return [int(b) for b in np.unique(body[m]) if len(np.unique(night[m][body[m] == b])) >= NMIN]
FK = {sid: findable_known(sid) for sid in sids}
K = 2 * sum(len(v) for v in FK.values())
top = cands[:K]
cut = top[-1][0] if len(top) == K else None

found = {sid: {} for sid in sids}; leads = []
for rank, (pv, sid, det) in enumerate(top):
    lab = T[sid]["body"][det]; nights = NIGHT[sid][det]
    vals, cnt = np.unique(lab, return_counts=True)
    b = int(vals[np.argmax(cnt)])
    frac = cnt.max() / len(lab)
    if frac >= FRAC and b >= 0:
        nn = len(np.unique(nights[lab == b]))
        if nn >= NMIN and b not in found[sid]:
            found[sid][b] = dict(rank=rank, p=pv, nights=nn)
    elif b == -1 and frac >= FRAC:
        nn = len(np.unique(nights))
        if nn >= NMIN:
            leads.append(dict(sid=sid, rank=rank, p=pv, det=det.tolist(), nights=nn))

MAX_STAGES, MAX_GROUPS, MAX_ROWS, MAX_BYTES = 6, 500000, 5000000, 100 << 20


def read_trace(sid, body, night, fk):
    """The optional trace of a scenario: {body: {stage name: {pure_nights, any_nights}}} and a status line. Never raises."""
    p = os.path.join(a.out, sid, "trace.json")
    if not os.path.exists(p):
        return {}, "none"
    try:
        if os.path.getsize(p) > MAX_BYTES:
            return {}, "ignored: larger than 100 MB"
        stages = json.load(open(p))["stages"][:MAX_STAGES]
        want = np.array(sorted(fk), dtype=np.int64); res = {b: {} for b in fk}; note = []
        for st in stages:
            name = str(st["name"])[:40]; G = st["groups"][:MAX_GROUPS]
            lens = np.fromiter((len(g) for g in G), np.int64, len(G))
            if lens.sum() > MAX_ROWS:
                keep = int(np.searchsorted(np.cumsum(lens), MAX_ROWS)); G, lens = G[:keep], lens[:keep]; note.append(f"{name} cut at {keep} groups")
            for b in fk:
                res[b][name] = dict(pure_nights=0, any_nights=0)
            if not len(G) or not lens.sum():
                continue
            rows = np.concatenate([np.asarray(g, dtype=np.int64).ravel() for g in G]); gid = np.repeat(np.arange(len(G)), lens)
            ok = (rows >= 0) & (rows < len(body)); rows, gid = rows[ok], gid[ok]
            u = np.unique(gid * len(body) + rows); gid, rows = u // len(body), u % len(body)
            size = np.bincount(gid, minlength=len(G)); lab = body[rows]; m = np.isin(lab, want)
            if not m.any():
                continue
            key = gid[m] * 100000 + lab[m]
            uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            nn = np.bincount(np.unique(inv.astype(np.int64) * 1000000 + night[rows][m].astype(np.int64)) // 1000000, minlength=len(uk))
            frac = cnt / size[uk // 100000]; bo = uk % 100000
            for b in fk:
                k = bo == b
                if k.any():
                    pure = nn[k & (frac >= FRAC)]
                    res[b][name] = dict(pure_nights=int(pure.max()) if len(pure) else 0, any_nights=int(nn[k].max()))
        return res, "ok" + ("; " + "; ".join(note) if note else "")
    except Exception as e:
        return {}, f"invalid: {type(e).__name__}"


trace_status = {}
# per-body feedback over every valid candidate (the scored list is the top K of the 2,000 best per scenario)
ORDER = {"none": 0, "mixed": 1, "few_nights": 1, "below_cut": 3, "found": 4}
grank = {(sid, id(det)): r for r, (pv, sid, det) in enumerate(cands)}
feedback = {}
for sid in sids:
    body, night = T[sid]["body"], NIGHT[sid]; want = set(FK[sid])
    fb = {b: dict(stage="none") for b in FK[sid]}
    for ir, (pv, det) in enumerate(ALLC.get(sid, [])):
        lab = body[det]
        vals, cnt = np.unique(lab[lab >= 0], return_counts=True)
        for b, c in zip(vals.tolist(), cnt.tolist()):
            if b not in want:
                continue
            nn = len(np.unique(night[det][lab == b]))
            if nn < 2:
                continue
            frac = c / len(lab); r = grank.get((sid, id(det)))
            if frac >= FRAC and nn >= NMIN:
                stage = "found" if r is not None and r < K else "below_cut"
            else:
                stage = "few_nights" if frac >= FRAC else "mixed"
            new = dict(stage=stage, nights_linked=nn, purity=round(frac, 3), size=len(lab), p=pv,
                       rank=None if r is None else r + 1, rank_in_scenario=ir + 1)
            old = fb[b]
            key = lambda d: (ORDER[d["stage"]], d.get("nights_linked", 0), d.get("purity", 0), d.get("p", -math.inf))
            if key(new) > key(old) and not (old["stage"] == new["stage"] == "found"):   # found: keep the best-ranked one
                fb[b] = new
    for b in FK[sid]:
        assert (fb[b]["stage"] == "found") == (b in found[sid]), (sid, b, fb[b])
    tr, trace_status[sid] = read_trace(sid, body, night, FK[sid])
    for b in FK[sid]:
        if b in tr:
            fb[b]["trace"] = tr[b]
    feedback[sid] = {str(b): v for b, v in fb.items()}

wk = fk = 0.0; per = {}
for sid in sids:
    rk = dict(known_findable=len(FK[sid]), known_found=0, planted_found=sum(1 for b in found[sid] if b >= 100000))
    for b in FK[sid]:
        w = T[sid]["known"][b]["weight"]; wk += w
        if b in found[sid]:
            fk += w; rk["known_found"] += 1
    per[sid] = dict(status=status[sid], **rk, leads=sum(1 for l in leads if l["sid"] == sid), seconds=timing.get(sid),
                    trace=trace_status[sid])
Rk = fk / wk if wk else 0.0
reward = round(Rk, 4)
secs = [v for v in timing.values() if isinstance(v, (int, float))]
gpu_hours = round(sum(secs) / 3600, 4) if secs else None
json.dump(dict(reward=reward, R_known=round(Rk, 4), weights=dict(known=[fk, wk]),
               gpu_hours=gpu_hours, slots=K, candidates=len(cands), cut_p=cut, per_scenario=per, feedback=feedback,
               found={s: {str(k): v for k, v in f.items()} for s, f in found.items()}, leads=leads),
          open(os.path.join(a.grade, "grade.json"), "w"), indent=1)
json.dump(dict(reward=reward, gpu_hours=gpu_hours, status="ok" if all(v == "ok" for v in status.values()) else "partial",
               seconds=sum(secs) if timing else None),
          open(os.path.join(a.grade, "public.json"), "w"))
print(json.dumps(dict(reward=reward, gpu_hours=gpu_hours, slots=K, candidates=len(cands), cut_p=cut, leads=len(leads),
                      per={s: (v["status"], v["known_found"], v["known_findable"], v["leads"]) for s, v in per.items()})))

if a.report:
    WHAT = dict(found="found", below_cut="linked, ranked below the cut", few_nights="linked on too few nights",
                mixed="mixed with other detections", none="in no candidate")
    L = [f"# Per-body feedback: reward {reward:.4f}, GPU-hours {gpu_hours}, {len(cands)} candidates read, top {K} scored (cut p {cut})", "",
         "One row per findable known body (detections on >= 5 nights in its scenario). Outcome over all valid candidates of its",
         "scenario; nights linked / purity / rank describe the candidate that got furthest with it. Trace columns (if the program",
         "wrote trace.json): most of its nights in one group of that step, pure (>= 80 % this body) / any.", "",
         "| scenario | body | weight | nights | outcome | nights linked | purity | rank (p) | trace |", "|---|---|---|---|---|---|---|---|---|"]
    for i, sid in enumerate(sids, 1):
        for b in sorted(FK[sid], key=lambda b: (ORDER[feedback[sid][str(b)]["stage"]], -T[sid]["known"][b]["weight"], b)):
            v = feedback[sid][str(b)]; k = T[sid]["known"][b]
            rk = "" if v["stage"] == "none" else (f"{v['rank']} ({v['p']:.3f})" if v.get("rank") else f"not read: {v['rank_in_scenario']} in its scenario ({v['p']:.3f})")
            tr = "; ".join(f"{n}: {x['pure_nights']}/{x['any_nights']}" for n, x in v.get("trace", {}).items())
            L.append(f"| {i} | {b} | {k['weight']} | {k['nights']} | {WHAT[v['stage']]} | {v.get('nights_linked', '')} | {v.get('purity', '')} | {rk} | {tr} |")
    open(a.report, "w").write("\n".join(L) + "\n")
