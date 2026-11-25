"""Recall and score of a group on the known bodies in its windows.

Reads the group job's output (OUT/<sid>/{candidates.json, exposures.csv, info.json, known_phot.json}) and the group's
known bodies (groups/_private/<id>/known.json, from known_bodies.py). Rules are the task grader's (rsi/harness/grade.py):
  - a window keeps its --k candidates with the highest p (25 as in the task; the finder may write 200);
  - a candidate's state is propagated (two-body about the Sun, light time) to every exposure; it is a hit on a body
    when it lies within max(--tol arcsec, the body's 3-sigma) of it on >= 3 distinct nights; a body counts once;
    scored bodies are tried before neutral ones; candidates inside 25 au are ignored;
  - tier of a body in a window: 1 = >= 3 nights with one exposure at S/N >= 5; 2 = >= 3 nights whose night stack
    reaches 5; 3 = the stack of all exposures reaches 5 on >= 3 nights; 0 = not findable; None = orbit too uncertain.
    Measured from the job's forced photometry when it is there, otherwise the tier predicted from the depth model;
  - weights 1 / 2 / 4 for tiers 1 / 2 / 3.
recall = bodies found / findable bodies (tiers 1-3);  score = weight found / weight of all findable bodies.

Output: groups/_private/<id>/score_<role>_bodies.json (every body, private) and groups/<id>/score_<role>.json (numbers only).

Usage: ~/anaconda3/bin/python -I groups/kit/score_group.py g099 OUT_DIR [--role validation] [--k 25]
"""
import argparse, csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "linker"))
from orbits import rv_to_elements, elements_to_rv, GM, C_AU

W_TIER = {1: 1.0, 2: 2.0, 3: 4.0}


def positions(c, mjds, obs):
    r0, v0 = np.array(c["state"][:3]), np.array(c["state"][3:])
    a, e, i, Om, om, M0 = rv_to_elements(r0[None], v0[None])
    if not (e[0] < 1 and a[0] > 0):
        return None
    n = math.sqrt(GM / a[0] ** 3)
    ra, de = [], []
    for t, o in zip(mjds, obs):
        tau = 0.0
        for _ in range(3):
            r, _ = elements_to_rv(a, e, i, Om, om, M0 + n * (t - tau - c["epoch"]))
            g = r[0] - o
            tau = np.linalg.norm(g) / C_AU
        ra.append(math.degrees(math.atan2(g[1], g[0])) % 360); de.append(math.degrees(math.asin(g[2] / np.linalg.norm(g))))
    return np.array(ra), np.array(de)


def sep(ra1, de1, ra2, de2):
    d1, d2 = np.radians(de1), np.radians(de2)
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(np.radians(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600


def stack(f, e):
    f, e = np.array(f, float), np.array(e, float)
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    if not ok.any():
        return 0.0
    w = 1 / e[ok] ** 2
    return float(np.sum(f[ok] * w) / np.sqrt(np.sum(w)))


def measured_tier(body, phot, max_sigma):
    if body["max_sigma"] > max_sigma:
        return None
    vis = [x for x in phot if "hdu" in x and x.get("flag") == "" and x.get("err")]
    nights = {}
    for x in vis:
        nights.setdefault(x["night"], []).append(x)
    n1 = sum(any((x["snr"] or 0) >= 5 for x in v) for v in nights.values())
    n2 = sum(stack([x["flux"] for x in v], [x["err"] for x in v]) >= 5 for v in nights.values())
    tot = stack([x["flux"] for x in vis], [x["err"] for x in vis]) if vis else 0.0
    return 1 if n1 >= 3 else 2 if n2 >= 3 else 3 if (tot >= 5 and len(nights) >= 3) else 0


def load_cands(f):
    out = []
    try:
        d = json.load(open(f))
        for c in (d.get("candidates", []) if isinstance(d, dict) else d):
            s = [float(x) for x in c["state"]]
            if len(s) == 6 and all(map(math.isfinite, s)) and math.isfinite(float(c["epoch_mjd"])):
                out.append(dict(epoch=float(c["epoch_mjd"]), state=s, p=float(c.get("p", 0.0)), mag=c.get("mag")))
    except Exception:
        pass
    return sorted(out, key=lambda c: -c["p"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gid"); ap.add_argument("out")
    ap.add_argument("--role", default="all"); ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--tol", type=float, default=2.0); ap.add_argument("--max-sigma", type=float, default=1.5)
    ap.add_argument("--known", help="known.json (default groups/_private/<id>/known.json)")
    ap.add_argument("--group-dir", help="default groups/<id>"); ap.add_argument("--private-dir", help="default groups/_private/<id>")
    a = ap.parse_args()
    gdir, pdir = a.group_dir or os.path.join(ROOT, a.gid), a.private_dir or os.path.join(ROOT, "_private", a.gid)
    W = json.load(open(os.path.join(gdir, "windows.json")))["windows"]
    K = json.load(open(a.known or os.path.join(pdir, "known.json")))["windows"]
    tw = gw = 0.0
    per_tier = {t: [0, 0] for t in (1, 2, 3)}
    bodies_out, wins, nrun, nmeas, seconds, leads = [], 0, 0, 0, 0.0, 0
    for w in W:
        if a.role != "all" and w["role"] != a.role:
            continue
        d = os.path.join(a.out, w["sid"])
        if not os.path.exists(os.path.join(d, "done.json")):
            continue
        wins += 1
        info = json.load(open(os.path.join(d, "info.json")))
        seconds += info.get("seconds") or 0
        if not os.path.exists(os.path.join(d, "exposures.csv")):
            continue
        nrun += info.get("status") == "ok"
        E = list(csv.DictReader(open(os.path.join(d, "exposures.csv"))))
        stamps = [os.path.basename(e["image"])[:17] for e in E]
        mjds = [float(e["mjd_mid"]) for e in E]
        obs = [np.array([float(e["obs_x"]), float(e["obs_y"]), float(e["obs_z"])]) for e in E]
        night = [int(np.floor(m - 0.5)) for m in mjds]
        ph = json.load(open(os.path.join(d, "known_phot.json"))) if os.path.exists(os.path.join(d, "known_phot.json")) else None
        nmeas += ph is not None
        B = []
        for b in K.get(str(w["key"]), []):
            tier = measured_tier(b, ph.get(b["pdes"], []), a.max_sigma) if ph is not None else b["tier_pred"]
            B.append(dict(b, tier=tier, recs={x["stamp"]: x for x in b["recs"]}))
        for b in B:
            if b["tier"] in W_TIER:
                tw += W_TIER[b["tier"]]; per_tier[b["tier"]][1] += 1
        taken = {}
        cands = load_cands(os.path.join(d, "candidates.json"))[:a.k]
        for rank, c in enumerate(cands):
            pos = positions(c, mjds, obs)
            if pos is None or np.linalg.norm(c["state"][:3]) < 25:
                continue
            ra, de = pos
            hit = None
            for b in sorted(B, key=lambda b: b["tier"] not in W_TIER):
                if b["pdes"] in taken:
                    continue
                ok = set()
                for k, st in enumerate(stamps):
                    x = b["recs"].get(st)
                    if x is None or x.get("ra") is None:
                        continue
                    tol = max(a.tol, min(x.get("s_ra") or 0, 60), min(x.get("s_dec") or 0, 60))
                    if sep(ra[k], de[k], x["ra"], x["dec"]) <= tol:
                        ok.add(night[k])
                if len(ok) >= 3:
                    hit = b
                    break
            if hit is not None:
                taken[hit["pdes"]] = dict(rank=rank + 1, p=c["p"])
                if hit["tier"] in W_TIER:
                    gw += W_TIER[hit["tier"]]; per_tier[hit["tier"]][0] += 1
            elif c["p"] >= 0.5:
                leads += 1
        for b in B:
            bodies_out.append(dict(window=w["key"], sid=w["sid"], pdes=b["pdes"], name=b.get("name"), H=b.get("H"), r=b.get("r"), V=b.get("V"),
                                   tier=b["tier"], tier_pred=b["tier_pred"], found=b["pdes"] in taken, **taken.get(b["pdes"], {})))
    nf, nt = sum(v[0] for v in per_tier.values()), sum(v[1] for v in per_tier.values())
    pub = dict(group=a.gid, role=a.role, k=a.k, windows_done=wins, windows_run_ok=nrun, windows_with_measured_tiers=nmeas,
               findable_bodies=nt, found=nf, recall=round(nf / nt, 4) if nt else None,
               score=round(gw / tw, 4) if tw else None, weight_found=gw, weight_total=tw,
               by_tier={str(t): dict(found=v[0], total=v[1]) for t, v in per_tier.items()},
               not_scored=dict(too_faint=dict(found=sum(b["found"] for b in bodies_out if b["tier"] == 0),
                                              total=sum(b["tier"] == 0 for b in bodies_out)),
                               uncertain_orbit=dict(found=sum(b["found"] for b in bodies_out if b["tier"] is None),
                                                    total=sum(b["tier"] is None for b in bodies_out))),
               unmatched_candidates_p_ge_0p5=leads, finder_seconds=round(seconds, 1))
    json.dump(pub, open(os.path.join(gdir, f"score_{a.role}.json"), "w"), indent=1)
    os.makedirs(pdir, exist_ok=True)
    json.dump(dict(pub, bodies=bodies_out), open(os.path.join(pdir, f"score_{a.role}_bodies.json"), "w"), indent=1)
    print(json.dumps(pub, indent=1))


if __name__ == "__main__":
    main()
