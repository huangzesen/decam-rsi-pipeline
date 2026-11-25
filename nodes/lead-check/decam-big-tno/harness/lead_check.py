#!/usr/bin/env python3
"""Is a decam-big-tno lead a real body, and how large would it be? (2026-11-13.)

A program's linkage always fits an orbit (the program built it by fitting one), so a good fit proves nothing. The test:
1. Fit one barycentric two-body orbit (planet-search/linker/orbit_fit.py) to the lead's detections.
2. Predict its place in EVERY exposure of the region within its own arc +- 60 days where the prediction is good
   (1-sigma < 1.5"), inside the exposure's field (< 1.0 deg from its pointing), and where it should be detectable
   (predicted magnitude <= the exposure's m5 - 0.3, i.e. expected S/N >= ~6.6).
3. Count those exposures with a detection within max(1", 3 sigma) of the prediction whose magnitude is within 1 mag of
   the predicted one and that is not a static source (a detection within 1" on 2 or more other nights; 2026-11-22). The lead's own exposures always
   hit, so the test that counts is in the OTHER predicted exposures ("extra"): a real body shows up in most of them
   (losses: chip gaps, masked pixels, S/N near the limit), a chance linkage in almost none.
   Null: the same count at the prediction moved 30" (the background hit rate).
   Verdicts (2026-11-13 04:30, after B-rank2's 9 of 10 hits all turned out to be its own detections):
   SEEN AGAIN: >= 2 extra exposures with a detection, >= 50 % of the extra ones, <= 1 null hit;
   MISSING: >= 3 extra exposures and < 25 % with a detection, or < 40 % of all predicted ones;
   NO EXTRA EXPOSURE: fewer than 2 extra exposures (only other images can test it); else UNCLEAR.
   Phase 1 (fits, slow) is cached in <gen>_plan.json; results merge into <gen>.json by id.
4. Size: magnitude = magzero - 2.5 log10(flux) per detection, V from the band (V-band offsets as build_scen.py: g +0.6,
   r -0.15, i -0.4, z -0.5, Y -0.55, VR +0.1), H = V - 5 log10(r Delta) - 0.15 * phase angle; median over detections.
   Diameter D = 1329 km / sqrt(albedo) * 10^(-H/5) for albedo 0.05, 0.10, 0.20. Fluxes near S/N 5 are biased bright.
Reads the scenario on Vista (exposures.json, and per exposure only the rows it needs, on the login node); writes
runs/<run>/lead_check/<gen>.json. Usage: python3 -I harness/lead_check.py RUN GEN [--ids substring,...] [--refit]
"""
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("$ALE_HOME/planet-search/linker"))
import orbit_fit as O                                     # noqa: E402
from orbits import radec_to_unit, unit_to_radec, ARCSEC   # noqa: E402

VISTA = "$TACC_USER@vista.tacc.utexas.edu"
VSSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/vista-$TACC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", VISTA]
VPY = "module load gcc/13.2.0 python3/3.11.8 >/dev/null 2>&1; nice -n 10 $TACC_WORK/vista/dbt/env/bin/python -I -c"
SCEN = "$TACC_SCRATCH/dbt/scen"
if os.environ.get("DBT_SITE") == "nersc":      # search38: the 5 % search regions' scenarios are on NERSC scratch
    VSSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes",
            "$NERSC_USER@perlmutter.nersc.gov"]
    VPY = "nice -n 10 $NERSC_SCRATCH/dbt/env/bin/python -I -c"
    SCEN = "$NERSC_SCRATCH/dbt/scen"
COL = {"g": 0.6, "r": -0.15, "i": -0.4, "z": -0.5, "Y": -0.55, "VR": 0.1}
# remote: (a) rows -> columns; (b) per exposure, nearest detection to each query point (cKDTree on the exposure's rows)
REMOTE = r'''
import json, sys, numpy as np
from scipy.spatial import cKDTree
S = "%SCEN%"; q = json.load(sys.stdin); out = {}
for sid, job in q.items():
    L = lambda k: np.load(f"{S}/{sid}/det_{k}.npy", mmap_mode="r")
    ra, dec, ex, fl, fe, sn = L("ra"), L("dec"), L("exp"), L("flux"), L("ferr"), L("snr")
    rows = job.get("rows", [])
    o = {"rows": {str(r): [float(ra[r]), float(dec[r]), int(ex[r]), float(fl[r]), float(fe[r]), float(sn[r])] for r in rows}}
    st = np.searchsorted(ex, np.arange(int(ex[-1]) + 2))
    near = {}
    for e, pts in job.get("near", {}).items():
        e = int(e); a, b = st[e], st[e + 1]
        r_, d_ = np.radians(np.asarray(ra[a:b])), np.radians(np.asarray(dec[a:b]))
        T = cKDTree(np.stack([np.cos(d_) * np.cos(r_), np.cos(d_) * np.sin(r_), np.sin(d_)], 1))
        P = np.radians(np.asarray(pts, float))
        U = np.stack([np.cos(P[:, 1]) * np.cos(P[:, 0]), np.cos(P[:, 1]) * np.sin(P[:, 0]), np.sin(P[:, 1])], 1)
        dd, ii = T.query(U)
        near[str(e)] = [[float(np.degrees(x) * 3600), int(a + i), float(sn[a + i]), float(fl[a + i]), float(ra[a + i]), float(dec[a + i])]
                        for x, i in zip(dd, ii)]
    o["near"] = near
    # 2026-11-22: is each matched source static? Count other-night exposures with a detection within 1" of it
    hr = sorted({v[1] for vv in near.values() for v in vv if v[0] <= 4.5})
    if hr:
        mjd = np.array([x_["mjd_mid"] for x_ in json.load(open(f"{S}/{sid}/exposures.json"))], float)
        hr_a = np.array(hr); hra, hde = np.radians(np.asarray(ra)[hr_a]), np.radians(np.asarray(dec)[hr_a])
        HT = cKDTree(np.stack([np.cos(hde) * np.cos(hra), np.cos(hde) * np.sin(hra), np.sin(hde)], 1))
        hmjd = mjd[np.asarray(ex)[hr_a]]; hexp = np.asarray(ex)[hr_a]; seen = [set() for _ in hr]; same = [set() for _ in hr]
        R1 = np.radians(1.0 / 3600); R03 = np.radians(0.5 / 3600)
        dlo, dhi = float(np.degrees(hde.min())) - 0.01, float(np.degrees(hde.max())) + 0.01
        HU = np.stack([np.cos(hde) * np.cos(hra), np.cos(hde) * np.sin(hra), np.sin(hde)], 1)
        for c0 in range(0, len(ra), 4000000):
            cr, cd, ce = np.asarray(ra[c0:c0 + 4000000]), np.asarray(dec[c0:c0 + 4000000]), np.asarray(ex[c0:c0 + 4000000])
            m = (cd >= dlo) & (cd <= dhi)
            if not m.any():
                continue
            rr, de_, ee = np.radians(cr[m]), np.radians(cd[m]), ce[m]
            CU = np.stack([np.cos(de_) * np.cos(rr), np.cos(de_) * np.sin(rr), np.sin(de_)], 1)
            CT = cKDTree(CU)
            for k, idx in enumerate(CT.query_ball_point(HU, R1)):     # every detection within 1" of hit k (not only the nearest hit)
                for j in idx:
                    dt_ = abs(mjd[ee[j]] - hmjd[k])
                    if dt_ > 1.0:
                        seen[k].add(int(ee[j]))
                    elif dt_ < 0.5 and ee[j] != hexp[k] and np.linalg.norm(CU[j] - HU[k]) <= R03:
                        same[k].add(int(ee[j]))
        o["static"] = {str(r): len(sv) for r, sv in zip(hr, seen)}
        o["samenight"] = {str(r): sorted(sv) for r, sv in zip(hr, same) if sv}
    out[sid] = o
json.dump(out, sys.stdout)
'''.replace("%SCEN%", SCEN)


def remote(q):
    assert "'" not in REMOTE
    p = subprocess.run(VSSH + [f"{VPY} '{REMOTE}'"], input=json.dumps(q), capture_output=True, text=True,
                       timeout=int(os.environ.get("LC_TIMEOUT", "1800")))   # u20's 235 leads needed more (2026-11-23)
    if p.returncode != 0:
        sys.exit(f"Vista read failed: {p.stderr[-500:]}")
    return json.loads(p.stdout)


def exposures(sid, cache):
    f = os.path.join(cache, f"{sid}_exposures.json")
    if not os.path.exists(f):
        p = subprocess.run(VSSH + [f"cat {SCEN}/{sid}/exposures.json"], capture_output=True, text=True, timeout=300)
        open(f, "w").write(p.stdout)
    return json.load(open(f))


def fit_lead(t_utc, ra, dec, sig):
    R, t = O.observer_bary(t_utc, "W84"); uo = radec_to_unit(np.radians(ra), np.radians(dec))
    t0 = float(np.median(t)); k = int(np.argmin(abs(t - t0))); o = np.argsort(t)
    best = None
    for r in (25, 30, 35, 40, 45, 50, 60, 75, 100, 150, 250):
        for vs in (1.0, 0.5, 0.0):
            p0 = np.concatenate([R[k] + r * uo[k], vs * r * (uo[o[-1]] - uo[o[0]]) / max(t[o[-1]] - t[o[0]], 1e-3)])
            try:
                p, cov, res, chi2, dof = O.fit(p0, t0, t, R, uo, np.maximum(sig, 0.1))
            except Exception:
                continue
            el = O.elements(p, t0, helio=True)
            if not (0 < el["a"] < 1e5 and el["e"] < 1):
                continue
            rms = float(np.sqrt((res ** 2).mean()))
            if best is None or rms < best[0]:
                best = (rms, p, cov, t0, el)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("gen"); ap.add_argument("--ids", default=""); ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    RUN = os.path.join(HERE, "runs", a.run); cache = os.path.join(RUN, "lead_check"); os.makedirs(cache, exist_ok=True)
    cfg = json.load(open(os.path.join(RUN, "config.json"))); letter = cfg.get("names") or {s: chr(65 + i) for i, s in enumerate(cfg["sids"])}
    led = [x for x in json.load(open(os.path.join(RUN, "leads_sent.json"))) if x["gen"] == a.gen]
    if a.ids:
        led = [x for x in led if any(s in x["id"] for s in a.ids.split(","))]
    # 1. the lead rows
    q = {}
    for x in led:
        q.setdefault(x["sid"], {"rows": []})["rows"] += x["det"]
    planf = os.path.join(cache, f"{a.gen}_plan.json")
    cached = {d["id"]: d for d in json.load(open(planf))} if os.path.exists(planf) and not a.refit else {}
    R1 = remote({k: v for k, v in q.items() if any(x["sid"] == k and x["id"] not in cached for x in led)}) if any(x["id"] not in cached for x in led) else {}
    results, q2, plan = [], {}, []
    for x in led:
        if x["id"] in cached:
            d = cached[x["id"]]
            if d.get("fail"):
                results.append(dict(id=x["id"], region=letter[x["sid"]], verdict=d["fail"])); continue
            keep = [tuple(k) for k in d["keep"]]
            near = q2.setdefault(x["sid"], {"near": {}})["near"]
            for e, r_, d_, s_, pm in keep:
                near.setdefault(str(e), []).extend([[r_, d_], [r_ + 30.0 / 3600 / np.cos(np.radians(d_)), d_]])
            q2[x["sid"]].setdefault("rows", []).extend(x["det"])
            plan.append((x, keep, d["info"], set(x["det"]), set(d["own_exp"]))); continue
        try:
            one = lead_fit(x, R1, cache, q2, plan, results, letter)
        except Exception as err:
            one = dict(id=x["id"], fail=f"fit failed: {type(err).__name__}")
            results.append(dict(id=x["id"], region=letter[x["sid"]], verdict=one["fail"]))
        cached[x["id"]] = one
        json.dump(list(cached.values()), open(planf + ".tmp", "w")); os.replace(planf + ".tmp", planf)
    R2 = remote(q2) if q2 else {}
    used = {}
    for x, keep, info, own, own_exp in plan:
        try:
            results.append(lead_result(x, keep, info, own, own_exp, R2, used, letter, E=exposures(x["sid"], cache)))
        except Exception as err:
            results.append(dict(id=x["id"], region=letter[x["sid"]], **info, verdict=f"check failed: {type(err).__name__}"))
        print(json.dumps(results[-1]), flush=True)
    outf = os.path.join(cache, f"{a.gen}.json")
    old = {d["id"]: d for d in json.load(open(outf))} if os.path.exists(outf) else {}
    old.update({d["id"]: d for d in results})
    json.dump(list(old.values()), open(outf + ".tmp", "w"), indent=1); os.replace(outf + ".tmp", outf)


def lead_fit(x, R1, cache, q2, plan, results, letter):
    if True:
        E = exposures(x["sid"], cache)
        D = np.array([R1[x["sid"]]["rows"][str(r)] for r in x["det"]])      # ra dec exp flux ferr snr
        ex = D[:, 2].astype(int); t_utc = np.array([E[e]["mjd_mid"] for e in ex])
        sig = 0.08 + 0.6 / np.maximum(D[:, 5], 3.0)
        b = fit_lead(t_utc, D[:, 0], D[:, 1], sig)
        if b is None:
            results.append(dict(id=x["id"], region=letter[x["sid"]], verdict="no bound orbit")); return dict(id=x["id"], fail="no bound orbit")
        rms, p, cov, t0, el = b
        # size from the lead's own detections
        Rb, tt = O.observer_bary(t_utc, "W84"); u0 = O.directions(p, t0, tt, Rb)
        tau = 0.0; pos = O.propagate(p[:3], p[3:], tt - t0)
        delta = np.linalg.norm(pos - Rb, axis=1)
        from astropy.coordinates import get_body_barycentric
        from astropy.time import Time
        sun = get_body_barycentric("sun", Time(tt, format="mjd", scale="tdb")).xyz.to("au").value.T
        rh = np.linalg.norm(pos - sun, axis=1)
        earth_sun = np.linalg.norm(Rb - sun, axis=1)
        alpha = np.degrees(np.arccos(np.clip((rh ** 2 + delta ** 2 - earth_sun ** 2) / (2 * rh * delta), -1, 1)))
        mag = np.array([E[e]["magzero"] if E[e].get("magzero") is not None else np.nan for e in ex], float) - 2.5 * np.log10(np.maximum(D[:, 3], 1e-9))
        V = mag - np.array([COL.get(E[e]["band"], 0.0) for e in ex])
        H = V - 5 * np.log10(rh * delta) - 0.15 * alpha
        Hm = float(np.nanmedian(H)) if np.isfinite(H).any() else float("nan")   # an exposure without a zero point no longer voids H and the test (2026-11-15)
        # 2. predictions over the region's exposures near the arc
        allt = np.array([e["mjd_mid"] for e in E]); lo, hi = t_utc.min() - 60, t_utc.max() + 60
        cand = np.flatnonzero((allt >= lo) & (allt <= hi))
        Rc, tc = O.observer_bary(allt[cand], "W84")
        uc, s1 = O.predict_sigma(p, cov, t0, tc, Rc)
        rac, decc = unit_to_radec(uc); rac, decc = np.degrees(rac) % 360, np.degrees(decc)
        posc = O.propagate(p[:3], p[3:], tc - t0); dc = np.linalg.norm(posc - Rc, axis=1)
        sunc = get_body_barycentric("sun", Time(tc, format="mjd", scale="tdb")).xyz.to("au").value.T
        rhc = np.linalg.norm(posc - sunc, axis=1)
        keep = []
        for j, e in enumerate(cand):
            m = E[e]
            if m.get("ra0") is None or m.get("m5") is None or s1[j] > 1.5:
                continue
            sep = np.degrees(np.arccos(np.clip(radec_to_unit(np.radians(m["ra0"]), np.radians(m["dec0"])) @ uc[j], -1, 1)))
            pm = Hm + 5 * np.log10(rhc[j] * dc[j]) + COL.get(m["band"], 0.0)
            if sep < 1.0 and pm <= m["m5"] - 0.3:
                keep.append((int(e), float(rac[j]), float(decc[j]), float(s1[j]), float(pm)))
        q2.setdefault(x["sid"], {"near": {}}).setdefault("rows", []).extend(x["det"])
        near = q2.setdefault(x["sid"], {"near": {}})["near"]
        for e, r_, d_, s_, pm in keep:
            off = 30.0 / 3600
            near.setdefault(str(e), []).extend([[r_, d_], [r_ + off / np.cos(np.radians(d_)), d_]])
        info = dict(rms=round(rms, 3), r=round(float(el["r"]), 2), a=round(float(el["a"]), 2), e=round(float(el["e"]), 3),
                    i=round(float(el["i"]), 2), H=round(Hm, 2) if np.isfinite(Hm) else None,
                    H_spread=round(float(np.nanstd(H)), 2) if np.isfinite(Hm) else None, n=len(ex),
                    nights=int(len(np.unique(np.floor(t_utc - 0.5)))), snr_med=round(float(np.median(D[:, 5])), 1),
                    arc_days=round(float(t_utc.max() - t_utc.min()), 1))
        own_exp = set(int(v) for v in ex)
        plan.append((x, keep, info, set(x["det"]), own_exp))
        return dict(id=x["id"], keep=keep, info=info, own_exp=sorted(own_exp))


def blocked_exposures(detail, why):
    """Extra exposures whose predicted place is taken by another source (too bright/faint, static, own place, unmoving):
    they cannot test the body either way (TRKSUB01, 2026-11-22: 'MISSING should need an empty place')."""
    b = {h["exp"] for h in detail if (h.get("dmag") is not None and abs(h["dmag"]) > 1.0) or (h.get("static_nights") or 0) >= 2}
    for w in why:
        b |= set(w["exp"]) if isinstance(w["exp"], list) else {w["exp"]}
    return b


def verdict_v5(n, hit, extra, extra_hit, null, blocked, hit_nights=None):
    """v6 (2026-11-22, TRKSUB01's comparison: faint bodies, known ones included, are found in well under half of their extra
    exposures; with all tests a false hit is 0.08 % per exposure): SEEN AGAIN = 2 or more kept extra hits on 2 or more
    nights (null <= 1); the old 50 % share still counts when the nights are not known."""
    ne = extra - blocked; nn = n - blocked
    if not n:
        return "no test (no predicted exposure)"
    # the 50 % share applies only when nights are unknown (as the docstring says; before 2026-11-22 23:0x it also let
    # one-night hit sets through, 21 cards; TRKSUB01's stage check)
    if extra_hit >= 2 and null <= 1 and ((hit_nights is not None and hit_nights >= 2) or (hit_nights is None and extra_hit >= 0.5 * ne)):
        return "SEEN AGAIN (detected where predicted in exposures the program did not use)"
    if (ne >= 3 and extra_hit < 0.25 * ne) or (nn > 0 and hit / nn < 0.4):
        return "MISSING (not detected where it should be)"
    if ne < 2:
        return ("NOT TESTED (other sources sit at the predicted places)" if blocked else
                "NO EXTRA EXPOSURE (only other images can test it)")
    return "UNCLEAR"


def lead_result(x, keep, info, own, own_exp, R2, used, letter, E=None):
    """2026-11-22 (TRKSUB01: on C###, C###, C### and C### every 'found' extra was a galaxy or a star of S/N 50-138): a source
    near the prediction counts only if its magnitude is within 1 mag of the predicted one and no source sits there on 2 or
    more other nights (static)."""
    cnt = used.setdefault(x["sid"], {})
    st = R2[x["sid"]].get("static", {})
    hit = null = own_hit = extra = extra_hit = rej_mag = rej_static = 0
    detail = []

    def ok(r, e, pm):
        nonlocal rej_mag, rej_static
        if r[1] in own:
            return True
        if len(r) > 3 and E is not None and E[e].get("magzero") is not None and r[3] > 0:
            if abs(E[e]["magzero"] - 2.5 * np.log10(r[3]) - pm) > 1.0:
                rej_mag += 1; return False
        if st.get(str(r[1]), 0) >= 2:
            rej_static += 1; return False
        return True
    rows2 = R2[x["sid"]].get("rows", {})
    ownpos = [(v[0], v[1], v[2]) for r, v in rows2.items() if int(r) in own]          # ra dec exp of the lead's detections

    def sepas(r1, d1, r2, d2):
        return float(np.hypot(((r1 - r2 + 180) % 360 - 180) * np.cos(np.radians(d1)), d1 - d2) * 3600)
    # 2026-11-22 (TRKSUB01, C###): (c) a hit within 0.5" of one of the lead's own detection places is that static place;
    # (d) hits of one night must move as predicted: two hits within 0.3" of each other while the prediction moved > 1" are static
    bad = set(); why = []
    pre = []
    start = dict(cnt)                                  # the pre-pass reads the same query slots; restore the counter after it
    for e, r_, d_, s_, pm in keep:
        k = cnt.get(e, 0); r0 = R2[x["sid"]]["near"][str(e)][k]; cnt[e] = k + 2
        if e not in own_exp and r0[0] <= max(1.0, 3 * s_) and len(r0) > 5 and r0[1] not in own:
            # static own place: the hit sits on one of the lead's own detection places although the predicted place has moved
            # > 1.5" away from it by this exposure (a real body seen minutes after its own detection is near that place too)
            if any(oe != e and sepas(r0[4], r0[5], ora, odec) < 0.5 and sepas(r_, d_, ora, odec) > 1.5 for ora, odec, oe in ownpos):
                bad.add((e, r0[1])); why.append(dict(exp=int(e), why="own place", d=round(min(sepas(r0[4], r0[5], a_, b_) for a_, b_, _ in ownpos), 2),
                                                        pred_moved=round(min(sepas(r_, d_, a_, b_) for a_, b_, _ in ownpos), 2)))
            pre.append((e, r0, r_, d_))
    # (e) (TRKSUB01, C###): a hit whose sky place still holds a detection (within 0.5": two S/N 5 places scatter ~0.3")
    # in another exposure of that night in which the body should have moved > 1.5" from it is a static source
    sn_ = R2[x["sid"]].get("samenight", {})
    if E is not None and pre:
        tk = sorted((E[e]["mjd_mid"], r_, d_) for e, r_, d_, s_, pm in keep)
        rates = [sepas(a[1], a[2], b[1], b[2]) / ((b[0] - a[0]) * 24) for a, b in zip(tk, tk[1:]) if 0 < b[0] - a[0] < 0.5]
        rate = float(np.median(rates)) if rates else 3.0             # arcsec per hour
        if os.environ.get("LC_DEBUG"):
            print("DEBUG", x["id"][-20:], "rate", round(rate, 2), "pre", [(int(e), r0[1], round(r0[0], 2)) for e, r0, r_, d_ in pre],
                  "samenight", {k: v for k, v in sn_.items() if any(int(k) == r0[1] for _, r0, _, _ in pre)}, file=sys.stderr)
        for e, r0, r_, d_ in pre:
            for xe in sn_.get(str(r0[1]), []):
                moved = rate * abs(E[xe]["mjd_mid"] - E[e]["mjd_mid"]) * 24
                if moved > 1.5:
                    bad.add((e, r0[1])); why.append(dict(exp=int(e), why="same-night static", other_exp=int(xe), body_moved=round(moved, 2)))
                    break
    for i_ in range(len(pre)):
        for j_ in range(i_ + 1, len(pre)):
            (e1, h1, p1r, p1d), (e2, h2, p2r, p2d) = pre[i_], pre[j_]
            if E is not None and abs(E[e1]["mjd_mid"] - E[e2]["mjd_mid"]) < 0.5 and sepas(h1[4], h1[5], h2[4], h2[5]) < 0.3 \
                    and sepas(p1r, p1d, p2r, p2d) > 1.0:
                bad.add((e1, h1[1])); bad.add((e2, h2[1]))
                why.append(dict(exp=[int(e1), int(e2)], why="unmoving", hit_sep=round(sepas(h1[4], h1[5], h2[4], h2[5]), 2),
                                pred_sep=round(sepas(p1r, p1d, p2r, p2d), 2)))
    cnt.clear(); cnt.update(start)
    rej_own = len(bad)
    for e, r_, d_, s_, pm in keep:
        k = cnt.get(e, 0); res = R2[x["sid"]]["near"][str(e)][k:k + 2]; cnt[e] = k + 2
        tol = max(1.0, 3 * s_)
        if (e, res[0][1]) in bad:
            res = [[1e9] + list(res[0][1:]), res[1]]
        mine = e in own_exp
        extra += not mine
        if not mine and res[0][0] <= tol and res[0][1] not in own:
            r0 = res[0]; dm = (E[e]["magzero"] - 2.5 * np.log10(r0[3]) - pm) if (len(r0) > 3 and E is not None and E[e].get("magzero") is not None and r0[3] > 0) else None
            detail.append(dict(exp=int(e), sep=round(r0[0], 2), snr=round(r0[2], 1), dmag=None if dm is None else round(float(dm), 2), static_nights=st.get(str(r0[1]))))
        if res[0][0] <= tol and ok(res[0], e, pm):
            hit += 1; own_hit += res[0][1] in own
            extra_hit += (not mine) and (res[0][1] not in own)
        null += res[1][0] <= tol and ok(res[1], e, pm)
    n = len(keep)
    bset = blocked_exposures(detail, why)
    blocked = len(bset)
    kept_exps = [h["exp"] for h in detail if h["exp"] not in bset]
    hit_nights = len({int(E[e_]["mjd_mid"] - 0.5) for e_ in kept_exps}) if E is not None else None
    Hh = info.get("H")
    D_km = {f"{alb:.2f}": round(1329 / np.sqrt(alb) * 10 ** (-Hh / 5)) for alb in (0.05, 0.10, 0.20)} if Hh is not None else {}
    verdict = verdict_v5(n, hit, extra, extra_hit, null, blocked, hit_nights)
    return dict(id=x["id"], region=letter[x["sid"]], **info, expected=n, found=hit, found_not_in_lead=hit - own_hit,
                extra=extra, extra_found=extra_hit, null_hits=int(null), rejected_magnitude=rej_mag, rejected_static=rej_static, rejected_own_place_or_unmoving=rej_own, rejected_why=why[:12], blocked=blocked, hit_nights=hit_nights, extra_hits=detail,
                D_km=D_km, verdict=verdict, check="v6b (v5 tests; SEEN AGAIN = 2+ kept extra hits on 2+ nights, 2026-11-22)")


if __name__ == "__main__":
    main()
