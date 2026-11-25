#!/usr/bin/env python3
"""Extend a decam-big-tno lead's arc into the region's other seasons (2013-2026) and predict where it is now.

Why (2026-11-13): OBSERVER_NAME can recover only bodies that are well placed now and whose place is known to arcseconds
(his reply of 2026-11-12: "they are all currently in the other direction of the Sun ... Can you do a search in the
opposite direction of the Sun"). A one-season arc from 2018 predicts 2026 to arcminutes or worse. The region's own
detection lists hold every public DECam exposure 2013-2026, so the body may be there in other seasons.

Method, one season at a time, nearest (smallest predicted uncertainty) first:
1. Fit the orbit (lead_check.fit_lead) to the detections so far; predict every unused exposure of the region with its
   1-sigma (predict_sigma); keep exposures inside the field (< 1.1 deg), bright enough (pred. mag <= m5 - 0.3) and
   sigma < 300".
2. Take the season (exposures within 200 days of each other) with the smallest median sigma and >= 2 nights; read on
   Vista every detection within max(2", 4 sigma) of the prediction in those exposures.
3. Anchor: each detection in the exposure with the smallest sigma. Refit with it added; predict the season again; count
   the other exposures with a detection within max(1", 3 sigma). Keep the anchor with the most nights (>= 2 other
   nights; or, when at most 20 anchors were possible, >= 1 other exposure whose source moved >= 1.5" from the anchor;
   joint rms <= 0.30"); add the matches; repeat from 1. Stop when no season qualifies (a season = >= 2 exposures).
A chance match needs a source within ~1" on 2+ further nights of one orbit: background ~0.75 sources per arcmin^2 gives
~1e-6 per anchor. Writes runs/<run>/arc/<lead id>.json (added detections, orbit, arc, place and 1-sigma on DATES).
Usage: ~/anaconda3/bin/python -I harness/arc_extend.py RUN ID_SUBSTRING [...]
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "harness"))
import lead_check as LC                                      # noqa: E402
from lead_check import O, radec_to_unit, unit_to_radec      # noqa: E402  (lead_check adds planet-search/linker to the path)

DATES = ["2026-11-20", "2026-12-15", "2027-01-15"]
REMOTE = r'''
import json, sys, numpy as np
from scipy.spatial import cKDTree
S = "%SCEN%"; q = json.load(sys.stdin); out = {}
for sid, job in q.items():
    L = lambda k: np.load(f"{S}/{sid}/det_{k}.npy", mmap_mode="r")
    ra, dec, ex, fl, fe, sn = L("ra"), L("dec"), L("exp"), L("flux"), L("ferr"), L("snr")
    st = np.searchsorted(ex, np.arange(int(ex[-1]) + 2))
    o = {"rows": {str(r): [float(ra[r]), float(dec[r]), int(ex[r]), float(fl[r]), float(fe[r]), float(sn[r])] for r in job.get("rows", [])}}
    ball = {}
    for e, pts in job.get("ball", {}).items():
        e = int(e); a, b = st[e], st[e + 1]
        r_, d_ = np.radians(np.asarray(ra[a:b])), np.radians(np.asarray(dec[a:b]))
        T = cKDTree(np.stack([np.cos(d_) * np.cos(r_), np.cos(d_) * np.sin(r_), np.sin(d_)], 1))
        res = []
        for pra, pdec, rad in pts:
            P = np.radians([pra, pdec]); U = [np.cos(P[1]) * np.cos(P[0]), np.cos(P[1]) * np.sin(P[0]), np.sin(P[1])]
            ii = T.query_ball_point(U, np.radians(rad / 3600))
            res.append([[int(a + i), float(ra[a + i]), float(dec[a + i]), float(sn[a + i])] for i in ii])
        ball[str(e)] = res
    o["ball"] = ball; out[sid] = o
json.dump(out, sys.stdout)
'''.replace("%SCEN%", LC.SCEN)


def remote(q):
    import subprocess
    assert "'" not in REMOTE
    p = subprocess.run(LC.VSSH + [f"{LC.VPY} '{REMOTE}'"], input=json.dumps(q), capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        sys.exit(f"Vista read failed: {p.stderr[-500:]}")
    return json.loads(p.stdout)


def refit(p, t0, t_utc, ra, dec, sig):
    R, t = O.observer_bary(t_utc, "W84"); uo = radec_to_unit(np.radians(ra), np.radians(dec))
    p2, cov, res, chi2, dof = O.fit(p, t0, t, R, uo, np.maximum(sig, 0.1))
    return p2, cov, float(np.sqrt((res ** 2).mean())), float(np.abs(res).max())


def predict(p, cov, t0, mjd_utc, site="W84"):
    R, t = O.observer_bary(np.atleast_1d(mjd_utc), site)
    u, s1 = O.predict_sigma(p, cov, t0, t, R)
    ra, dec = unit_to_radec(u)
    return np.degrees(ra) % 360, np.degrees(dec), s1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("ids", nargs="+"); a = ap.parse_args()
    RUN = os.path.join(HERE, "runs", a.run); cache = os.path.join(RUN, "lead_check"); outd = os.path.join(RUN, "arc")
    os.makedirs(outd, exist_ok=True)
    led = [x for x in json.load(open(os.path.join(RUN, "leads_sent.json"))) if any(s in x["id"] for s in a.ids)]
    for x in led:
        sid = x["sid"]; E = LC.exposures(sid, cache)
        allt = np.array([e["mjd_mid"] for e in E])
        R1 = remote({sid: {"rows": x["det"]}})[sid]["rows"]
        D = {int(r): v for r, v in R1.items()}                          # row -> ra dec exp flux ferr snr
        rows = list(x["det"])
        def arrays():
            M = np.array([D[r] for r in rows]); ex = M[:, 2].astype(int)
            return allt[ex], M[:, 0], M[:, 1], 0.08 + 0.6 / np.maximum(M[:, 5], 3.0), ex
        t_utc, ra, dec, sig, ex = arrays()
        b = LC.fit_lead(t_utc, ra, dec, sig)
        if b is None:
            print(x["id"], "no bound orbit"); continue
        rms, p, cov, t0, el = b
        lc = {d["id"]: d for f in [os.path.join(cache, x["gen"] + ".json")] if os.path.exists(f) for d in json.load(open(f))}
        Hm = (lc.get(x["id"]) or {}).get("H")
        log, tried = [], set()
        for it in range(12):
            used = set(ex.tolist()) | tried
            cand = np.array([k for k in range(len(E)) if k not in used and E[k].get("ra0") is not None and E[k].get("m5") is not None])
            if not len(cand):
                break
            pr, pd, s1 = predict(p, cov, t0, allt[cand])
            # field, depth (H from the lead check; r ~ distance now, Delta ~ r - 1 near opposition)
            rr_ = float(el["r"]); ok = []
            for j, k in enumerate(cand):
                m = E[k]
                sep = np.degrees(np.arccos(np.clip(radec_to_unit(np.radians(m["ra0"]), np.radians(m["dec0"])) @ radec_to_unit(np.radians(pr[j]), np.radians(pd[j])), -1, 1)))
                deep = Hm is None or Hm + 5 * np.log10(rr_ * max(rr_ - 1, 1)) + LC.COL.get(m["band"], 0.0) <= m["m5"]
                if sep < 1.1 and s1[j] < 300 and deep:
                    ok.append(j)
            if not ok:
                break
            ok = np.array(ok); ks = cand[ok]; tk = allt[ks]
            # seasons: split where consecutive exposures are > 200 d apart
            order = np.argsort(tk); groups, cur = [], [order[0]]
            for i0, i1 in zip(order[:-1], order[1:]):
                if tk[i1] - tk[i0] > 200:
                    groups.append(cur); cur = []
                cur.append(i1)
            groups.append(cur)
            groups = [g for g in groups if len(g) >= 2]
            if not groups:
                break
            g = min(groups, key=lambda g: np.median(s1[ok][g]))
            gk = ks[g]; gs = s1[ok][g]; gra, gdec = pr[ok][g], pd[ok][g]
            rad = np.clip(np.maximum(60.0, 4 * gs), 60.0, 1200.0)           # >= 60": also tells whether the place is on a CCD
            q = {sid: {"ball": {str(int(k)): [[float(r_), float(d_), float(rr)]] for k, r_, d_, rr in zip(gk, gra, gdec, rad)}}}
            B = remote(q)[sid]["ball"]
            hits = {int(k): B[str(int(k))][0] for k in gk}
            def near(lst, r0, d0, lim):
                if not lst:
                    return []
                U = radec_to_unit(np.radians([h[1] for h in lst]), np.radians([h[2] for h in lst]))
                d = np.degrees(np.arccos(np.clip(U @ radec_to_unit(np.radians(r0), np.radians(d0)), -1, 1))) * 3600
                return [h for h, x_ in zip(lst, d) if x_ <= lim]
            covered = [int(k) for k in gk if hits[int(k)]]                  # any source within 60": on a CCD
            if not covered:
                tried |= set(int(k) for k in gk)
                log.append(dict(season=f"{tk[g].min():.0f}-{tk[g].max():.0f}", exposures=len(g), covered=0, found=False)); continue
            jj = [j for j, k in enumerate(gk) if int(k) in covered]
            j0 = min(jj, key=lambda j: gs[j]); anchor_k = int(gk[j0])
            anchors = near(hits[anchor_k], gra[j0], gdec[j0], max(2.0, 4 * gs[j0]))
            best = None
            for row, ar, ad, asn in anchors[:400]:
                if row in D:
                    continue
                tt = np.append(t_utc, allt[anchor_k]); rr = np.append(ra, ar); dd = np.append(dec, ad)
                ss = np.append(sig, 0.08 + 0.6 / max(asn, 3.0))
                try:
                    p2, c2, rms2, mx2 = refit(p, t0, tt, rr, dd, ss)
                except Exception:
                    continue
                if rms2 > 0.30:
                    continue
                qr, qd, qs = predict(p2, c2, t0, allt[gk])
                add = []
                for j, k in enumerate(gk):
                    if int(k) == anchor_k:
                        continue
                    tol = max(1.0, 3 * qs[j])
                    for r2, ra2, de2, sn2 in hits[int(k)]:
                        dsep = np.degrees(np.arccos(np.clip(radec_to_unit(np.radians(ra2), np.radians(de2)) @ radec_to_unit(np.radians(qr[j]), np.radians(qd[j])), -1, 1))) * 3600
                        if dsep <= tol:
                            add.append((dsep, r2, ra2, de2, sn2, int(k)))
                nights = len(np.unique(np.floor(allt[[a_[5] for a_ in add]] - 0.5))) if add else 0
                # one more exposure is enough when the place was predicted tightly (few anchors to try) and the
                # matched source has moved >= 1.5" from the anchor (not a static source): chance ~1e-3 per anchor
                moved = [a_ for a_ in add if np.degrees(np.arccos(np.clip(radec_to_unit(np.radians(a_[2]), np.radians(a_[3])) @ radec_to_unit(np.radians(ar), np.radians(ad)), -1, 1))) * 3600 >= 1.5]
                good = nights >= 2 or (len(anchors) <= 20 and len(moved) >= 1)
                if good and (best is None or (nights, len(add)) > (best[0], len(best[5]))):
                    best = (nights, row, ar, ad, asn, add)
            if best is None:
                log.append(dict(season=f"{tk[g].min():.0f}-{tk[g].max():.0f}", exposures=len(g), covered=len(covered), anchors=len(anchors),
                                sigma_med=round(float(np.median(gs)), 1), found=False))
                tried |= set(int(k) for k in gk)
                continue
            nights, row, ar, ad, asn, add = best
            newrows = [row]; seen_k = {anchor_k}
            for dsep, r2, ra2, de2, sn2, k in sorted(add):
                if k not in seen_k and r2 not in D:
                    newrows.append(r2); seen_k.add(k)
            NR = remote({sid: {"rows": newrows}})[sid]["rows"]
            for r_, v in NR.items():
                D[int(r_)] = v
            rows = rows + newrows
            t_utc, ra, dec, sig, ex = arrays()
            p, cov, rms, mx = refit(p, t0, t_utc, ra, dec, sig)
            if rms > 0.30:                                         # joint fit broke: undo
                rows = rows[:-len(newrows)]; t_utc, ra, dec, sig, ex = arrays(); p, cov, rms, mx = refit(p, t0, t_utc, ra, dec, sig)
                tried |= set(int(k) for k in gk); log.append(dict(season=f"{tk[g].min():.0f}", found=False, why="joint fit > 0.3\""))
                continue
            el = O.elements(p, t0, helio=True)
            log.append(dict(season=f"{tk[g].min():.0f}-{tk[g].max():.0f}", exposures=len(g), sigma_med=round(float(np.median(gs)), 1),
                            found=True, added=len(newrows), nights=nights + 1, rms=round(rms, 3)))
            print(x["id"], log[-1], flush=True)
        from astropy.time import Time
        now = [Time(d).mjd for d in DATES]
        nr, nd, ns = predict(p, cov, t0, now, "W84")
        el = O.elements(p, t0, helio=True)
        nights = len(np.unique(np.floor(t_utc - 0.5)))
        res = dict(id=x["id"], n=len(rows), nights=int(nights), arc_days=round(float(t_utc.max() - t_utc.min()), 1),
                   arc_years=round(float(t_utc.max() - t_utc.min()) / 365.25, 2), rms=round(float(rms), 3),
                   a=round(float(el["a"]), 3), e=round(float(el["e"]), 4), i=round(float(el["i"]), 3), r=round(float(el["r"]), 2),
                   rows=rows, seasons=log, epoch_tdb=t0, p=[float(v) for v in p], cov=np.asarray(cov).tolist(),
                   now=[dict(date=d, ra=round(float(r_), 5), dec=round(float(d_), 5), sigma_arcsec=round(float(s_), 1))
                        for d, r_, d_, s_ in zip(DATES, nr, nd, ns)])
        json.dump(res, open(os.path.join(outd, x["id"] + ".json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "p", "cov", "seasons")}), flush=True)


if __name__ == "__main__":
    main()
