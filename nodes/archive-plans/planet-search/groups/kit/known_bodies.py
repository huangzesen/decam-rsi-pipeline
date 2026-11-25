"""Every known TNO that crosses a group's exposures, from the exposure metadata alone (no image needed).

For each group:
  1. candidates: every TNO of the JPL SBDB snapshot whose two-body position lies within --radius deg of an exposure
     centre on that night;
  2. JPL Horizons astrometric RA/Dec, V, r, delta and the 3-sigma position uncertainty at every such exposure's
     mid-time, observer W84 (DECam); one query per body and 75 epochs (the service refuses longer lists), cached in groups/_private/horizons/;
  3. per window, every body within --fov deg of a pointing: its positions, the S/N predicted from the depth model
     (NSC DR2 depths, TNO colours, as in linker/footprint/groups.py), and a predicted tier with the task's rules:
     1 = >= 3 nights with one exposure at S/N >= 5; 2 = >= 3 nights whose night stack reaches 5; 3 = the stack of all
     exposures reaches 5 on >= 3 nights; 0 = not findable; None = orbit too uncertain (3-sigma > --max-sigma arcsec).
The predicted tier ignores CCD gaps and real image quality. When a window's images are on disk the group job measures
the real S/N at these positions (run_group.py), and score_group.py then uses the measured tiers.

Output: groups/_private/<id>/known.json. Workers and finder programs never read groups/_private/.

Usage: ~/anaconda3/bin/python -I groups/kit/known_bodies.py g099 g121 ...
"""
import argparse, json, os, sys, time, urllib.parse, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PS = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(PS, "linker"))
from orbits import elements_to_rv, ecl_to_eq, earth_eq, GM

D90 = {"g": 23.52, "r": 23.30, "i": 22.79, "z": 22.23, "Y": 21.30, "VR": 23.50}
VM = {"g": -0.45, "r": 0.3, "i": 0.6, "z": 0.8, "Y": 0.85, "VR": 0.15}


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def horizons(pdes, jds, cache):
    f = os.path.join(cache, pdes.replace(" ", "_").replace("/", "_") + ".json")
    have = json.load(open(f)) if os.path.exists(f) else {}
    need = [j for j in jds if f"{j:.6f}" not in have]
    cmd = f"{pdes};" if pdes.isdigit() else f"DES={pdes};"
    for k in range(0, len(need), 75):
        q = {"format": "json", "COMMAND": f"'{cmd}'", "OBJ_DATA": "NO", "MAKE_EPHEM": "YES", "EPHEM_TYPE": "OBSERVER",
             "CENTER": "'W84'", "TLIST": " ".join(f"{j:.6f}" for j in need[k:k + 75]), "TLIST_TYPE": "JD",
             "TIME_TYPE": "UT", "QUANTITIES": "'1,9,19,20,36'", "ANG_FORMAT": "DEG", "EXTRA_PREC": "YES", "CSV_FORMAT": "YES"}
        url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q)
        res = None
        for t in range(5):
            try:
                res = json.loads(urllib.request.urlopen(url, timeout=180).read()).get("result", "")
                break
            except Exception as e:
                print("  horizons retry", pdes, t, repr(e)[:80], flush=True); time.sleep(5 * (t + 1))
        i, j = (res or "").find("$$SOE"), (res or "").find("$$EOE")
        if i < 0:
            return None
        lines = res[i + 5:j].strip().splitlines()
        if len(lines) != len(need[k:k + 75]):
            return None
        fl = lambda s: float(s) if s not in ("", "n.a.") else None
        for jd, line in zip(need[k:k + 75], lines):
            c = [x.strip() for x in line.split(",")]
            have[f"{jd:.6f}"] = [fl(c[3]), fl(c[4]), fl(c[5]), fl(c[7]), fl(c[9]), fl(c[11]), fl(c[12])]
        json.dump(have, open(f, "w"))
        time.sleep(0.25)
    return [have[f"{j:.6f}"] for j in jds]


def stack(s):
    return float(np.sqrt(np.sum(np.square(np.clip(s, 0, None))))) if len(s) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--sbdb", default=os.path.join(PS, "linker", "data_real", "sbdb", "tno_20261031.json"))
    ap.add_argument("--radius", type=float, default=1.6); ap.add_argument("--fov", type=float, default=1.1)
    ap.add_argument("--max-sigma", type=float, default=1.5)
    a = ap.parse_args()
    cache = os.path.join(ROOT, "_private", "horizons")
    os.makedirs(cache, exist_ok=True)
    S = json.load(open(a.sbdb))
    rows = [dict(zip(S["fields"], r)) for r in S["data"]]
    rows = [r for r in rows if all(r[k] not in (None, "") for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H"))
            and float(r["e"]) < 1 and float(r["a"]) > 0]
    el = {k: np.array([float(r[k]) for r in rows]) for k in ("a", "e", "i", "om", "w", "ma", "epoch", "H")}
    t0, nmot = el["epoch"] - 2400000.5, np.sqrt(GM / el["a"] ** 3)
    cosr, cosf = np.cos(np.radians(a.radius)), np.cos(np.radians(a.fov))
    for gid in a.ids:
        W = json.load(open(os.path.join(ROOT, gid, "windows.json")))["windows"]
        E = {}                                   # stamp -> exposure (an exposure can sit in one window only)
        for w in W:
            for e in w["exposures"]:
                E[e["stamp"]] = dict(e, mid=e["mjd_start"] + e["exptime"] / 2 / 86400, key=w["key"])
        st = sorted(E, key=lambda s: E[s]["mid"])
        mid = np.array([E[s]["mid"] for s in st])
        ue = unit(np.array([E[s]["ra"] for s in st]), np.array([E[s]["dec"] for s in st]))
        night = np.floor(mid - 0.5).astype(int)
        cand = {}                                # body index -> exposure indices
        for n in np.unique(night):
            ix = np.flatnonzero(night == n)
            tm = float(mid[ix].mean())
            r, _ = elements_to_rv(el["a"], el["e"], np.radians(el["i"]), np.radians(el["om"]), np.radians(el["w"]),
                                  np.radians(el["ma"]) + nmot * (tm - t0))
            g = ecl_to_eq(r) - earth_eq(tm)
            g /= np.linalg.norm(g, axis=1)[:, None]
            hit = (g @ ue[ix].T) > cosr           # bodies x exposures of the night
            for b, k in zip(*np.nonzero(hit)):
                cand.setdefault(int(b), []).append(int(ix[k]))
        print(f"{gid}: {len(st)} exposures on {np.unique(night).size} nights, {len(cand)} candidate bodies", flush=True)
        out = {str(w["key"]): [] for w in W}
        nfail = 0
        for nb, (b, ks) in enumerate(sorted(cand.items())):
            eph = horizons(rows[b]["pdes"], [float(mid[k]) + 2400000.5 for k in ks], cache)
            if eph is None:
                nfail += 1
                print("  no ephemeris", rows[b]["pdes"], flush=True)
                continue
            per = {}
            for k, q in zip(ks, eph):
                if q[0] is None:
                    continue
                e = E[st[k]]
                cs = float(unit(q[0], q[1]) @ ue[k])
                if cs < cosf:
                    continue
                lim = D90[e["band"]] + VM[e["band"]] + 1.25 * np.log10(e["exptime"] / 90) - 0.3
                snr = 5 * 10 ** (0.4 * (lim - q[2])) if q[2] is not None else None
                per.setdefault(e["key"], []).append(dict(stamp=st[k], night=int(night[k]), mjd_mid=round(float(mid[k]), 6),
                                                         ra=q[0], dec=q[1], V=q[2], r=q[3], delta=q[4], s_ra=q[5], s_dec=q[6],
                                                         sep=round(float(np.degrees(np.arccos(min(1.0, cs)))), 4),
                                                         snr_pred=None if snr is None else round(float(snr), 2)))
            for key, recs in per.items():
                sig = max(max(x["s_ra"] or 99, x["s_dec"] or 99) for x in recs)
                tier = None
                if sig <= a.max_sigma:
                    nights = {}
                    for x in recs:
                        nights.setdefault(x["night"], []).append(x["snr_pred"] or 0.0)
                    n1 = sum(max(v) >= 5 for v in nights.values())
                    n2 = sum(stack(v) >= 5 for v in nights.values())
                    tot = stack([s for v in nights.values() for s in v])
                    tier = 1 if n1 >= 3 else 2 if n2 >= 3 else 3 if (tot >= 5 and len(nights) >= 3) else 0
                out[str(key)].append(dict(pdes=rows[b]["pdes"], name=rows[b]["full_name"].strip(), H=float(el["H"][b]),
                                          first_obs=rows[b]["first_obs"], max_sigma=sig, tier_pred=tier,
                                          r=recs[0]["r"], V=recs[0]["V"], recs=recs))
            if nb % 100 == 0:
                print(f"  {nb}/{len(cand)} bodies", flush=True)
        tiers = [b["tier_pred"] for v in out.values() for b in v]
        summ = dict(bodies_in_windows=len(tiers), distinct=len({b["pdes"] for v in out.values() for b in v}),
                    tier1=tiers.count(1), tier2=tiers.count(2), tier3=tiers.count(3), tier0=tiers.count(0),
                    imprecise=tiers.count(None), windows_with_findable=sum(any(b["tier_pred"] in (1, 2, 3) for b in v) for v in out.values()),
                    no_ephemeris=nfail)
        d = os.path.join(ROOT, "_private", gid)
        os.makedirs(d, exist_ok=True)
        json.dump(dict(id=gid, made=time.strftime("%Y-%m-%d %H:%M"), source=os.path.basename(a.sbdb) + " + JPL Horizons (W84)",
                       fov_deg=a.fov, max_sigma=a.max_sigma, summary=summ, windows=out), open(os.path.join(d, "known.json"), "w"))
        print(f"{gid} done: {summ}", flush=True)


if __name__ == "__main__":
    main()
