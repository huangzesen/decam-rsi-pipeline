"""Second pass after confirm2.py. (1) Are the known bodies really at the places predicted? SkyBoT gives each control's catalogued
place on each target night; S/N is measured there too. (2) Refit on the detections from confirm2 (a longer arc, a narrower family
of distances) and search the nights the first family could not place, known bodies alongside.
Usage: python -I refit.py OUTDIR
"""
import json, math, os, subprocess, sys
import numpy as np
H = os.path.expanduser("$ALE_HOME/planet-search/rsi")
sys.path.insert(0, H + "/candidates")
import confirm as C
import confirm2 as C2
G = C.G
from astropy.io import fits
from scipy.ndimage import maximum_filter
ARC, PS = C.ARC, C.PS
USE = {("F3", 0.997, False): ["2014-04-29", "2014-04-30", "2014-05-03"], ("F3", 0.917, False): ["2014-04-30", "2014-05-03", "2014-05-04"],
       ("F9", 0.995, False): ["2018-01-27"], ("F3", 0.995, True): ["2014-04-29", "2014-04-30"], ("F3", 0.967, True): ["2014-04-30", "2014-05-03"]}


def skybot(m, ra, de, rd=0.03):
    url = f"https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?-ep={m + 2400000.5:.6f}&-ra={ra:.6f}&-dec={de:.6f}&-rd={rd}&-mime=text&-output=all&-loc=W84&-objFilter=111&-from=planet-search"
    out = subprocess.run(["curl", "-s", "--max-time", "60", url], capture_output=True, text=True).stdout.splitlines()
    hdr = next((l for l in out if l.startswith("#") and "|" in l), None)
    if not hdr:
        return []
    cols = [c.strip().lstrip("#").strip() for c in hdr.split("|")]
    rows = [dict(zip(cols, [y.strip() for y in l.split("|")])) for l in out if "|" in l and not l.startswith("#")]
    far = []
    for r in rows:
        if float(r["Dh(ua)"]) > 20:
            h = [float(v) for v in r["RA(h)"].split()]; d = r["DE(deg)"]; sgn = -1 if d.startswith("-") else 1; dd = [abs(float(v)) for v in d.split()]
            far.append(dict(name=r["Name"], ra=15 * (h[0] + h[1] / 60 + h[2] / 3600), dec=sgn * (dd[0] + dd[1] / 60 + dd[2] / 3600), V=r["Mv"], err=r["Err(arcsec)"]))
    return far


def main():
    out = sys.argv[1]
    R = json.load(open(H + "/candidates/confirm2.json")); V = json.load(open(H + "/candidates/vet.json"))
    F = json.load(open(H + "/candidates/files_confirm.json"))["exposures"]
    nf = os.path.join(H, "data/confirm/NOMD5.txt"); bad = {l.split()[2] for l in open(nf)} if os.path.exists(nf) else set()
    Tg = []
    for b, e in F.items():
        ps = [os.path.join(H, "data/confirm", [f["name"] for f in e["files"] if f"_oo{c}_" in f["name"]][0]) for c in "iwd"]
        if all(os.path.exists(p) for p in ps) and not any(os.path.basename(p) in bad for p in ps):
            h = fits.getheader(ps[0], 0)
            Tg.append(dict(f=e["f"], cal=e["cal"], band=e["band"], t=e["t"], paths=ps, m=float(h["MJD-OBS"]) + float(h["EXPTIME"]) / 2 / 86400))
    cache = {}
    def tex(t):
        k = t["paths"][0]
        if k not in cache:
            cache[k] = G.Exposure(*t["paths"])
        return cache[k]
    def family(fam, ep, m):
        ob = C.observer(m)
        ps = [G.positions(dict(state=v[0], epoch=ep), [m], [ob]) for v in fam.values()]
        ps = [(float(p[0][0]), float(p[1][0])) for p in ps if p is not None]
        ra0 = float(np.median([p[0] for p in ps])); de0 = float(np.median([p[1] for p in ps]))
        return ra0, de0, max(math.hypot((p[0] - ra0) * math.cos(math.radians(de0)), p[1] - de0) for p in ps) * 3600
    res = []
    print("(1) known bodies at their catalogued places", flush=True)
    for x in [y for y in R if y["known"]]:
        print(f"== {x['f']} p={x['p']} fit rms {x['rms']:.2f}", flush=True)
        for n in x["nights"]:
            L = [t for t in Tg if t["f"] == x["f"] and t["cal"] == n["cal"]]
            far = skybot(n["mjd"], n["ra"], n["dec"])
            if not far:
                print(f"   {n['cal']}: no distant body within 108\" of our prediction", flush=True); continue
            cd = math.cos(math.radians(n["dec"])); far.sort(key=lambda b: math.hypot((b["ra"] - n["ra"]) * cd, b["dec"] - n["dec"]))
            b = far[0]; sep = math.hypot((b["ra"] - n["ra"]) * cd, b["dec"] - n["dec"]) * 3600
            # one exposure time, sky motion ignored within the night: the catalogue place moves with the stack below
            its = []
            for t in L:
                b2 = skybot(t["m"], b["ra"], b["dec"], 0.01); b2 = [q for q in b2 if q["name"] == b["name"]]
                if b2 and tex(t).locate(b2[0]["ra"], b2[0]["dec"]) is not None:
                    its.append((tex(t), b2[0]["ra"], b2[0]["dec"]))
            s = C2.small(its, 6.0, 1.5) if its else None
            print(f"   {n['cal']}: {b['name']} V {b['V']} err {b['err']}\": {sep:.1f}\" from our detection spot; S/N {s} at the catalogued place ({len(its)} exp); confirm2 had {n['snr']}", flush=True)
            for o in cache.values():
                o.cache.clear()
        for o in cache.values():
            o.close()
        cache.clear()
    print("(2) refit on the detections, then search", flush=True)
    for x in R:
        key = (x["f"], x["p"], x["known"])
        if key not in USE:
            continue
        v = next(y for y in V if y["f"] == x["f"] and y["p"] == x["p"] and y["known"] == x["known"] and y["sid"] == x["sid"] and x["known"] == y["known"])
        m_, ra_, de_, _ = v["track"][0]; ob_ = C.observer(m_); c, bd = None, 1e9
        for c_ in json.load(open(f"{H}/runs/r1/gens/gen032/ls6_runs/{x['sid']}/candidates.json"))["candidates"]:
            if round(c_["p"], 3) != x["p"]:
                continue
            p_ = G.positions(dict(state=np.array(c_["state"], float), epoch=float(c_["epoch_mjd"])), [m_], [ob_])
            if p_ is not None and math.hypot((p_[0][0] - ra_) * math.cos(math.radians(de_)), p_[1][0] - de_) < bd:
                bd, c = math.hypot((p_[0][0] - ra_) * math.cos(math.radians(de_)), p_[1][0] - de_), c_
        S0 = np.array(c["state"], float); ep = float(c["epoch_mjd"])
        obs0 = [(o[0], o[1], o[2], C.observer(o[0]), o[3]) for o in x["obs"]]
        fam0 = {r: C.fit_r(obs0, S0, ep, r) for r in x["rs"]}
        add = []
        for n in x["nights"]:
            if n["cal"] in USE[key]:
                ra0, de0, _ = family(fam0, ep, n["mjd"])
                add.append((n["mjd"], ra0 + n["dx"] * ARC / math.cos(math.radians(de0)), de0 + n["dy"] * ARC, C.observer(n["mjd"]), n["snr"]))
        obs1 = obs0 + add; fam1 = {}
        for r in range(26, 92, 2):
            try:
                fam1[r] = C.fit_r(obs1, S0, ep, r)
            except Exception:
                pass
        best = min(w[1] for w in fam1.values()); fam1 = {r: w for r, w in fam1.items() if w[1] <= max(.35, 1.5 * best)}
        if not fam1:
            print("==", key, "no fit", flush=True); continue
        arc = max(o[0] for o in obs1) - min(o[0] for o in obs1)
        print(f"== {x['f']} p={x['p']} {'known' if x['known'] else 'CANDIDATE'}: arc {arc:.1f} d, rms {best:.2f}, r {min(fam1)}-{max(fam1)} (was {min(x['rs'])}-{max(x['rs'])}); residuals by night {[round(float(math.hypot(*(lambda p: ((p[0][0]-o[1])*math.cos(math.radians(o[2]))*3600,(p[1][0]-o[2])*3600))(G.positions(dict(state=fam1[min(fam1, key=lambda q: fam1[q][1])][0], epoch=ep), [o[0]], [o[3]])))),2) for o in obs1]}", flush=True)
        by = {}
        for t in Tg:
            if t["f"] == x["f"]:
                by.setdefault(t["cal"], []).append(t)
        rows = []
        for cal, L in sorted(by.items()):
            if cal in USE[key]:
                continue
            its = []
            for t in L:
                ra0, de0, sp = family(fam1, ep, t["m"])
                if tex(t).locate(ra0, de0) is not None:
                    its.append((t, ra0, de0, sp))
            if not its:
                print(f"   {cal}: not on a CCD", flush=True); continue
            spread = max(i[3] for i in its); Rr = max(1.5, spread + 1.0)
            if spread > 20:
                print(f"   {cal}: spread {spread:.0f}\", still too wide", flush=True); continue
            half = max(30.0, 2 * Rr + 10)
            ax, N, D, used = C.comoving([(tex(i[0]), i[1], i[2]) for i in its], half)
            if not used:
                continue
            z = C.snr_map(N, D); s, dx, dy = C.peak(ax, z, Rr)
            X, Y = np.meshgrid(ax, ax); mf = maximum_filter(np.where(D > 0, z, -9), footprint=C2.disk(Rr)); ok = (D > 0) & (np.hypot(X, Y) > 2 * Rr)
            chance = float((mf[ok] >= s).mean()) if ok.any() else None
            cd = math.cos(math.radians(its[0][2])); per = [C2.small([(tex(i[0]), i[1] + dx * ARC / cd, i[2] + dy * ARC)]) for i in its]
            mid = its[len(its) // 2]; rap, dep = mid[1] + dx * ARC / cd, mid[2] + dy * ARC
            motion = math.hypot((its[-1][1] - its[0][1]) * cd, its[-1][2] - its[0][2]) * 3600
            stat = C2.small([(tex(i[0]), rap, dep) for i in its])
            other = [v_ for cal2, L2 in sorted(by.items()) if cal2 != cal for v_ in [C2.small([(tex(t), rap, dep) for t in L2 if tex(t).locate(rap, dep) is not None]) if any(tex(t).locate(rap, dep) is not None for t in L2) else None] if v_ is not None]
            row = dict(cal=cal, band=sorted(set(i[0]["band"] for i in its)), n=len(its), spread=round(spread, 1), R=round(Rr, 1), snr=round(s, 1), dx=round(dx, 2), dy=round(dy, 2), chance=None if chance is None else round(chance, 4), per=per, static=stat, motion=round(motion, 1), other=other, mjd=float(np.mean([i[0]["m"] for i in its])))
            rows.append(row)
            print(f"   {cal} {'/'.join(row['band'])} {row['n']} exp R={row['R']}\": S/N {row['snr']} at ({dx:+.1f},{dy:+.1f}) chance {row['chance']} per-exp {per} moved {row['motion']}\" static {stat} other {other}", flush=True)
            for o in cache.values():
                o.cache.clear()
        res.append(dict(f=x["f"], p=x["p"], known=x["known"], arc=arc, rms=best, rs=sorted(fam1), epoch=ep, states={str(r): list(map(float, w[0])) for r, w in fam1.items()}, obs=[(o[0], o[1], o[2], o[4]) for o in obs1], nights=rows))
        json.dump(res, open(os.path.join(out, "refit.json"), "w"), indent=0)
        for o in cache.values():
            o.close()
        cache.clear()
    print("DONE")


if __name__ == "__main__":
    main()
