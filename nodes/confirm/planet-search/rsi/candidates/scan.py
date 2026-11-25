"""Star-free search of every other image on disk. For each body the fitted orbits (one per distance) trace a curve on the sky on each night;
small comoving stacks are tiled along that curve, every peak's sky place is checked on other nights (a star shows there), and what survives is
listed with the distance it implies. Exposures the program saw are left out. Known bodies run blind too; only afterwards their catalogued place
(SkyBoT) is compared with the peaks. Usage: python -I scan.py OUTDIR "F3:0.997:0,F3:0.995:1,..."
"""
import json, math, os, sys
import numpy as np
H = os.path.expanduser("$ALE_HOME/planet-search/rsi")
sys.path.insert(0, H + "/candidates")
import confirm as C, confirm2 as C2, refit as RF
G = C.G
from astropy.io import fits
from astropy.coordinates import Angle
import astropy.units as u
from scipy.ndimage import maximum_filter
ARC, PS = C.ARC, C.PS
HALF, STEP, THR = 12.0, 16.0, 5.0


def exposures():
    out, seen = [], set()
    nf = H + "/data/confirm/NOMD5.txt"; bad = {l.split()[2] for l in open(nf)} if os.path.exists(nf) else set()
    for name in ("files_confirm.json", "files_confirm2.json"):
        for b, e in json.load(open(f"{H}/candidates/{name}"))["exposures"].items():
            if b in seen:
                continue
            ps = [os.path.join(H, "data/confirm", [f["name"] for f in e["files"] if f"_oo{c}_" in f["name"]][0]) for c in "iwd"]
            if not all(os.path.exists(p) for p in ps) or any(os.path.basename(p) in bad for p in ps):
                continue
            h = fits.getheader(ps[0], 0); seen.add(b)
            ra, de = e.get("ra"), e.get("dec")
            if ra is None:
                ra = Angle(h["RA"], u.hourangle).deg; de = Angle(h["DEC"], u.deg).deg
            out.append(dict(base=b, cal=e["cal"], band=e["band"], t=float(h["EXPTIME"]), ra=float(ra), dec=float(de), paths=ps, m=float(h["MJD-OBS"]) + float(h["EXPTIME"]) / 2 / 86400))
    return out


def start(x, v):
    m_, ra_, de_, _ = v["track"][0]; ob_ = C.observer(m_); c, bd = None, 1e9
    for c_ in json.load(open(f"{H}/runs/r1/gens/gen032/ls6_runs/{x['sid']}/candidates.json"))["candidates"]:
        if round(c_["p"], 3) != x["p"]:
            continue
        p_ = G.positions(dict(state=np.array(c_["state"], float), epoch=float(c_["epoch_mjd"])), [m_], [ob_])
        if p_ is not None and math.hypot((p_[0][0] - ra_) * math.cos(math.radians(de_)), p_[1][0] - de_) < bd:
            bd, c = math.hypot((p_[0][0] - ra_) * math.cos(math.radians(de_)), p_[1][0] - de_), c_
    return np.array(c["state"], float), float(c["epoch_mjd"])


def main():
    out, want = sys.argv[1], sys.argv[2].split(",")
    R2 = json.load(open(H + "/candidates/confirm2.json")); Rf = json.load(open(H + "/candidates/refit.json")); V = json.load(open(H + "/candidates/vet.json"))
    Ex = exposures(); print(len(Ex), "exposures on disk", flush=True)
    for w in want:
        f, p, kn = w.split(":"); p = float(p); kn = kn == "1"
        x = next(q for q in R2 if q["f"] == f and q["p"] == p and q["known"] == kn)
        v = next(q for q in V if q["f"] == f and q["p"] == p and q["known"] == kn and q["sid"] == x["sid"])
        fit = next((q for q in Rf if q["f"] == f and q["p"] == p and q["known"] == kn), None)
        if fit:
            ep = fit["epoch"]; fam = {int(r): np.array(s) for r, s in fit["states"].items()}; arc_end = max(o[0] for o in fit["obs"])
        else:
            S0, ep = start(x, v); obs = [(o[0], o[1], o[2], C.observer(o[0]), o[3]) for o in x["obs"]]; f1 = {}
            for r in range(26, 92, 2):
                try:
                    f1[r] = C.fit_r(obs, S0, ep, r)
                except Exception:
                    pass
            best = min(q[1] for q in f1.values()); fam = {r: q[0] for r, q in f1.items() if q[1] <= max(.35, 1.5 * best)}; arc_end = max(o[0] for o in obs)
        rs = sorted(fam)
        P = json.load(open(f"{H}/private/scenarios/{x['sid']}.json")); used = {e["image"].split("_ooi_")[0] for e in P["all_exposures"]}
        tag = f"{f} p={p} {'known' if kn else 'CANDIDATE'}"
        print(f"== {tag} r {rs[0]}-{rs[-1]}", flush=True)
        L = [t for t in Ex if t["base"] not in used]
        for t in L:
            ob = C.observer(t["m"]); c = []
            for r in rs:
                q = G.positions(dict(state=fam[r], epoch=ep), [t["m"]], [ob])
                c.append((float(q[0][0]), float(q[1][0])) if q is not None else (np.nan, np.nan))
            t["curve"] = np.array(c); a = t["curve"][len(rs) // 2]
            t["near"] = np.isfinite(a[0]) and math.hypot((t["ra"] - a[0]) * math.cos(math.radians(a[1])), t["dec"] - a[1]) < 1.4 + np.hypot((t["curve"][-1, 0] - t["curve"][0, 0]) * math.cos(math.radians(a[1])), t["curve"][-1, 1] - t["curve"][0, 1])
        by = {}
        for t in L:
            if t["near"]:
                by.setdefault(t["cal"], []).append(t)
        refs = sorted([t for t in L if t["near"]], key=lambda t: -t["t"])
        rows = []
        for cal, nl in sorted(by.items()):
            exs = [(t, G.Exposure(*t["paths"])) for t in nl]
            t0 = nl[0]; cc = t0["curve"]; cd = math.cos(math.radians(cc[0, 1])); seg = np.hypot(np.diff(cc[:, 0]) * cd, np.diff(cc[:, 1])) * 3600; Lc = float(seg.sum())
            if Lc > 3000:
                print(f"   {cal}: curve {Lc:.0f}\", too long for now", flush=True)
                for _, e in exs:
                    e.close()
                continue
            nt = 1 if Lc < 4 else int(math.ceil(Lc / STEP)) + 1; ks = np.linspace(0, len(rs) - 1, nt) if nt > 1 else [(len(rs) - 1) / 2]
            pk, tiles = [], 0
            for k in ks:
                i = min(int(k), len(rs) - 2) if len(rs) > 1 else 0; w_ = k - i
                items = []
                for t, e in exs:
                    a = t["curve"][i] if len(rs) == 1 else t["curve"][i] * (1 - w_) + t["curve"][i + 1] * w_
                    if e.locate(a[0], a[1]) is not None or any(e.locate(a[0] + sx * HALF * ARC / cd, a[1] + sy * HALF * ARC) for sx in (-1, 1) for sy in (-1, 1)):
                        items.append((e, float(a[0]), float(a[1]), t))
                if not items:
                    continue
                ax, N, D, used_ = C.comoving([(q[0], q[1], q[2]) for q in items], HALF)
                if not used_:
                    continue
                tiles += 1; z = C.snr_map(N, D); mx = (z == maximum_filter(z, size=9)) & (z >= THR) & (D > 0)
                for j, i2 in zip(*np.nonzero(mx)):
                    a0, d0 = items[len(items) // 2][1], items[len(items) // 2][2]
                    pk.append(dict(s=float(z[j, i2]), dx=float(ax[i2]), dy=float(ax[j]), r=(rs[i] + (rs[i + 1] - rs[i]) * w_) if len(rs) > 1 else rs[0], items=items, ra=a0 + ax[i2] * ARC / cd, dec=d0 + ax[j] * ARC, band=None))
                for _, e in exs:
                    if len(e.cache) > 8:
                        e.cache.clear()
            pk.sort(key=lambda q: -q["s"]); keep = []
            for q in pk:
                if all(math.hypot((q["ra"] - o["ra"]) * cd, q["dec"] - o["dec"]) * 3600 > 2 for o in keep):
                    keep.append(q)
            ok = []
            for q in keep[:14]:
                per = [C2.small([(i_[0], i_[1] + q["dx"] * ARC / cd, i_[2] + q["dy"] * ARC)]) for i_ in q["items"]]
                oth, seen_n, opened = [], set(), []
                for t in refs:
                    if abs(t["m"] - q["items"][0][3]["m"]) < 5 or t["cal"] in seen_n or len(oth) >= 5:
                        continue
                    e2 = G.Exposure(*t["paths"]); opened.append(e2)
                    if e2.locate(q["ra"], q["dec"]) is not None:
                        s_ = C2.small([(e2, q["ra"], q["dec"])])
                        if s_ is not None:
                            oth.append(s_); seen_n.add(t["cal"])
                    e2.close()
                good = [s_ for s_ in per if s_ is not None]
                passed = len(oth) >= 2 and max(oth) < 3.5 and (len(good) < 2 or sum(s_ >= 2.5 for s_ in good) >= max(2, len(good) - 1))
                if passed:
                    ok.append(dict(cal=cal, n=len(q["items"]), bands=sorted(set(i_[3]["band"] for i_ in q["items"])), t=q["items"][0][3]["t"], mjd=float(np.mean([i_[3]["m"] for i_ in q["items"]])), snr=round(q["s"], 1), r=round(float(q["r"]), 1), ra=q["ra"], dec=q["dec"], per=per, other=oth))
            band = "/".join(sorted(set(t["band"] for t in nl)))
            print(f"   {cal} {band} {len(nl)} exp, curve {Lc:.0f}\", {tiles} tiles, {len(pk)} peaks, {len(ok)} pass" + ("" if ok or not tiles else ""), flush=True)
            cat = None
            if kn and tiles:
                m0 = float(np.mean([t["m"] for t in nl])); a = t0["curve"][len(rs) // 2]; sk = RF.skybot(m0, float(a[0]), float(a[1]), 0.5 if Lc > 300 else 0.1)
                if sk:
                    b = min(sk, key=lambda q: math.hypot((q["ra"] - a[0]) * cd, q["dec"] - a[1])); cat = (b["name"], b["ra"], b["dec"])
                    print(f"      catalogue {b['name']} V {b['V']}", flush=True)
            for o in ok[:6]:
                sep = f" {math.hypot((o['ra'] - cat[1]) * cd, o['dec'] - cat[2]) * 3600:.1f}\" from catalogue" if cat else ""
                print(f"      S/N {o['snr']} r~{o['r']} per-exp {o['per']} other {o['other']}{sep}", flush=True)
            rows += ok
            for _, e in exs:
                e.close()
        json.dump(rows, open(os.path.join(out, f"scan_{f}_{p}_{int(kn)}.json"), "w"), indent=0)
    print("DONE")


if __name__ == "__main__":
    main()
