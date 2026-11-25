"""Measured version of confirm.py: is there a source AT the predicted place, compared with what chance gives?

Bodies: the 4 unmatched candidates and every known body vetted in the same fields (blind controls, same procedure, no
truth used). Centroids within 1" of the program's own track (a nearby star must not pull them), orbit family by distance,
then on each target night a comoving stack over a +-30" box. Reported per night: S/N peak within R of the prediction, the
fraction of the box where a circle of radius R would hold a peak at least as high (chance, stars included), S/N at that
place in each exposure, in a static stack of the same exposures, and on the other nights (a star shows up there).
Usage: python -I confirm2.py OUTDIR
"""
import json, math, os, sys
import numpy as np
H = os.path.expanduser("$ALE_HOME/planet-search/rsi")
sys.path.insert(0, H + "/candidates")
import confirm as C
G = C.G
from astropy.io import fits
from scipy.ndimage import maximum_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ARC, PS = C.ARC, C.PS
HALF = 30.0


def disk(r):
    n = int(r / PS) + 1; y, x = np.mgrid[-n:n + 1, -n:n + 1]
    return np.hypot(x, y) * PS <= r


def small(ex_list, half=6.0, r=1.0):
    """S/N within r of the centre of a stack of (Exposure, ra, dec), or None when nothing lands on a CCD."""
    ax, N, D, used = C.comoving(ex_list, half)
    if not used or (D > 0).sum() < 200 or not (D[len(ax) // 2 - 3:len(ax) // 2 + 4, len(ax) // 2 - 3:len(ax) // 2 + 4] > 0).any():
        return None
    return round(C.peak(ax, C.snr_map(N, D), r)[0], 1)


def readable(ps):
    """Files kept despite a bad md5 can be really damaged: every CCD must read, or the exposure is left out."""
    try:
        e = G.Exposure(*ps)
        for c in e.ccds:
            e.data(c[0]); e.cache.clear()
        e.close()
        return True
    except Exception as ex:
        print("left out, damaged:", os.path.basename(ps[0]), repr(ex)[:80], flush=True)
        return False


def main():
    out = sys.argv[1]
    nf = os.path.join(H, "data/confirm/NOMD5.txt")
    nomd5 = {l.split()[2] for l in open(nf)} if os.path.exists(nf) else set()
    V = json.load(open(H + "/candidates/vet.json"))
    new = [x for x in V if not x["known"] and x["smax"] is not None and x["smax"] < 3 and x["drop1"] >= 5 and x["p"] >= .9]
    ctrl = [x for x in V if x["known"] and x["f"] in ("F3", "F9") and x.get("nexp", 0) >= 3]
    Xs = sorted(new, key=lambda x: -x["p"]) + sorted(ctrl, key=lambda x: (x["f"], -x["p"]))
    print(len(new), "candidates,", len(ctrl), "controls", flush=True)
    F = json.load(open(H + "/candidates/files_confirm.json"))["exposures"]
    Tg = []
    for b, e in F.items():
        ps = [os.path.join(H, "data/confirm", [f["name"] for f in e["files"] if f"_oo{c}_" in f["name"]][0]) for c in "iwd"]
        if all(os.path.exists(p) for p in ps):
            if any(os.path.basename(p) in nomd5 for p in ps) and not readable(ps):
                continue
            h = fits.getheader(ps[0], 0)
            Tg.append(dict(f=e["f"], cal=e["cal"], band=e["band"], t=e["t"], paths=ps, m=float(h["MJD-OBS"]) + float(h["EXPTIME"]) / 2 / 86400))
    print(len(Tg), "target exposures on disk of", len(F), flush=True)
    cache = {}
    def tex(t):
        k = t["paths"][0]
        if k not in cache:
            cache[k] = G.Exposure(*t["paths"])
        return cache[k]
    res, gallery = [], []
    for x in Xs:
        P = json.load(open(f"{H}/private/scenarios/{x['sid']}.json")); E = {e["image"]: e for e in P["all_exposures"]}
        d = os.path.join(H, "data", str(P["window"]))
        m_, ra_, de_, _ = x["track"][0]; ob_ = C.observer(m_); c, bd = None, 1e9
        for c_ in json.load(open(f"{H}/runs/r1/gens/gen032/ls6_runs/{x['sid']}/candidates.json"))["candidates"]:
            if round(c_["p"], 3) != x["p"]:
                continue
            p_ = G.positions(dict(state=np.array(c_["state"], float), epoch=float(c_["epoch_mjd"])), [m_], [ob_])
            if p_ is not None and math.hypot((p_[0][0] - ra_) * math.cos(math.radians(de_)), p_[1][0] - de_) < bd:
                bd, c = math.hypot((p_[0][0] - ra_) * math.cos(math.radians(de_)), p_[1][0] - de_), c_
        if c is None:
            continue
        S0 = np.array(c["state"], float); ep = float(c["epoch_mjd"])
        tag = f"{x['f']} p={x['p']}" + (" known" if x["known"] else " CANDIDATE")
        exs = {}
        nights = {}
        for m, ra, de, img in x["track"]:
            nights.setdefault(int(np.floor(m - .5)), []).append((m, ra, de, img))
        obsv = []
        for n, tr in sorted(nights.items()):
            items = []
            for t in tr:
                if t[3] not in exs:
                    e = E[t[3]]; exs[t[3]] = G.Exposure(os.path.join(d, e["image"]), os.path.join(d, e["weight"]), os.path.join(d, e["mask"]))
                items.append((exs[t[3]], t[1], t[2]))
            ax, N, D, used = C.comoving(items, 6.0)
            if not used:
                continue
            s, dx, dy = C.peak(ax, C.snr_map(N, D), 1.0)
            if s >= 4:
                m = np.mean([t[0] for t in tr]); ra = np.mean([t[1] for t in tr]); de = np.mean([t[2] for t in tr])
                obsv.append((m, ra + dx * ARC / math.cos(math.radians(de)), de + dy * ARC, C.observer(m), s))
        for o in exs.values():
            o.close()
        if len(obsv) < 3:
            print("==", tag, "under 3 nights of centroids", flush=True); continue
        fam = {}
        for r in (26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 75, 80, 90):
            try:
                fam[r] = C.fit_r(obsv, S0, ep, r)
            except Exception:
                pass
        best = min(v[1] for v in fam.values())
        fam = {r: v for r, v in fam.items() if v[1] <= max(.5, 2 * best)}
        print("==", tag, f"nights {len(obsv)} rms {best:.2f} r {min(fam)}-{max(fam)}", flush=True)
        rows, by = [], {}
        for t in Tg:
            if t["f"] == x["f"]:
                by.setdefault(t["cal"], []).append(t)
        for cal, L in sorted(by.items()):
            its = []
            for t in L:
                ob = C.observer(t["m"])
                ps = [G.positions(dict(state=v[0], epoch=ep), [t["m"]], [ob]) for v in fam.values()]
                ps = [(float(p[0][0]), float(p[1][0])) for p in ps if p is not None]
                if not ps:
                    continue
                ra0 = float(np.median([p[0] for p in ps])); de0 = float(np.median([p[1] for p in ps]))
                sp = max(math.hypot((p[0] - ra0) * math.cos(math.radians(de0)), p[1] - de0) for p in ps) * 3600
                if tex(t).locate(ra0, de0) is not None:
                    its.append((t, ra0, de0, sp))
            if not its:
                continue
            spread = max(i[3] for i in its); R = max(1.5, spread + 1.0)
            if spread > 20:                       # the orbit can't place it yet; a box this wide says nothing and ran the Mac out of memory
                print(f"   {cal}: spread {spread:.0f}\", skipped until the orbit is refined", flush=True)
                continue
            half = max(HALF, 2 * R + 10)
            ax, N, D, used = C.comoving([(tex(i[0]), i[1], i[2]) for i in its], half)
            if not used:
                continue
            z = C.snr_map(N, D); s, dx, dy = C.peak(ax, z, R)
            X, Y = np.meshgrid(ax, ax); fp = disk(R); mf = maximum_filter(np.where(D > 0, z, -9), footprint=fp)
            ok = (D > 0) & (np.hypot(X, Y) > 2 * R)
            chance = float((mf[ok] >= s).mean()) if ok.any() else None
            cd = math.cos(math.radians(its[0][2]))
            per = [small([(tex(i[0]), i[1] + dx * ARC / cd, i[2] + dy * ARC)]) for i in its]
            mid = its[len(its) // 2]; rap, dep = mid[1] + dx * ARC / cd, mid[2] + dy * ARC
            motion = math.hypot((its[-1][1] - its[0][1]) * cd, its[-1][2] - its[0][2]) * 3600
            stat = small([(tex(i[0]), rap, dep) for i in its])
            other = []
            for cal2, L2 in sorted(by.items()):
                if cal2 == cal:
                    continue
                on = [(tex(t), rap, dep) for t in L2 if tex(t).locate(rap, dep) is not None]
                if on:
                    v = small(on)
                    if v is not None:
                        other.append(v)
            row = dict(cal=cal, band=sorted(set(i[0]["band"] for i in its)), n=len(its), t=its[0][0]["t"], spread=round(spread, 1), R=round(R, 1), snr=round(s, 1), dx=round(dx, 2), dy=round(dy, 2),
                       chance=None if chance is None else round(chance, 4), per=per, static=stat, motion=round(motion, 1), other=other, ra=rap, dec=dep, mjd=float(np.mean([i[0]["m"] for i in its])))
            rows.append(row)
            print(f"   {cal} {'/'.join(row['band'])} {row['n']}x{row['t']:.0f}s R={row['R']}\": S/N {row['snr']} at ({dx:+.1f},{dy:+.1f}) chance {row['chance']} per-exp {per} moved {row['motion']}\" static {stat} other-nights {other}", flush=True)
            k = int(8 / PS); c0 = len(ax) // 2
            gallery.append((tag, cal, z[c0 - k:c0 + k + 1, c0 - k:c0 + k + 1], R, s))
            for o in cache.values():
                o.cache.clear()
        res.append(dict(f=x["f"], p=x["p"], known=x["known"], sid=x["sid"], rms=best, rs=sorted(fam), epoch=ep, obs=[(o[0], o[1], o[2], o[4]) for o in obsv], nights=rows))
        json.dump(res, open(os.path.join(out, "confirm2.json"), "w"), indent=0)
        for o in cache.values():             # CCD data adds up: the run was killed with no error when these were kept for every body
            o.close()
        cache.clear()
    tags = list(dict.fromkeys(g[0] for g in gallery)); cals = sorted(set(g[1] for g in gallery))
    fig, axs = plt.subplots(len(tags), len(cals), figsize=(1.5 * len(cals), 1.6 * len(tags)), squeeze=False)
    for a in axs.ravel():
        a.axis("off")
    for tag, cal, z, R, s in gallery:
        a = axs[tags.index(tag)][cals.index(cal)]; a.axis("on")
        a.imshow(z, origin="lower", cmap="gray_r", vmin=-2, vmax=8, extent=(-8, 8, -8, 8)); a.add_patch(plt.Circle((0, 0), R, fill=False, ec="r", lw=.5))
        a.set_title(f"{cal[2:]} {s:.1f}", fontsize=6); a.set_xticks([]); a.set_yticks([])
        a.set_ylabel(tag.replace(" CANDIDATE", " *"), fontsize=5)
    plt.tight_layout(); plt.savefig(os.path.join(out, "confirm2.png"), dpi=150); plt.close(fig)
    print("DONE")


if __name__ == "__main__":
    main()
