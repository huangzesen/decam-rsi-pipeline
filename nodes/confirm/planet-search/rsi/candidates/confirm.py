"""Do the strong unmatched candidates appear at their predicted places in other DECam images?

For each candidate and a known control: comoving per-night stacks on the scenario's own exposures give centroids; orbits
fitted to them at a range of distances predict the places on other nights (the spread is the search box); comoving
matched-filter S/N maps on the target nights' exposures give a peak; forced photometry at the peak's sky position on all
other nights tests for a static source. No knowledge of the answer enters: the target images were never used.
Usage: python -I confirm.py OUTDIR [--fit-only]
"""
import glob, json, math, os, sys, warnings, logging
import numpy as np
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from astropy.utils.exceptions import AstropyWarning; warnings.simplefilter("ignore", AstropyWarning)
H = os.path.expanduser("$ALE_HOME/planet-search/rsi")
sys.path.insert(0, H + "/bin"); sys.path.insert(0, H + "/harness")
import grade as G
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation, get_body_barycentric, solar_system_ephemeris
from scipy.ndimage import convolve
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SITE = EarthLocation(lat=-30.16606 * u.deg, lon=-70.81489 * u.deg, height=2215 * u.m)
ARC = 1 / 3600.; PS = 0.263


def observer(mjd):
    t = Time(mjd, format="mjd", scale="utc")
    with solar_system_ephemeris.set("builtin"):
        e = get_body_barycentric("earth", t).xyz.to(u.au).value; s = get_body_barycentric("sun", t).xyz.to(u.au).value
    return e - s + SITE.get_gcrs_posvel(t)[0].xyz.to(u.au).value


def comoving(items, half):
    """items: (Exposure, ra, dec) of one night. Returns offsets grid and summed num, den over a +-half arcsec comoving box."""
    n = int(half / PS); ax = np.arange(-n, n + 1) * PS
    N = np.zeros((len(ax), len(ax))); D = np.zeros_like(N); used = 0
    for ex, ra, de in items:
        cd = math.cos(math.radians(de)); ks = set()
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                l = ex.locate(ra + sx * half * ARC / cd, de + sy * half * ARC)
                if l: ks.add(l[0])
        for k in ks:
            cc = [c for c in ex.ccds if c[0] == k][0]
            w = cc[1]; fw = cc[4] or 4.0
            DX, DY = np.meshgrid(ax, ax)
            xx, yy = w.all_world2pix(ra + DX.ravel() * ARC / cd, de + DY.ravel() * ARC, 0)
            xx = np.rint(xx).astype(int).reshape(DX.shape); yy = np.rint(yy).astype(int).reshape(DX.shape)
            x0, x1 = max(0, xx.min() - 25), min(cc[2], xx.max() + 26); y0, y1 = max(0, yy.min() - 25), min(cc[3], yy.max() + 26)
            if x1 - x0 < 30 or y1 - y0 < 30:
                continue
            im, wt, ms = ex.data(k)
            a = im[y0:y1, x0:x1].astype(float); wg = np.where(ms[y0:y1, x0:x1] == 0, np.clip(wt[y0:y1, x0:x1].astype(float), 0, None), 0)
            if (wg > 0).sum() < 100:
                continue
            a = a - np.median(a[wg > 0]); sd = 1.4826 * np.median(np.abs(a[wg > 0]))
            wg = np.where(np.abs(a) > 30 * sd, 0, wg)                       # bright stars out of the sum
            s = fw / 2.355; r = int(3 * s) + 1; yg, xg = np.mgrid[-r:r + 1, -r:r + 1]
            g = np.exp(-(xg ** 2 + yg ** 2) / (2 * s * s)); g /= g.sum()
            num = convolve(wg * a, g, mode="constant"); den = convolve(wg, g * g, mode="constant")
            ok = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
            N[ok] += num[yy[ok] - y0, xx[ok] - x0]; D[ok] += den[yy[ok] - y0, xx[ok] - x0]; used += 1
    return ax, N, D, used


def snr_map(N, D):
    z = np.where(D > 0, N / np.sqrt(np.where(D > 0, D, 1)), 0.0)
    v = z[D > 0]
    if v.size > 200:
        z = (z - np.median(v)) / max(1.4826 * np.median(np.abs(v - np.median(v))), .5)
    return z


def peak(ax, z, r=None):
    m = z.copy()
    if r is not None:
        X, Y = np.meshgrid(ax, ax); m = np.where(np.hypot(X, Y) <= r, m, -9)
    j, i = np.unravel_index(np.argmax(m), m.shape)
    x0 = max(i - 5, 0); y0 = max(j - 5, 0); sub = np.clip(z[y0:j + 6, x0:i + 6], 0, None)
    X, Y = np.meshgrid(ax[x0:i + 6], ax[y0:j + 6]); t = sub.sum() or 1
    return float(m[j, i]), float((X * sub).sum() / t), float((Y * sub).sum() / t)


def fit_r(obsv, S0, ep, r):
    """Orbit at heliocentric distance r at the epoch through the night centroids: 2 sky offsets + 2 velocities."""
    mj = [o[0] for o in obsv]; ob = [o[3] for o in obsv]; ra = np.array([o[1] for o in obsv]); de = np.array([o[2] for o in obsv])
    o0 = observer(ep); g = S0[:3] - o0; uv = g / np.linalg.norm(g); b = o0 @ uv
    rho = -b + math.sqrt(max(b * b - o0 @ o0 + r * r, 1)); X0 = o0 + rho * uv
    e1 = np.cross(uv, [0, 0, 1.]); e1 /= np.linalg.norm(e1); e2 = np.cross(uv, e1)
    V0 = S0[3:] * (rho / np.linalg.norm(g))
    def state(q):
        return np.r_[X0 + q[0] * rho * ARC * math.pi / 180 * e1 + q[1] * rho * ARC * math.pi / 180 * e2, V0 + q[2] * 1e-3 * e1 + q[3] * 1e-3 * e2]
    def res(q):
        p = G.positions(dict(state=state(q), epoch=ep), mj, ob)
        if p is None:
            return np.full(2 * len(mj), 1e3)
        return np.r_[(p[0] - ra) * np.cos(np.radians(de)) * 3600, (p[1] - de) * 3600]
    s = least_squares(res, np.zeros(4), diff_step=1e-2, max_nfev=40)
    return state(s.x), float(np.sqrt(np.mean(s.fun ** 2)))


def main():
    out = sys.argv[1]; fit_only = "--fit-only" in sys.argv
    V = json.load(open(H + "/candidates/vet.json"))
    C = [x for x in V if not x["known"] and x["smax"] is not None and x["smax"] < 3 and x["drop1"] >= 5 and x["p"] >= .9]
    ctrl = [x for x in V if x["known"] and x["f"] == "F3"]; ctrl.sort(key=lambda x: -x["p"]); C = ctrl[:2] + sorted(C, key=lambda x: -x["p"])
    F = json.load(open(H + "/candidates/files_confirm.json"))["exposures"]
    Tg = []
    for b, e in F.items():
        im = [f["name"] for f in e["files"] if "_ooi_" in f["name"]][0]
        Tg.append(dict(base=b, f=e["f"], cal=e["cal"], band=e["band"], t=e["t"], cover=e["cover"],
                       paths=[os.path.join(H, "data/confirm", [f["name"] for f in e["files"] if f"_oo{c}_" in f["name"]][0]) for c in "iwd"]))
    res = []
    for x in C:
        P = json.load(open(f"{H}/private/scenarios/{x['sid']}.json")); E = {e["image"]: e for e in P["all_exposures"]}
        d = os.path.join(H, "data", str(P["window"]))
        m_, ra_, de_, _ = x["track"][0]; ob_ = observer(m_); c, bd = None, 1e9        # p is rounded: pick by the track
        for c_ in json.load(open(f"{H}/runs/r1/gens/gen032/ls6_runs/{x['sid']}/candidates.json"))["candidates"]:
            if round(c_["p"], 3) != x["p"]:
                continue
            p_ = G.positions(dict(state=np.array(c_["state"], float), epoch=float(c_["epoch_mjd"])), [m_], [ob_])
            if p_ is not None and math.hypot((p_[0][0] - ra_) * math.cos(math.radians(de_)), p_[1][0] - de_) < bd:
                bd, c = math.hypot((p_[0][0] - ra_) * math.cos(math.radians(de_)), p_[1][0] - de_), c_
        S0 = np.array(c["state"], float); ep = float(c["epoch_mjd"])
        tag = f"{x['f']} p={x['p']}" + (" CONTROL(known)" if x["known"] else "")
        print("==", tag, flush=True)
        exs = {}
        def ex(img):
            if img not in exs:
                e = E[img]; exs[img] = G.Exposure(os.path.join(d, e["image"]), os.path.join(d, e["weight"]), os.path.join(d, e["mask"]))
            return exs[img]
        nights = {}
        for m, ra, de, img in x["track"]:
            nights.setdefault(int(np.floor(m - .5)), []).append((m, ra, de, img))
        obsv = []
        for n, tr in sorted(nights.items()):
            ax, N, D, used = comoving([(ex(t[3]), t[1], t[2]) for t in tr], 6.0)
            z = snr_map(N, D)
            if used == 0:
                continue
            s, dx, dy = peak(ax, z, 2.0)         # within the matching tolerance: a nearby star must not pull the centroid
            m = np.mean([t[0] for t in tr]); ra = np.mean([t[1] for t in tr]); de = np.mean([t[2] for t in tr])
            print(f"   scenario night {n}: S/N {s:.1f} offset ({dx:+.2f},{dy:+.2f})\" from {len(tr)} exp", flush=True)
            if s >= 4:
                obsv.append((m, ra + dx * ARC / math.cos(math.radians(de)), de + dy * ARC, observer(m), s))
        for o in exs.values():
            o.close()
        fam = {}
        for r in (26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 75, 80, 90):
            try:
                st, rms = fit_r(obsv, S0, ep, r)
                fam[r] = (st, rms)
            except Exception as e_:
                print("   fit failed r", r, repr(e_)[:80])
        print("   fit rms by r:", {r: round(v[1], 2) for r, v in fam.items()}, flush=True)
        best = min(v[1] for v in fam.values())
        fam = {r: v for r, v in fam.items() if v[1] <= max(.5, 2 * best)}      # distances the night centroids allow
        print("   distances kept:", sorted(fam), "best rms", round(best, 2), flush=True)
        rows = []
        tg = [t for t in Tg if t["f"] == x["f"] and (x["known"] or x["p"] in t["cover"])]
        if x["known"]:
            tg = [t for t in Tg if t["f"] == x["f"]]
        by = {}
        for t in tg:
            by.setdefault(t["cal"], []).append(t)
        figs = []
        for cal, L in sorted(by.items()):
            if fit_only:
                m = Time(cal + "T05:00:00").mjd; ob = observer(m)
                ps = [G.positions(dict(state=v[0], epoch=ep), [m], [ob]) for v in fam.values()]
                ps = [(float(p[0][0]), float(p[1][0])) for p in ps if p is not None]
                ra0 = np.median([p[0] for p in ps]); de0 = np.median([p[1] for p in ps])
                print(f"   {cal}: predicted {ra0:.4f} {de0:+.4f}; family spread {max(math.hypot((p[0]-ra0)*math.cos(math.radians(de0)), p[1]-de0) for p in ps)*3600:.1f}\"", flush=True)
                continue
            have = [t for t in L if all(os.path.exists(p) for p in t["paths"])]
            items = []
            for t in have:
                h = fits.getheader(t["paths"][0], 0); m = float(h["MJD-OBS"]) + float(h["EXPTIME"]) / 2 / 86400
                ob = observer(m)
                ps = [G.positions(dict(state=v[0], epoch=ep), [m], [ob]) for v in fam.values()]
                ps = [(float(p[0][0]), float(p[1][0])) for p in ps if p is not None]
                ra0 = float(np.median([p[0] for p in ps])); de0 = float(np.median([p[1] for p in ps]))
                spread = max(math.hypot((p[0] - ra0) * math.cos(math.radians(de0)), p[1] - de0) for p in ps) * 3600
                items.append((t, m, ra0, de0, spread))
            if not items:
                continue
            half = min(240.0, max(20.0, 1.5 * max(i[4] for i in items) + 10))
            exo = []
            for t, m, ra0, de0, sp in items:
                e_ = G.Exposure(*t["paths"])
                if e_.locate(ra0, de0) is not None or any(e_.locate(ra0 + a * half * ARC, de0 + b * half * ARC) for a in (-1, 1) for b in (-1, 1)):
                    exo.append((e_, ra0, de0))
                else:
                    e_.close()
            if not exo:
                print(f"   {cal}: not on any CCD of {len(items)} exposures", flush=True)
                continue
            ax, N, D, used = comoving(exo, half)
            z = snr_map(N, D)
            s, dx, dy = peak(ax, z)
            rows.append(dict(cal=cal, band=sorted(set(i[0]["band"] for i in items)), n=len(exo), t=items[0][0]["t"], spread=round(max(i[4] for i in items), 1), half=half, snr=round(s, 1), dx=round(dx, 1), dy=round(dy, 1), ra=exo[0][1] + dx * ARC / math.cos(math.radians(exo[0][2])), dec=exo[0][2] + dy * ARC, mjd=np.mean([i[1] for i in items])))
            print(f"   {cal} {rows[-1]['band']} {rows[-1]['n']} x {rows[-1]['t']:.0f}s: spread {rows[-1]['spread']}\" box +-{half:.0f}\": peak S/N {s:.1f} at ({dx:+.1f},{dy:+.1f})\"", flush=True)
            figs.append((cal, ax, z, dx, dy, s))
            for e_, _, _ in exo:
                e_.close()
        res.append(dict(f=x["f"], p=x["p"], known=x["known"], obs=[(o[0], o[1], o[2], o[4]) for o in obsv], fit={str(r): v[1] for r, v in fam.items()}, targets=rows))
        if figs:
            fig, axs = plt.subplots(1, len(figs), figsize=(2.6 * len(figs), 2.9), squeeze=False)
            for a, (cal, axx, z, dx, dy, s) in zip(axs[0], figs):
                a.imshow(z, origin="lower", cmap="gray_r", vmin=-2, vmax=7, extent=(axx[0], axx[-1], axx[0], axx[-1])); a.plot(dx, dy, "o", mfc="none", mec="r", ms=14, lw=.5)
                a.set_title(f"{cal}\nS/N {s:.1f}", fontsize=7); a.tick_params(labelsize=6)
            fig.suptitle(tag, fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(out, f"confirm_{x['f']}_{x['p']}.png"), dpi=110); plt.close(fig)
    json.dump(res, open(os.path.join(out, "confirm.json"), "w"), indent=0)
    print("DONE")


if __name__ == "__main__":
    main()
