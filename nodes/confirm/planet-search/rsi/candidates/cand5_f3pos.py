"""Measured places of the four F3 targets (two candidates, TRKSUB01, KNOWN_BODY_F3) on the F3 nights, to replace the program's orbit positions used by
cand2-cand4 grids. Per night: a comoving matched-filter stack (confirm.comoving's method, but CP mask 0 or 7 kept, since 7 marks sources that moved)
on a +-2.5" grid along the program's orbit, the peak within 1.2" of it, its weighted centroid, and so a sky place at the night's mean time.
TRKSUB01's places are compared with its filed orbit. Writes cand5_f3pos.json. Usage, from harness/: python -I ../candidates/cand5_f3pos.py"""
import json, math, os, sys
import numpy as np
from scipy.ndimage import convolve
H = os.path.expanduser("$ALE_HOME/planet-search/rsi"); sys.path.insert(0, H + "/candidates")
import confirm as C, suspects as Su
G = C.G; PS = 0.263; ARC = 1 / 3600.
cfg = json.load(open(H + "/runs/r1/config.json")); sid = cfg["eval"][2]
P = json.load(open(f"{H}/private/scenarios/{sid}.json")); E = sorted(P["all_exposures"], key=lambda e: e["mjd_mid"]); d0 = os.path.join(H, "data", str(P["window"])); mid = len(E) // 2
mj = [e["mjd_mid"] for e in E]; ob = [np.array([e["obs_x"], e["obs_y"], e["obs_z"]]) for e in E]; night = [int(math.floor(m - .5)) for m in mj]


def find(gen, ra, de, r=None, tol=1.):
    best = None
    for c in json.load(open(f"{H}/runs/r1/gens/{gen}/ls6_runs/{sid}/candidates.json"))["candidates"]:
        q = G.positions(dict(state=np.array(c["state"], float), epoch=c.get("epoch_mjd")), [mj[mid]], [ob[mid]])
        if q is None or (r is not None and abs(np.linalg.norm(c["state"][:3]) - r) > .06):
            continue
        if G.sep(float(q[0][0]), float(q[1][0]), ra, de) < tol and (best is None or c["p"] > best["p"]):
            best = c
    return best


def comoving(items, half):
    n = int(half / PS); ax = np.arange(-n, n + 1) * PS; N = np.zeros((len(ax), len(ax))); D = np.zeros_like(N)
    for ex, ra, de in items:
        cd = math.cos(math.radians(de)); ks = set()
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                l = ex.locate(ra + sx * half * ARC / cd, de + sy * half * ARC)
                if l:
                    ks.add(l[0])
        for k in ks:
            cc = [c for c in ex.ccds if c[0] == k][0]; w = cc[1]; fw = cc[4] or 4.0; DX, DY = np.meshgrid(ax, ax)
            xx, yy = w.all_world2pix(ra + DX.ravel() * ARC / cd, de + DY.ravel() * ARC, 0); xx = np.rint(xx).astype(int).reshape(DX.shape); yy = np.rint(yy).astype(int).reshape(DX.shape)
            x0, x1 = max(0, xx.min() - 25), min(cc[2], xx.max() + 26); y0, y1 = max(0, yy.min() - 25), min(cc[3], yy.max() + 26)
            if x1 - x0 < 30 or y1 - y0 < 30:
                continue
            im, wt, ms = ex.data(k); a = im[y0:y1, x0:x1].astype(float); m_ = ms[y0:y1, x0:x1]
            wg = np.where((m_ == 0) | (m_ == 7), np.clip(wt[y0:y1, x0:x1].astype(float), 0, None), 0)
            if (wg > 0).sum() < 100:
                continue
            a = a - np.median(a[wg > 0]); sd = 1.4826 * np.median(np.abs(a[wg > 0])); wg = np.where(np.abs(a) > 30 * sd, 0, wg)
            s = fw / 2.355; r = int(3 * s) + 1; yg, xg = np.mgrid[-r:r + 1, -r:r + 1]; g = np.exp(-(xg ** 2 + yg ** 2) / (2 * s * s)); g /= g.sum()
            num = convolve(wg * a, g, mode="constant"); den = convolve(wg, g * g, mode="constant"); ok = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
            N[ok] += num[yy[ok] - y0, xx[ok] - x0]; D[ok] += den[yy[ok] - y0, xx[ok] - x0]
    return ax, N, D


S = json.load(open(H + "/candidates/screen_new_p05_08.json")); T = {}
for x in S:
    if x["field"] == "F3" and (x["verdict"] == "PASS" and x["kind"] == "candidate"):
        T[f"F3_r{x['r']}_p{x['p']}"] = find(x["gen"], x["ra"][mid], x["dec"][mid], x["r"])
    if x["field"] == "F3" and x["kind"] == "control":
        T["GB138_control"] = find(x["gen"], x["ra"][mid], x["dec"][mid], x["r"])
z = next(o for o in json.load(open(H + "/candidates/new_cands.json")) if o["field"] == "F3" and abs(o["r"] - 44.1) < .2 and o["rounds"] >= 60)
T["TRKSUB01_control"] = find("gen083", z["ra"], z["dec"], tol=5.)
X = {k: G.Exposure(os.path.join(d0, e["image"]), os.path.join(d0, e["weight"]), os.path.join(d0, e["mask"])) for k, e in enumerate(E)}
O = json.load(open(H + "/candidates/mpc_report/report_orbit_cfht.json")); OUT = {}
for n, c in T.items():
    ra, de = G.positions(dict(state=np.array(c["state"], float), epoch=c.get("epoch_mjd")), mj, ob); out = []
    for nt in sorted(set(night)):
        ks = [k for k in range(len(E)) if night[k] == nt]; ax, N, D = comoving([(X[k], float(ra[k]), float(de[k])) for k in ks], 2.5); z_ = C.snr_map(N, D)
        pk, ox, oy = C.peak(ax, z_, 1.2); m = float(np.mean([mj[k] for k in ks])); r0, d0_ = float(np.mean(ra[ks])), float(np.mean(de[ks]))
        rm, dm = r0 + ox / 3600 / math.cos(math.radians(d0_)), d0_ + oy / 3600; row = dict(night=nt, t=m, peak=pk, off=[ox, oy], ra=rm, dec=dm, n=len(ks))
        if n == "TRKSUB01_control":
            q = G.positions(dict(state=np.array(O["state"], float), epoch=O["epoch"]), [m], [Su.obs_b(m)])
            row["minus_filed"] = [((rm - q[0][0] + 180) % 360 - 180) * math.cos(math.radians(dm)) * 3600, (dm - q[1][0]) * 3600]
            row["prog_minus_filed"] = [((r0 - q[0][0] + 180) % 360 - 180) * math.cos(math.radians(dm)) * 3600, (d0_ - q[1][0]) * 3600]
        out.append(row)
        for k in ks:
            X[k].cache.clear()
    OUT[n] = out
    print(n, [(o["night"], round(o["peak"], 1), [round(v, 2) for v in o["off"]]) for o in out], flush=True)
    if n == "TRKSUB01_control":
        print("   measured minus filed orbit:", [[round(v, 2) for v in o["minus_filed"]] for o in out], "| program orbit minus filed:", [[round(v, 2) for v in o["prog_minus_filed"]] for o in out], flush=True)
json.dump(OUT, open(H + "/candidates/cand5_f3pos.json", "w"), indent=0)
