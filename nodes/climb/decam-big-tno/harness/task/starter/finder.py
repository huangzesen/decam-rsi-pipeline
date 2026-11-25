"""Starter finder (the plain search): static cut, then tracklet-less GPU linking (link_gpu.py) one season at a time at
30-80 au and over pairs of seasons at 80-1,000 au, until the time budget runs out.

Usage: python finder.py --scen DIR --out DIR --budget-s SECONDS
Writes OUT/candidates.json = [{"det": [row ids of DIR/det_*.npy], "p": float}, ...], best first.

1. Static cut: positions on a tangent plane at the region centre in 0.6" cells; a detection is static when detections on
   >= 2 other nights fall in its 3 x 3 block of cells. Kept: not static, S/N >= 5, flag bit 1 clear, 0.4 <= sharp <= 2.5.
2. Seeds (pairs start only from these; any kept detection can join a linkage): flag == 0, 0.5 <= sharp <= 1.8; an exposure
   with more than 3 x the median seeds and > 2,000 gives none.
3. Seasons: +-6 months around the region's opposition. link_gpu.py per season (30-80 au, dgamma 1e-4, >= 4 nights), then
   per two consecutive seasons (80-1,000 au, dgamma 1e-5, >= 5 nights), each with an equal share of what is left.
4. Every linkage on >= 5 nights is a candidate; p from its nights and fit residual.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AS = np.pi / 180 / 3600


def static_cut(ra, dec, night, cell_as=0.6):
    r, d = np.radians(ra), np.radians(dec)
    v = np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], 1)
    c = v.mean(0); c /= np.linalg.norm(c)
    ex = np.array([-c[1], c[0], 0.0]); ex /= np.linalg.norm(ex); ey = np.cross(c, ex)
    den = v @ c
    cx = np.floor((v @ ex) / den / AS / cell_as).astype(np.int64); cy = np.floor((v @ ey) / den / AS / cell_as).astype(np.int64)
    del v
    cx -= cx.min() - 2; cy -= cy.min() - 2
    W = int(cy.max() + 3); cell = cx * W + cy
    _, nn = np.unique(night, return_inverse=True)
    cn = np.unique(cell * 65536 + nn)
    cn9 = np.unique(np.concatenate([cn + (dx * W + dy) * 65536 for dx in (-1, 0, 1) for dy in (-1, 0, 1)]))
    uc, cnt = np.unique(cn9 // 65536, return_counts=True)
    i = np.minimum(np.searchsorted(uc, cell), len(uc) - 1)
    return np.where(uc[i] == cell, cnt[i], 1) >= 3, (np.arctan2(c[1], c[0]) % (2 * np.pi), np.arcsin(c[2]))


def main():
    T0 = time.time()
    ap = argparse.ArgumentParser(); ap.add_argument("--scen", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = lambda k: np.load(os.path.join(a.scen, f"det_{k}.npy"), mmap_mode="r")
    E = json.load(open(os.path.join(a.scen, "exposures.json")))
    et = np.array([e["mjd_mid"] for e in E]); obs = np.array([e["obs_xyz"] for e in E])
    exp = np.asarray(L("exp")); night_e = np.floor(et - 0.5).astype(np.int64)
    ra, dec = np.asarray(L("ra")), np.asarray(L("dec"))
    static, (ra0, dec0) = static_cut(ra, dec, night_e[exp])
    snr, sharp, flag = np.asarray(L("snr")), np.asarray(L("sharp")), np.asarray(L("flag"))
    keep = np.flatnonzero(~static & (snr >= 5) & ((flag & 1) == 0) & (sharp >= 0.4) & (sharp <= 2.5))
    seed = (flag[keep] == 0) & (sharp[keep] >= 0.5) & (sharp[keep] <= 1.8)
    per = np.bincount(exp[keep][seed], minlength=len(E)); med = np.median(per[per > 0]) if (per > 0).any() else 0
    seed &= ~((per > 3 * med) & (per > 2000))[exp[keep]]
    print(f"{len(ra):,} detections, static {static.mean():.1%}, kept {len(keep):,}, seeds {seed.sum():,}, {time.time() - T0:.0f}s", file=sys.stderr)
    del ra, dec, static
    # seasons around opposition
    eps = np.radians(23.4393)
    lam = np.arctan2(np.sin(ra0) * np.cos(eps) + np.tan(dec0) * np.sin(eps), np.cos(ra0)) % (2 * np.pi)
    opp0 = 51544.5 + 79.0 + ((np.degrees(lam) + 180) % 360) / 360 * 365.25
    sea = np.floor((et - opp0) / 365.25 + 0.5).astype(int)
    seasons = [[float(et[sea == s].min() - 0.5), float(et[sea == s].max() + 0.5)] for s in np.unique(sea)]
    tmp = tempfile.mkdtemp(prefix="finder_")
    f = os.path.join(tmp, "w.npz")
    np.savez(f, det_ra=np.radians(np.asarray(L("ra"))[keep]), det_dec=np.radians(np.asarray(L("dec"))[keep]),   # the linker works in radians
             det_exp=exp[keep].astype(np.int64),
             det_obj=-np.ones(len(keep), np.int32), det_mag=np.zeros(len(keep), np.float32), det_ndet=np.ones(len(keep), np.int8),
             exp_t=et, exp_ra=np.zeros(len(E)), exp_dec=np.zeros(len(E)), exp_obs=obs,
             obj_a=np.array([100.]), obj_e=np.array([0.]), obj_inc=np.array([0.]), obj_Om=np.array([0.]), obj_om=np.array([0.]),
             obj_M0=np.array([0.]), obj_t0=np.array([float(et[0])]))
    R = json.load(open(os.path.join(a.scen, "region.json")))
    json.dump(dict(ra0=float(ra0), dec0=float(dec0), seasons_mjd=seasons), open(f.replace(".npz", ".json"), "w"))
    np.savez(os.path.join(tmp, "seed.npz"), seed=seed)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    # runs: each season near (30-80 au), then each pair of seasons far (80-1000 au); seasons with few nights skipped
    nights_in = [len(np.unique(night_e[sea == s])) for s in np.unique(sea)]
    runs = [(k, 1, 30, 80, "1e-4", 4) for k in range(len(seasons)) if nights_in[k] >= 4]
    runs += [(k, 2, 80, 1000, "1e-5", 5) for k in range(len(seasons) - 1) if nights_in[k] + nights_in[k + 1] >= 5]
    core = R["radius_deg"]
    cands = []
    for i, (k, span, rmin, rmax, dg, mn) in enumerate(runs):
        left = a.budget_s - (time.time() - T0) - 30
        if left < 30:
            break
        share = left / (len(runs) - i)
        out = os.path.join(tmp, f"links_{i}.npz")
        cmd = [sys.executable, os.path.join(HERE, "link_gpu.py"), f, "--season", str(k), "--span", str(span), "--rmin", str(rmin),
               "--rmax", str(rmax), "--dgamma", dg, "--minnights", str(mn), "--core", f"{core:.2f}", "--fp32", "--refine", "--polish",
               "--gdot-tol", "4", "--extend", "index", "--device", dev, "--seed-mask", os.path.join(tmp, "seed.npz"),
               "--budget", f"{max(5.0, 0.6 * share):.0f}", "--save-links", out, "--out", os.path.join(tmp, f"res_{i}.json")]
        try:
            subprocess.run(cmd, cwd=HERE, stdout=subprocess.DEVNULL, stderr=sys.stderr, timeout=max(20.0, share), check=True)
            Z = np.load(out); sel = Z["sel"]
            for j in range(len(Z["nights"])):
                if Z["nights"][j] < 5:
                    continue
                det = keep[sel[Z["det"][Z["off"][j]:Z["off"][j + 1]]]]
                p = float(Z["nights"][j] / (Z["nights"][j] + 5.0) / (1.0 + float(Z["rms"][j]) / 0.3))
                cands.append(dict(det=[int(x) for x in det], p=round(min(max(p, 0.0), 1.0), 6)))
        except Exception as ex:
            print(f"run {i} (season {k}, span {span}) stopped: {type(ex).__name__}", file=sys.stderr)
    shutil.rmtree(tmp, ignore_errors=True)
    cands.sort(key=lambda c: -c["p"])
    json.dump(cands[:2000], open(os.path.join(a.out, "candidates.json"), "w"))
    print(f"{len(cands)} candidates from {len(runs)} runs, {time.time() - T0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
