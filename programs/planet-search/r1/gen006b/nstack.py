"""Gen 6b: pixel-level within-night shift-and-stack -> pseudo-exposures of night-stack detections.

For each night with >= 3 exposures that have stored S/N maps, the maps are resampled onto 1"/px tangent-plane
mosaics (stack.build_mosaics), stationary sources are masked (pixel positive on several nights), and the mosaics
are shifted (integer px) to the night's mid-time along every rate of hough.rate_grid (bodies at 25-150 au) and
summed: z = sum(clip(S/N, -4, 5)) / sqrt(n).  The max over rates is searched for local maxima z >= ZT with
n >= 3 exposures.  Each peak becomes a detection of a *pseudo-exposure* (one per night, mid-time, interpolated
observer position) appended to rows/results, so the existing linker sees night-stack detections of bodies too faint
for single exposures (the weight-2 regime: per-exposure S/N 2-3, no >= 3 sigma catalogue detection needed).
The linker treats pseudo-exposure detections as extra seeds only (link.py: `pseudo`).
"""
import math, os, time
import numpy as np
from orbit import ARC, T0, log

ZT = float(os.environ.get("TNO_NSTACK_ZT", 4.8))


def add_night_stacks(rows, meta, results, deadline):
    import torch
    import stack, hough
    from forced import Forced
    from orbit import radec_to_unit, tangent_frame
    t0 = time.time()
    dev = stack._dev(torch)
    FP = Forced(results)
    et = np.array([float(r["mjd_mid"]) for r in rows])
    eobs = np.array([[float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])] for r in rows])
    enight = np.floor(et - 0.5).astype(int)
    mexps = [e for e in sorted(results) if FP.has(e)]
    nights = {}
    for e in mexps:
        nights.setdefault(int(enight[e]), []).append(e)
    nights = {k: v for k, v in nights.items() if len(v) >= 3}
    if not nights:
        return 0
    if "ra" in meta:
        ra0, de0 = math.radians(meta["ra"]), math.radians(meta["dec"])
    else:
        ra0 = math.radians(float(rows[0]["ra_center"])); de0 = math.radians(float(rows[0]["dec_center"]))
    F = tangent_frame(ra0, de0)
    # observer velocity (au/day) by finite differences over all exposures
    o = np.argsort(et)
    ts, xs = et[o], eobs[o]
    PS = stack.PS
    npseudo = 0
    # one mosaic grid for all nights (extent from the CCD centres)
    cs = []
    for e in mexps:
        for g, m, cu, cmin in FP.maps[e]:
            c = F @ cu
            cs.append((c[0] / c[2], c[1] / c[2], g["rad"]))
    cs = np.array(cs)
    lo = (cs[:, :2] - cs[:, 2:3]).min(0); hi = (cs[:, :2] + cs[:, 2:3]).max(0)
    x0, y0 = int(math.floor(lo[0] / PS)) - 2, int(math.floor(lo[1] / PS)) - 2
    W, H = int(math.ceil(hi[0] / PS)) - x0 + 2, int(math.ceil(hi[1] / PS)) - y0 + 2
    if W * H > 300e6:
        log("nstack: mosaic too big", W, H)
        return 0
    lim = (x0, y0, W, H)
    # static mask over all nights (as in stack.py): per night the max over <= 2 exposures; a pixel is static if
    # S/N > 2.5 on >= 2 nights or > 6 on >= 2 nights... (>= 2 nights with data needed)
    alln = {}
    for e in mexps:
        alln.setdefault(int(enight[e]), []).append(e)
    if len(alln) < 2:
        return 0
    cnt = torch.zeros((H, W), device=dev, dtype=torch.int16)
    for nn_, es in alln.items():
        es = sorted(es, key=lambda e: et[e])[:2 if len(alln) <= 8 else 1]
        mo = stack.build_mosaics(torch, dev, FP, es, F, lim)
        mx = None
        for e in es:
            v = torch.nan_to_num(mo[e][0, 0], nan=-99.0)
            mx = v if mx is None else torch.maximum(mx, v)
        del mo
        cnt += (mx > 2.5).to(torch.int16)
        del mx
    static = torch.nn.functional.max_pool2d((cnt >= 2).float()[None, None], 3, 1, 1)[0, 0] > 0
    del cnt
    log(f"nstack: static mask {W}x{H}, {float(static.float().mean()):.3f} masked, {time.time() - t0:.1f}s")
    done_n = set()
    for nn in sorted(nights, key=lambda q: -len(nights[q])):
        if time.time() > deadline:
            log("nstack: deadline")
            break
        ee = sorted(nights[nn], key=lambda e: et[e])
        if len(ee) > 10:
            ee = [ee[i] for i in np.linspace(0, len(ee) - 1, 10).astype(int)]
        mos = stack.build_mosaics(torch, dev, FP, ee, F, lim)
        tm = 0.5 * (et[ee[0]] + et[ee[-1]])
        T_h = (et[ee[-1]] - et[ee[0]]) * 24.0
        sel = (ts > tm - 0.3) & (ts < tm + 0.3)
        if sel.sum() >= 2 and np.ptp(ts[sel]) > 0:
            vo = np.array([np.polyfit(ts[sel] - tm, xs[sel, k], 1)[0] for k in range(3)])
        else:
            vo = (xs[-1] - xs[0]) / max(ts[-1] - ts[0], 1e-3)
        vof = F @ vo
        V, dv = hough.rate_grid(vof, max(T_h, 0.3))
        if len(V) > 400:
            V = V[np.linspace(0, len(V) - 1, 400).astype(int)]
        # random order: a sweep cut short by its time share still covers the rate range evenly (coarser)
        V = V[np.random.default_rng(nn).permutation(len(V))]
        nleft = sum(1 for q in nights if q not in done_n)
        share = max(0.0, deadline - time.time()) / max(nleft, 1)
        done_n.add(nn)
        big = torch.zeros((H, W), device=dev, dtype=torch.bool)
        for e in ee:
            big |= torch.nan_to_num(mos[e][0, 0], nan=0.0) > 30
        bad = static | (torch.nn.functional.max_pool2d(big.float()[None, None], 9, 1, 4)[0, 0] > 0)
        del big
        Mc = torch.zeros((len(ee), H, W), device=dev, dtype=torch.float16)
        Vc = torch.zeros((len(ee), H, W), device=dev, dtype=torch.float16)
        for k, e in enumerate(ee):
            m_ = mos.pop(e)[0, 0]
            ok = torch.isfinite(m_) & ~bad
            Mc[k] = torch.where(ok, m_.clamp(-4, 5), torch.zeros((), device=dev, dtype=m_.dtype))
            Vc[k] = ok.to(torch.float16)
            del m_, ok
        n = len(ee)
        dth = (et[ee] - tm) * 24.0
        zmax = torch.full((H, W), -9.0, device=dev, dtype=torch.float16)
        imax = torch.zeros((H, W), device=dev, dtype=torch.int16)
        tl = time.time()
        nv = 0
        for vi, (vx, vy) in enumerate(V):
            if time.time() > deadline or time.time() - tl > share:
                break
            nv += 1
            S = torch.zeros((H, W), device=dev, dtype=torch.float16)
            C = torch.zeros((H, W), device=dev, dtype=torch.float16)
            for k in range(n):
                # body at p(tm) is at p + v*dt in exposure k -> sample exposure k at shifted index
                sx = int(round(vx * dth[k])); sy = int(round(vy * dth[k]))
                xa, xb = max(0, -sx), min(W, W - sx)
                ya, yb = max(0, -sy), min(H, H - sy)
                if xb <= xa or yb <= ya:
                    continue
                S[ya:yb, xa:xb] += Mc[k, ya + sy:yb + sy, xa + sx:xb + sx]
                C[ya:yb, xa:xb] += Vc[k, ya + sy:yb + sy, xa + sx:xb + sx]
            z = torch.where(C >= 3, S / C.clamp(min=1).sqrt(), torch.full_like(S, -9.0))
            better = z > zmax
            zmax = torch.where(better, z, zmax)
            imax = torch.where(better, torch.full_like(imax, vi), imax)
        zf = zmax.float()
        pk = (zf >= ZT) & (zf == torch.nn.functional.max_pool2d(zf[None, None], 7, 1, 3)[0, 0])
        yy, xx = torch.nonzero(pk, as_tuple=True)
        zz = zf[yy, xx]
        if len(zz) > 3000:
            top = torch.argsort(zz, descending=True)[:3000]
            yy, xx, zz = yy[top], xx[top], zz[top]
        # sub-pixel parabola on zmax
        def g(dy, dx):
            return zf[(yy + dy).clamp(0, H - 1), (xx + dx).clamp(0, W - 1)]
        c0 = zz
        ddx = (g(0, 1) - g(0, -1)) / (2 * (2 * c0 - g(0, 1) - g(0, -1)).clamp(min=1e-3))
        ddy = (g(1, 0) - g(-1, 0)) / (2 * (2 * c0 - g(1, 0) - g(-1, 0)).clamp(min=1e-3))
        px = (xx.float() + ddx.clamp(-0.5, 0.5)).cpu().numpy().astype(float) + x0
        py = (yy.float() + ddy.clamp(-0.5, 0.5)).cpu().numpy().astype(float) + y0
        zz = zz.cpu().numpy().astype(float)
        vix = imax[yy, xx].long().cpu().numpy()
        del Mc, Vc, zmax, imax, zf, S, C
        if dev == "cuda":
            torch.cuda.empty_cache()
        elif dev == "mps":
            torch.mps.empty_cache()
        xi, eta = px * PS, py * PS
        v = np.stack([xi, eta, np.ones_like(xi)], 1) @ F
        v /= np.linalg.norm(v, axis=1)[:, None]
        ra = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
        de = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
        # pseudo exposure
        ei = len(rows)
        r = dict(rows[ee[0]])
        r["exp_id"] = f"nstack{nn}"
        r["mjd_mid"] = repr(float(tm))
        r["magzero"] = ""
        ob = np.array([np.interp(tm, ts, xs[:, k]) for k in range(3)])
        r["obs_x"], r["obs_y"], r["obs_z"] = [repr(float(q)) for q in ob]
        rows.append(r)
        det = np.c_[ra, de, zz, np.full(len(zz), 0.8), np.full(len(zz), np.nan), np.full(len(zz), -1.0),
                    V[vix, 0], V[vix, 1]]
        results[ei] = dict(ei=ei, det=det, fwhm=np.nan, err=None, maps=[], pseudo=True)
        npseudo += 1
        log(f"nstack night {nn}: {n} exp, {W}x{H}, {nv}/{len(V)} rates (dv {dv:.2f}\"/h, T {T_h:.1f} h), "
            f"{len(zz)} peaks >= {ZT} (top {np.sort(zz)[::-1][:5].round(1).tolist()}), {time.time() - tl:.1f}s stack, "
            f"{time.time() - t0:.1f}s total")
    return npseudo
