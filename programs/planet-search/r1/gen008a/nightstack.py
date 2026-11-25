"""Gen 8a: per-night shift-and-stack -> pseudo-exposure detections for the linker.

For every night with >= 2 exposures that have stored S/N maps, the maps are resampled onto one tangent-plane mosaic
(1"/px, stack.build_mosaics), clipped to [-3, 4] and summed along a grid of in-night rates (vx, vy) with integer-pixel
shifts.  z = sum / sqrt(n valid).  The max over rates is peak-found; peaks with z >= ZN become detections of a
pseudo-exposure at the night's mean time and mean observer position (appended to rows/results).  The existing
linker then treats them like any exposure: bodies at S/N ~2-3 per exposure that only show up in a night stack can be
linked across nights.  Static sources: pixels where the zero-rate stack is > 5 are masked (grown by 2 px).
"""
import math, os, time
import numpy as np
from orbit import ARC, T0, log, radec_to_unit, tangent_frame

PS = 1.0 * ARC


def run_nightstack(rows, results, meta, deadline):
    import torch
    import stack
    from forced import Forced
    dev = stack._dev(torch)
    FP = Forced(results)
    et = np.array([float(r["mjd_mid"]) for r in rows])
    enight = np.floor(et - 0.5).astype(int)
    mexps = [e for e in sorted(results) if FP.has(e)]
    if not mexps:
        return
    # frame + mosaic extent from all detections
    D = np.concatenate([results[e]["det"] for e in results if len(results[e]["det"])])
    u = radec_to_unit(np.radians(D[:, 0]), np.radians(D[:, 1]))
    if "ra" in meta:
        ra0, de0 = math.radians(meta["ra"]), math.radians(meta["dec"])
    else:
        ra0 = math.atan2(np.mean(u[:, 1]), np.mean(u[:, 0])); de0 = math.asin(np.mean(u[:, 2]))
    F = tangent_frame(ra0, de0)
    ug = u @ F.T
    th = ug[:, :2] / ug[:, 2:3]
    lo = np.percentile(th, 0.1, axis=0) - 0.005
    hi = np.percentile(th, 99.9, axis=0) + 0.005
    del D, u, ug, th
    x0, y0 = int(math.floor(lo[0] / PS)), int(math.floor(lo[1] / PS))
    W, H = int(math.ceil(hi[0] / PS)) - x0, int(math.ceil(hi[1] / PS)) - y0
    lim = (x0, y0, W, H)
    ZN = float(os.environ.get("TNO_NS_Z", 4.6))
    SMAX = float(os.environ.get("TNO_NS_SMAX", 5.0))
    VMAX = float(os.environ.get("TNO_NS_VMAX", 5.5))     # "/hr
    nights = {}
    for e in mexps:
        nights.setdefault(int(enight[e]), []).append(e)
    nights = {k: v for k, v in nights.items() if len(v) >= 2}
    log(f"nightstack: {len(nights)} nights with >= 2 maps, mosaic {W}x{H} on {dev}")
    added = 0
    for nn, ee in sorted(nights.items(), key=lambda kv: -len(kv[1])):
        if time.time() - T0 > deadline:
            log("nightstack: deadline")
            break
        t1 = time.time()
        ee = ee[:24]
        te = et[ee]
        tref = float(np.mean(te))
        dth = (te - tref) * 24.0            # hours
        span = float(dth.max() - dth.min())
        if span < 0.05:
            continue
        dv = max(1.4 / span, 0.35)
        nst = int(math.floor(VMAX / dv))
        while (2 * nst + 1) ** 2 > 500:
            dv *= 1.15; nst = int(math.floor(VMAX / dv))
        rates = [(i * dv, j * dv) for i in range(-nst, nst + 1) for j in range(-nst, nst + 1)
                 if math.hypot(i * dv, j * dv) <= VMAX + 1e-6]
        rates.sort(key=lambda r: math.hypot(*r))     # slow rates first if the deadline cuts the loop
        P = int(math.ceil(VMAX * float(np.abs(dth).max()))) + 2
        mos = stack.build_mosaics(torch, dev, FP, ee, F, lim)
        S = []
        for e in ee:
            m = mos.pop(e)[0, 0]
            m = torch.nn.functional.pad(m.float(), (P, P, P, P), value=float("nan"))
            S.append(m.half())
        del mos
        n = len(S)
        # zero-rate stack -> static mask
        acc = torch.zeros((H, W), device=dev); cnt = torch.zeros((H, W), device=dev)
        for m in S:
            c = m[P:P + H, P:P + W].float()
            v = torch.isfinite(c)
            acc += torch.where(v, c.clamp(-3, 4), torch.zeros_like(c)); cnt += v.float()
        z0 = acc / torch.sqrt(cnt.clamp(min=1))
        bad = torch.nn.functional.max_pool2d((z0 > 5.0).float()[None, None], 5, 1, 2)[0, 0] > 0
        del acc, cnt, z0
        best = torch.full((H, W), -99.0, device=dev)
        barg = torch.zeros((H, W), dtype=torch.int16, device=dev)
        need = max(2.0, 0.6 * n)
        for ri, (vx, vy) in enumerate(rates):
            if time.time() - T0 > deadline + 15:
                break
            acc = torch.zeros((H, W), device=dev); cnt = torch.zeros((H, W), device=dev)
            for k, m in enumerate(S):
                # body at x at tref is at x + v*dt at time t_k
                sx = int(round(vx * dth[k])); sy = int(round(vy * dth[k]))
                c = m[P + sy:P + sy + H, P + sx:P + sx + W].float()
                v = torch.isfinite(c)
                acc += torch.where(v, c.clamp(-3, 4), torch.zeros_like(c)); cnt += v.float()
            z = torch.where(cnt >= need, acc / torch.sqrt(cnt.clamp(min=1)), torch.full_like(acc, -99.0))
            upd = z > best
            best = torch.where(upd, z, best)
            barg = torch.where(upd, torch.full_like(barg, ri), barg)
            del acc, cnt, z, upd
        # static mask only for rates that move > 1.5 px within the night (slower peaks: the linker's stationary
        # filter handles stars, and masking them would kill every body on short nights)
        tmax = float(np.abs(dth).max())
        moving = torch.tensor([math.hypot(vx, vy) * tmax > 1.5 for vx, vy in rates], device=dev)
        best[bad & moving[barg.long()]] = -99.0
        mp = torch.nn.functional.max_pool2d(best[None, None], 5, 1, 2)[0, 0]
        pk = (best >= ZN) & (best == mp)
        idx = torch.nonzero(pk)
        # keep only night-stack-only sources: drop peaks seen at S/N >= SMAX in any single exposure along the rate
        # (those are in the single-exposure catalogue already; stars/bright sources saturate the clipped stack)
        if len(idx):
            R = torch.tensor(rates, dtype=torch.float32, device=dev)[barg[idx[:, 0], idx[:, 1]].long()]
            mx = torch.full((len(idx),), -99.0, device=dev)
            for k, m in enumerate(S):
                sy = torch.round(R[:, 1] * float(dth[k])).long(); sx = torch.round(R[:, 0] * float(dth[k])).long()
                val = m[P + idx[:, 0] + sy, P + idx[:, 1] + sx].float()
                mx = torch.maximum(mx, torch.nan_to_num(val, nan=-99.0))
            nall = len(idx)
            idx = idx[mx < SMAX]
            log(f"nightstack night {nn}: {nall} peaks, {len(idx)} without a single-exposure S/N >= {SMAX}")
        if len(idx) > 20000:
            vals = best[idx[:, 0], idx[:, 1]]
            idx = idx[torch.argsort(-vals)[:20000]]
        if len(idx) == 0:
            del S, best, barg, mp
            continue
        iy, ix = idx[:, 0], idx[:, 1]
        iyc = iy.clamp(1, H - 2); ixc = ix.clamp(1, W - 2)
        c0 = best[iyc, ixc]
        def q(a, b):
            den = a - 2 * c0 + b
            o = torch.where(den < -1e-6, 0.5 * (a - b) / den, torch.zeros_like(den))
            return o.clamp(-0.5, 0.5)
        ox = q(best[iyc, ixc - 1].clamp(min=-5), best[iyc, ixc + 1].clamp(min=-5))
        oy = q(best[iyc - 1, ixc].clamp(min=-5), best[iyc + 1, ixc].clamp(min=-5))
        px = (ixc.float() + ox).cpu().numpy().astype(np.float64)
        py = (iyc.float() + oy).cpu().numpy().astype(np.float64)
        zz = c0.cpu().numpy().astype(np.float64)
        rr = barg[iyc, ixc].long().cpu().numpy()
        del S, best, barg, mp
        tx = (px + x0) * PS; ty = (py + y0) * PS
        vv = np.c_[tx, ty, np.ones_like(tx)] @ F
        vv /= np.linalg.norm(vv, axis=1)[:, None]
        ra = np.degrees(np.arctan2(vv[:, 1], vv[:, 0])) % 360.0
        de = np.degrees(np.arcsin(np.clip(vv[:, 2], -1, 1)))
        det = np.c_[ra, de, zz, np.full(len(zz), 0.8), np.zeros(len(zz)), np.full(len(zz), 98.0), px, py]
        # pseudo-exposure row
        r0 = dict(rows[ee[0]])
        r0["exp_id"] = f"nightstack_{nn}"
        r0["mjd_mid"] = repr(tref)
        ob = np.array([[float(rows[e]["obs_x"]), float(rows[e]["obs_y"]), float(rows[e]["obs_z"])] for e in ee])
        om = ob.mean(0)
        r0["obs_x"], r0["obs_y"], r0["obs_z"] = repr(om[0]), repr(om[1]), repr(om[2])
        r0["magzero"] = ""
        ei = len(rows)
        rows.append(r0)
        results[ei] = dict(ei=ei, det=det, fwhm=np.nan, err=None, maps=[], pseudo=True)
        added += len(det)
        log(f"nightstack night {nn}: {n} exp, span {span:.2f} h, {len(rates)} rates (dv {dv:.2f}\"/h), "
            f"{len(det)} peaks z>={ZN}, top z {np.sort(zz)[::-1][:5].round(1).tolist()} ({time.time() - t1:.1f}s)")
        if dev == "mps":
            torch.mps.empty_cache()
        elif dev == "cuda":
            torch.cuda.empty_cache()
    log(f"nightstack: {added} pseudo detections")
