"""Gen 8c: pixel-level within-night shift-and-stack on the GPU -> extra linker detections for faint bodies.

Bodies visible only in a stack of one night's exposures (grader weight 2) rarely have >= 3 sigma detections in single
exposures, so neither the pair seeds nor the catalogue-level hough stage (gen 2) can see them.  For every night with
>= NS_MIN exposures that have stored S/N maps:
  1. the S/N maps of the night are resampled onto one tangent-plane mosaic per exposure (stack.build_mosaics, 1"/px);
  2. static pixels (S/N > 3 in >= half of the night's exposures, grown by 1 px) are zeroed; values clipped to [-4, 5];
  3. for each rate (vx, vy) of hough.rate_grid (rates a body at 25-150 au can have; step ~1.4"/night length) the
     mosaics are shifted by integer pixels to the reference exposure's time and summed: S = sum / sqrt(n valid);
  4. local maxima (5x5) of max-over-rates S >= thr become *virtual detections*: (ra, dec) at the reference exposure
     (the middle exposure of the night), S/N = S.  link.run appends them to the catalogue of that exposure and makes
     them seeds of their night.  The linker, orbit fit, extension and scoring then treat them like detections.
Results are cached per detection set, so the null-calibration run (night-permuted, same within-night times) reuses
them; this keeps the null junk level comparable.
"""
import math, os, time
import numpy as np
from orbit import ARC, T0, log

PS = 1.0 * ARC


def _dev(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_CACHE = {}


def virtual_dets(FP, F, et, enight, done, vobs, u_use, enight_use, tlimit, ckey=None, thr=None, nmin=None,
                 maxexp=16, cap=1500, nrate=300):
    """Returns (rows (N, 8) like detect's det: ra, dec, sn, sharp, flux, ccd=-1, 0, 0; exposure index (N,))."""
    key = (ckey, tuple(int(e) for e in done))
    if key in _CACHE:
        return _CACHE[key]
    thr = float(os.environ.get("TNO_NS_THR", 5.5)) if thr is None else thr
    nmin = int(os.environ.get("TNO_NS_MIN", 3)) if nmin is None else nmin
    out_r, out_e = [np.zeros((0, 8))], [np.zeros(0, int)]
    t0 = time.time()
    try:
        import torch
        import stack, hough
        dev = _dev(torch)
        mexps = [int(e) for e in done if FP.has(e)]
        nights = {}
        for e in mexps:
            nights.setdefault(int(enight[e]), []).append(e)
        todo = sorted([n for n in nights if len(nights[n]) >= nmin], key=lambda n: -len(nights[n]))
        for nn in todo:
            if time.time() - t0 > tlimit:
                log("nightstack: time limit")
                break
            ee = sorted(nights[nn], key=lambda e: et[e])
            if len(ee) > maxexp:
                ee = [ee[int(round(x))] for x in np.linspace(0, len(ee) - 1, maxexp)]
            ne = len(ee)
            eref = ee[ne // 2]
            # mosaic limits from this night's usable detections
            m = enight_use == nn
            if m.sum() < 50:
                continue
            ug = u_use[m] @ F.T
            th = ug[:, :2] / ug[:, 2:3]
            lo = np.percentile(th, 0.05, axis=0) - 0.003
            hi = np.percentile(th, 99.95, axis=0) + 0.003
            x0, y0 = int(math.floor(lo[0] / PS)), int(math.floor(lo[1] / PS))
            W, H = int(math.ceil(hi[0] / PS)) - x0, int(math.ceil(hi[1] / PS)) - y0
            if W * H > 2.5e8 or W < 100 or H < 100:
                log(f"nightstack night {nn}: mosaic {W}x{H} skipped")
                continue
            lim = (x0, y0, W, H)
            tb = time.time()
            mos = stack.build_mosaics(torch, dev, FP, ee, F, lim)
            Ms = [mos[e][0, 0] for e in ee]
            del mos
            # static pixels
            cnt = torch.zeros((H, W), dtype=torch.float16, device=dev)
            for M in Ms:
                cnt += (M > 3.0).to(torch.float16)
            static = cnt >= max(2.0, 0.5 * ne)
            static = torch.nn.functional.max_pool2d(static.float()[None, None], 3, 1, 1)[0, 0] > 0
            del cnt
            dt_h = np.array([(et[e] - et[eref]) * 24.0 for e in ee])
            T_h = float(dt_h.max() - dt_h.min())
            vof = F @ vobs[ee].mean(0)
            V, dv = hough.rate_grid(vof, max(T_h, 0.05), nmax=nrate)
            # faint static sources (galaxies, faint stars) below the static mask stack up at slow rates: when every
            # trial body moves >= 3" over the night, subtract the night's per-pixel median first
            med = None
            if ne >= 5 and float(np.percentile(np.linalg.norm(V, axis=1), 25)) * T_h >= 3.0:
                med = torch.empty((H, W), dtype=torch.float16, device=dev)
                CH = max(16, int(4e7 // (W * ne)))
                for r0 in range(0, H, CH):
                    blk = torch.stack([M[r0:r0 + CH].float() for M in Ms])
                    med[r0:r0 + CH] = torch.nan_to_num(stack._nanmed_mps(torch, blk), nan=0.0).to(torch.float16)
                    del blk
            X, Vd = [], []
            for M in Ms:
                ok = torch.isfinite(M) & ~static
                Mv = torch.nan_to_num(M, nan=0.0)
                if med is not None:
                    Mv = Mv - med
                X.append(torch.where(ok, torch.clamp(Mv, -4.0, 5.0), torch.zeros_like(M)).to(torch.float16))
                Vd.append(ok.to(torch.float16))
                del Mv
            med_used = med is not None
            del Ms, static, med
            # integer shift per (rate, exposure); distinct shift tuples only
            S = np.rint(V[:, None, :] * dt_h[None, :, None] / (PS / ARC)).astype(int)   # (R, ne, 2) px
            _, uidx = np.unique(S.reshape(len(S), -1), axis=0, return_index=True)
            uidx = np.sort(uidx)
            best = torch.full((H, W), -1e9, dtype=torch.float32, device=dev)
            bidx = torch.zeros((H, W), dtype=torch.int16, device=dev)
            nmin_pix = max(2.0, 0.6 * ne)
            tr = time.time()
            nr = 0
            for r in uidx:
                if time.time() - t0 > tlimit:
                    break
                acc = torch.zeros((H, W), dtype=torch.float32, device=dev)
                cn = torch.zeros((H, W), dtype=torch.float16, device=dev)
                for k in range(ne):
                    # value at pixel p of the stack = X_k[p + shift_k]
                    sx, sy = int(S[r, k, 0]), int(S[r, k, 1])
                    xa, xb = max(0, -sx), min(W, W - sx)
                    ya, yb = max(0, -sy), min(H, H - sy)
                    if xb <= xa or yb <= ya:
                        continue
                    acc[ya:yb, xa:xb] += X[k][ya + sy:yb + sy, xa + sx:xb + sx]
                    cn[ya:yb, xa:xb] += Vd[k][ya + sy:yb + sy, xa + sx:xb + sx]
                cf = cn.float()
                s = torch.where(cf >= nmin_pix, acc / torch.sqrt(torch.clamp(cf, min=1.0)),
                                torch.full_like(acc, -1e9))
                imp = s > best
                best = torch.where(imp, s, best)
                bidx = torch.where(imp, torch.full_like(bidx, int(r)), bidx)
                nr += 1
                del acc, cn, cf, s, imp
            del X, Vd
            mp = torch.nn.functional.max_pool2d(best[None, None], 5, 1, 2)[0, 0]
            pk = (best >= thr) & (best >= mp)
            ys, xs = torch.nonzero(pk, as_tuple=True)
            sv = best[ys, xs].cpu().numpy().astype(np.float64)
            xs = xs.cpu().numpy(); ys = ys.cpu().numpy()
            o = np.argsort(-sv)[:cap]
            xs, ys, sv = xs[o], ys[o], sv[o]
            del best, bidx, mp, pk
            if dev == "cuda":
                torch.cuda.empty_cache()
            elif dev == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            if len(sv):
                f = np.stack([(xs + x0) * PS, (ys + y0) * PS, np.ones(len(xs))], 1)
                uv = f @ F                     # frame -> ICRS (F rows are the frame axes)
                uv /= np.linalg.norm(uv, axis=1)[:, None]
                ra = np.degrees(np.arctan2(uv[:, 1], uv[:, 0])) % 360.0
                de = np.degrees(np.arcsin(np.clip(uv[:, 2], -1, 1)))
                rows = np.c_[ra, de, sv, np.ones(len(sv)), np.zeros(len(sv)), -np.ones(len(sv)),
                             np.zeros(len(sv)), np.zeros(len(sv))]
                out_r.append(rows); out_e.append(np.full(len(sv), eref, int))
            log(f"nightstack night {nn}: {ne} exp over {T_h:.2f} h, {W}x{H} px, {nr}/{len(uidx)} shift sets "
                f"(dv {dv:.2f}\"/h), {len(sv)} peaks >= {thr} (top {sv[:3].round(1).tolist()}), "
                f"medsub {med_used} build {tr - tb:.1f}s stack {time.time() - tr:.1f}s")
    except Exception as ex:
        import traceback; traceback.print_exc()
        log("nightstack failed", ex)
    res = (np.concatenate(out_r), np.concatenate(out_e))
    _CACHE[key] = res
    log(f"nightstack: {len(res[0])} virtual detections in {time.time() - t0:.1f}s")
    return res
