"""Fifth-night check of the four-detection linkages ("quads") of the sparse pass (gen013 of this run).

A quad is two nights of one season joined to two nights of another: 8 measurements, 6 fitted parameters, so it has only
2 degrees of freedom and a whole region holds tens of thousands of chance quads.  The sparse pass therefore hands only
its few thousand best-fitting quads to the growth, and at the scale of a whole region a real quad of ordinary fit is cut
by that cap (on a tenth of a region the cap never binds, and the same body is found).  But a quad already fixes the whole
track: its fitted model predicts where the body is in every other exposure to about an arcsecond, and a chance quad has a
kept detection there on another night only a few per cent of the time.  So every passing quad is taken through all the
exposures once (one batched prediction and one lookup in the per-exposure index), and the quads that have a detection on
the track on at least one further night go to the growth ahead of the capped list, whatever their rank in it.

The model is the linear one the quads were fitted with (finder.sp_linfit): with tt = t - tref and mg3 = MU gamma^3,
    theta (rf + gdot tt - gamma Z) = (alpha, beta) c + (alpha-dot, beta-dot) s - gamma (X, Y)
    c = 1 - mg3 tt^2 / 2 + mg3 gdot tt^3 / 2,   s = tt - mg3 tt^3 / 6,   rf = 1 - mg3 tt^2 / 2 + mg3 gdot tt^3 / 3
(X, Y, Z): the observer in the tangent frame.  Nothing here knows a region, a body or a date.
"""
import os
import time

import numpy as np
import torch

TOL_IN = float(os.environ.get("FINDER_F5TOL", 1.0))       # arcsec: a detection this near the track counts inside the quad's own span
TOL_OUT = float(os.environ.get("FINDER_F5TOLOUT", 1.5))    # ... plus this per unit of (time outside the span / span)
TOL_MAX = float(os.environ.get("FINDER_F5TOLMAX", 3.0))    # ... and never more than this (the index serves 4 arcsec)
EXT = float(os.environ.get("FINDER_F5EXT", 1.2))           # exposures up to this many spans outside the quad's span are looked at
ROWS = int(float(os.environ.get("FINDER_F5ROWS", 3e6)))    # (quad, exposure) predictions per batch


def verify(found, G, MU, AS, deadline, log=None, dump=None):
    """found: {key: quad dict with det (4 kept indices), st (alpha, beta, rates), g, gd, tref}.
    Sets c["n5"] (further nights with a detection on the track) and c["x5"] (those detections, kept indices) on every
    quad that was reached before the deadline; returns the number of quads looked at."""
    dev = G["dev"]
    keys = list(found.keys())
    nq = len(keys)
    if nq == 0:
        return 0
    f8 = torch.float64
    D = torch.as_tensor(np.array([np.asarray(found[k]["det"], dtype=np.int64)[:4] for k in keys]), device=dev)
    st = torch.tensor([found[k]["st"] for k in keys], dtype=f8, device=dev)
    g = torch.tensor([found[k]["g"] for k in keys], dtype=f8, device=dev)
    gd = torch.tensor([found[k]["gd"] for k in keys], dtype=f8, device=dev)
    tref = torch.tensor([found[k]["tref"] for k in keys], dtype=f8, device=dev)
    et = G["et"].double(); ob = G["ob"].double(); nE = len(et)
    nightE = G["night_e"].long(); de = G["de"].long(); th = G["th"]
    fc = torch.as_tensor(G["fc_np"], dtype=f8, device=dev) if "fc_np" in G else None
    fr = torch.as_tensor(G["fr_np"], dtype=f8, device=dev) if "fr_np" in G else None
    tq = et[de[D]]
    t0 = tq.min(1).values; t1 = tq.max(1).values
    span = (t1 - t0).clamp(min=60.0)
    ownn = nightE[de[D]]
    index = G["index"]
    nn = int(nightE.max()) + 1
    ch = max(16, ROWS // max(nE, 1))
    done = 0
    dumps = []
    for s in range(0, nq, ch):
        if time.time() > deadline:
            break
        e = min(nq, s + ch)
        tt = et[None, :] - tref[s:e, None]
        gg = g[s:e, None]; gdd = gd[s:e, None]
        mg3 = MU * gg.clamp(min=1e-3, max=0.2) ** 3
        gdq = gdd.clamp(-1e-4, 1e-4)
        cq = 1 - 0.5 * mg3 * tt ** 2 + 0.5 * mg3 * gdq * tt ** 3
        sq = tt - mg3 * tt ** 3 / 6.0
        rf = 1 - 0.5 * mg3 * tt ** 2 + (mg3 / 3.0) * gdq * tt ** 3
        den = rf + gdd * tt - gg * ob[None, :, 2]
        px = (st[s:e, 0:1] * cq + st[s:e, 2:3] * sq - gg * ob[None, :, 0]) / den
        py = (st[s:e, 1:2] * cq + st[s:e, 3:4] * sq - gg * ob[None, :, 1]) / den
        out = torch.maximum(t0[s:e, None] - et[None, :], et[None, :] - t1[s:e, None]).clamp(min=0.0) / span[s:e, None]
        ok = (out <= EXT) & torch.isfinite(px) & torch.isfinite(py)
        for k in range(4):
            ok &= nightE[None, :] != ownn[s:e, k:k + 1]
        if fc is not None:
            ok &= (px - fc[None, :, 0]) ** 2 + (py - fc[None, :, 1]) ** 2 <= (fr[None, :].clamp(min=0.0)) ** 2
            ok &= fr[None, :] > 0
        qi, ei = torch.nonzero(ok, as_tuple=True)
        done = e
        if len(qi) == 0:
            continue
        qth = torch.stack([px[qi, ei], py[qi, ei]], 1)
        hit = index.query(ei, qth, TOL_MAX * AS)
        m = hit >= 0
        if not bool(m.any()):
            continue
        qi, ei, hit, qth = qi[m], ei[m], hit[m], qth[m]
        d = torch.sqrt(((th[hit].double() - qth) ** 2).sum(1)) / AS
        o_ = out[qi, ei]
        if dump is not None:
            dumps.append(np.stack([(qi + s).cpu().numpy(), hit.cpu().numpy(), d.cpu().numpy(), o_.cpu().numpy(), ei.cpu().numpy()], 1))
        dn = d / (TOL_IN + TOL_OUT * o_).clamp(max=TOL_MAX)
        m = dn <= 1.0
        if not bool(m.any()):
            continue
        qi, ei, hit, dn = qi[m], ei[m], hit[m], dn[m]
        kn = torch.unique(qi * nn + nightE[ei])
        cnt = torch.bincount(kn // nn, minlength=e - s).cpu().numpy()
        dmin = torch.full((e - s,), float("inf"), dtype=f8, device=dev).scatter_reduce_(0, qi, dn, "amin").cpu().numpy()
        qc = qi.cpu().numpy(); hc = hit.cpu().numpy()
        for j in np.flatnonzero(cnt > 0):
            c = found[keys[s + j]]
            c["n5"] = int(cnt[j]); c["d5"] = float(dmin[j]); c["x5"] = hc[qc == j].tolist()
    if dump is not None:
        kn_ = np.asarray(G["keep_np"])
        np.savez(dump, quads=kn_[D.cpu().numpy()], hits=(np.concatenate(dumps) if dumps else np.zeros((0, 5))),
                 hitrows=(kn_[np.concatenate(dumps)[:, 1].astype(np.int64)] if dumps else np.zeros(0, np.int64)),
                 chi=np.array([found[k].get("chi", found[k]["rms"]) for k in keys]), rms=np.array([found[k]["rms"] for k in keys]),
                 span=span.cpu().numpy())
    return done
