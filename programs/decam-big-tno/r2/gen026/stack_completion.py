"""Follow resolved, repeated motion across the complete observing baseline.

Two densely sampled nights measure a real angular trajectory but leave range
and radial speed uncertain. Search that orbit family, then require independent
nights and an exact Kepler fit using the ordinary carry and confidence rules.
"""
import time
import numpy as np


def complete(F, deadline):
    G = F.G
    groups = G.get('stack_groups', [])
    if not groups or time.time() >= deadline:
        return []
    started = time.time()
    seeds = F.stack_seeds(groups, G['keep_np'][:G['nk']], G['th_np'],
                         G['de_np'], G['et_np'], G['ob'].cpu().numpy(),
                         G['night_e_np'])
    if not seeds:
        return []
    x = np.asarray([s['x'] for s in seeds])
    a, v, g = x[:, :2], x[:, 3:5], x[:, 2]
    radius_factor = np.sqrt(1 + np.sum(a*a, axis=1))
    # Minimize physical speed over the unknown radial component. The apparent
    # angular rate includes -alpha * radial_rate; this minimization avoids
    # rejecting inclined tracks far from the tangent point. A margin covers
    # short-arc astrometric uncertainty and the approximate initialization.
    minimum_speed2 = np.sum(v*v, axis=1) - np.sum(a*v, axis=1)**2 / radius_factor**2
    allowed = minimum_speed2 * radius_factor < 2.2 * F.MU * g**3
    seeds = [s for s, ok in zip(seeds, allowed) if ok]
    if not seeds:
        return []
    # Interleave groups and cover coarse range samples before finer ones, so a
    # timed prefix does not search only the nearest distances of the first arc.
    by_group = {}
    for s in seeds:
        by_group.setdefault(s['grp'], []).append(s)
    ordered = []
    for values in by_group.values():
        queue = [(0, len(values))]
        result = []
        for lo, hi in queue:
            if lo >= hi:
                continue
            mid = (lo + hi) // 2
            result.append(values[mid])
            queue.extend(((lo, mid), (mid + 1, hi)))
        ordered.append(result)
    seeds = [v[i] for i in range(max(map(len, ordered))) for v in ordered if i < len(v)]
    F.log('resolved stack completion:', len(seeds), 'physical range hypotheses from', len(groups), 'groups')
    reach = float(np.ptp(G['et_np']) + 1.)
    raw = F.carry_safe(seeds, started + .80*max(0., deadline-started),
                       reach=reach, fixg=True, gpr=F.NSTK_GPR, near_deg=2.6)
    G['stack_trace'] = [G['keep_np'][np.asarray(c['det'], dtype=np.int64)].tolist() for c in raw[:20000]]
    good = []
    if raw:
        for L in F.grow(F.merge(raw), deadline):
            e = np.asarray(L['exps']); x = np.asarray(L['x']); d = np.asarray(L['det'])
            nn = len(np.unique(G['night_e_np'][e]))
            rms = np.sqrt(np.mean(np.asarray(L['res'])**2))
            if nn < 5 or not np.isfinite(rms) or rms >= .35:
                continue
            if (not np.isfinite(x).all() or x[2] <= 0 or
                np.sum(x[3:6]**2)*np.sqrt(1+np.sum(x[:2]**2)) >= 1.99*F.MU*x[2]**3):
                continue
            good.append(dict(L, nn=nn, nax=int((d >= G['nk']).sum()),
                             nn0=len(np.unique(G['night_e_np'][e[d < G['nk']]]))))
    G.setdefault('aux_completed', []).extend(good)
    F.log('resolved stack completion:', len(good), 'verified arcs;', round(time.time()-started, 1), 's')
    return good
