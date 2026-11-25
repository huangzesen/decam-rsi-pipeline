"""Evaluate fitted orbits and check same-night predictions in the original images."""
import ctypes
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logit
import torch

import image_followup as I
import orbit_population

QUERY = np.dtype([('row', 'i4'), ('exposure', 'i4')] +
                 [(k, 'f4') for k in ('dx', 'dy', 'dm', 'error', 'scatter')])
RESULT = np.dtype([(k, 'i4') for k in ('row', 'controls', 'covered')] +
                  [(k, 'f4') for k in ('statistic', 'sky', 'expected', 'dx', 'dy', 'dm')])


def queries(F, sky, items):
    G = F.G
    times, nights = G['et_np'], G['night_e_np']
    neighbours = [np.flatnonzero((nights == n) & (np.arange(len(times)) != i))
                  for i, n in enumerate(nights)]
    owners, records = [], []
    for item in items:
        e, rows = item['e'], item['d']
        # One highest-quality discovery measurement conditions the test on
        # each night. All other visits, including ones omitted from the
        # report during measurement selection, are available for checking.
        anchors = item['take']
        for k in anchors:
            i = e[k]
            visits = neighbours[i]
            if not len(visits):
                continue
            valid = np.abs(times[visits] - times[i]) < .45
            if 'pt_np' in G:
                sep = np.linalg.norm(G['pt_np'][visits] - item['xy'][k], axis=1)
                valid &= sep < np.deg2rad(1.6)
            visits = visits[valid]
            if not len(visits):
                continue
            predicted_mag = item['mag'][k] + item['color'] * (G['lev_e'][visits] - G['lev_e'][i])
            depth = G['m5c_np'][visits]
            margin = np.nan_to_num(depth - predicted_mag, nan=-.25)
            visits = visits[margin > -1.0]
            margin = margin[margin > -1.0]
            if not len(visits):
                continue
            # Limit the influence and cost of a single dense time series.
            score = margin + .26 * np.log1p(144. * abs(times[visits] - times[i]))
            order = np.argsort(-score, kind='stable')[:8]
            visits, margin = visits[order], margin[order]
            ee = np.r_[i, visits]
            records.append((item, k, ee, margin))
    if not records:
        return np.empty(0, QUERY), owners
    parts = []
    for off in range(0, len(records), 256):
        rec = records[off:off+256]
        width = max(len(ee) for _, _, ee, _ in rec)
        ex = np.stack([np.pad(ee, (0, width-len(ee)), mode='edge') for _, _, ee, _ in rec])
        te = torch.as_tensor(ex, device=G['dev'])
        state = torch.as_tensor(np.stack([v['x'] for v, _, _, _ in rec]), device=G['dev'])
        ref = torch.as_tensor([float(v['ref']) for v, _, _, _ in rec], dtype=torch.float64, device=G['dev'])
        pred = F.kmodel(state, G['et'][te] - ref[:, None], G['ob'][te]).cpu().numpy()
        for (v, k, ee, margin), positions in zip(rec, pred):
            shift = (positions[1:len(ee)] - positions[:1]) / F.AS
            if not np.isfinite(shift).all() or np.max(np.abs(shift)) > 110.:
                continue
            visit = ee[1:]
            dl = G['lev_e'][visit] - G['lev_e'][ee[0]]
            sn = max(4.5, float(v['snr'][k]))
            predicted_sn = np.clip(5. * 10.**(.4 * margin), 4., 60.)
            q = np.empty(len(visit), QUERY)
            q['row'] = v['d'][k]
            q['exposure'] = visit
            q['dx'], q['dy'] = shift.T
            q['dm'] = v['color'] * dl
            q['error'] = np.sqrt(.12**2 + (1.08/sn)**2 + (1.08/predicted_sn)**2 +
                                 (.045 * np.linalg.norm(shift, axis=1))**2)
            q['scatter'] = np.sqrt(.13**2 + (1.09/sn)**2 + (1.09/predicted_sn)**2 + (.17*dl)**2)
            parts.append(q)
            owners.extend((v, k, int(j), float(m)) for j, m in zip(visit, margin))
    return np.concatenate(parts) if parts else np.empty(0, QUERY), owners


def evaluate(F, candidates, sky, deadline):
    G = F.G
    cat = I.arrays(G['scen'])
    orbits = dict(G.get('survey_orbits', {}))
    orbits.update(G.get('report_orbits', {}))
    ordered = sorted(candidates, key=lambda c: -c['p'])[:1000]
    copies = []
    for c in ordered:
        key = tuple(sorted(c['det']))
        if len(key) > 48 or c.get('nights', 5) > 20:
            continue
        copy = dict(c, _parent=c)
        if key in orbits:
            copy['_orbit'] = orbits[key]
        blended = G.get('report_components', {}).get(key)
        if blended:
            copy['_components'] = blended
        copies.append(copy)
    items = I.prepare(F, copies, cat)
    if sky is not None:
        classes = np.zeros(len(sky.exposure), 'u1')
        if 'keep_np' in G:
            classes[G['keep_np'][:G['nk']]] = 1
        fn = sky.lib.companion_measurements
        p, i = ctypes.c_void_p, ctypes.c_int
        fn.argtypes = [p, p, i, p, i, ctypes.c_double, p]
        fn.restype = i
    tested, added, fitted = 0, 0, 0
    for off in range(0, len(items), 96):
        if time.monotonic() + .45 >= deadline:
            break
        group = items[off:off+96]
        I.model(F, group)
        population = orbit_population.evidence(F, group)
        for v, value in zip(group, population):
            if np.isfinite(value):
                c = v['candidate']['_parent']
                c['p'] = float(expit(logit(np.clip(c['p'], 1.e-14, 1.-1.e-14)) + .90 * value))
                c['population_log_odds'] = round(float(.90*value), 3)
        fitted += len(group)
        if sky is None:
            continue
        q, owners = queries(F, sky, [v for v in group if G['companion_visits'][v['e']].any()])
        if not len(q):
            continue
        hit = np.zeros(len(q), RESULT)
        n = fn(sky.handle, q.ctypes.data, len(q), classes.ctypes.data, 2,
               max(.02, deadline-time.monotonic()), hit.ctypes.data)
        if n < 0:
            raise RuntimeError('same-night catalogue lookup failed')
        tested += n
        by_night = {}
        for (v, k, visit, margin), obs, a in zip(owners[:n], hit[:n], q[:n]):
            if obs['row'] < 0 or obs['controls'] < 10:
                continue
            snr = max(4.5, float(v['snr'][k]))
            sep = np.hypot(a['dx'], a['dy'])
            # The original score includes resolved motion and often also
            # includes simple repeat counts. Compare the observed statistic
            # to the local null, discounting that shared evidence.
            s = float(obs['statistic'])
            ratio = s / max(.004, float(obs['sky']))
            contrast = np.log(max(.20, ratio))
            weight = min(1., obs['controls'] / 30.)
            reliability = float(expit((margin+.12)/.29))
            value = weight * (np.log1p(reliability * max(0., ratio-1.)) -
                              .70 - 1.35*min(1., sep/max(.25, 2.*a['error'])))
            if value <= 0.:
                continue
            c = v['candidate']['_parent']
            night = int(G['night_e_np'][visit])
            result = by_night.setdefault((id(c), night), [])
            result.append((float(value), int(obs['row']), s, contrast, sep, snr, visit))
        for v in group:
            c = v['candidate']['_parent']
            votes, extras = [], []
            existing = set(map(int, v['e']))
            for night in np.unique(G['night_e_np'][v['e']]):
                matches = by_night.get((id(c), int(night)))
                if not matches:
                    continue
                matches.sort(reverse=True)
                votes.append(matches[0][0])
                # Keep an omitted measurement only when the local controls
                # and the measured colour and relative position all agree.
                for _, row, stat, contrast, sep, snr, visit in matches[:2]:
                    if visit not in existing and stat > .41 and contrast > 2.65 and sky.arrays[0][row] >= 4.8:
                        extras.append(row)
                        existing.add(visit)
            if not votes:
                continue
            shift = .72 * min(8., sum(sorted(votes, reverse=True)[:4]))
            c['paired_log_odds'] = round(shift, 3)
            c['p'] = float(expit(logit(np.clip(c['p'], 1.e-14, 1.-1.e-14)) + shift))
            if extras:
                c['det'] = sorted(set(c['det']).union(extras))
                c['paired_added'] = len(extras)
                added += len(extras)
    return tested, added, fitted


def run(F, outdir, deadline):
    G = F.G
    sky = (G.get('survey_control') or {}).get('original_sky')
    if sky is not None and not hasattr(sky.lib, 'companion_measurements'):
        sky = None
    path = Path(outdir) / 'candidates.json'
    candidates = json.loads(path.read_text())
    if not candidates:
        return
    _, labels, counts = np.unique(G['night_e_np'], return_inverse=True, return_counts=True)
    G['companion_visits'] = counts[labels] > 1
    G['ob_np_follow'] = G['ob'].cpu().numpy()
    tested, added, fitted = evaluate(F, candidates, sky, deadline)
    if fitted:
        candidates.sort(key=lambda c: -c['p'])
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(candidates))
        os.replace(temp, path)
        F.log('population evidence:', fitted, 'orbits; companion images:', tested, 'apertures,', added, 'additional measurements')
