"""Compare the observed nights of a short arc with simulated orbits and field contaminants."""
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logit


def frequencies(sequences, code):
    size = int(code.max()) + 1
    one = np.zeros(size)
    two = np.zeros((size, size))
    lengths = np.array(list(map(len, sequences)))
    for length in np.unique(lengths):
        ids = np.flatnonzero(lengths == length)
        group = code[np.stack([sequences[i] for i in ids])]
        one += np.bincount(group.ravel(), minlength=size) / length
        left, right = np.triu_indices(length, 1)
        keys = group[:, left] * size + group[:, right]
        counts = np.bincount(keys.ravel(), minlength=size*size).reshape(size, size)
        two += (counts + counts.T) / (length*(length-1))
    return one / one.sum(), two / two.sum()


def pair_odds(signal, background, code):
    single, pair = signal
    other, noise = frequencies(background, code)
    # The independent marginals give previously unseen pairs finite support.
    marginal = .5*(single+other)
    prior = np.maximum(.38*(pair+noise) + .24*marginal[:, None]*marginal, 1.e-9)
    return pair, noise, prior


def adjust(candidates, exposure, snr, visits, groups, observed):
    nights = visits['night']
    dates = []
    for candidate in candidates:
        rows = np.asarray(candidate['det'], 'i8')
        ordered = exposure[rows[np.argsort(-snr[rows])]]
        _, take = np.unique(nights[ordered], return_index=True)
        dates.append(ordered[take])
    rng = np.random.default_rng(4371)
    signal = []
    for _, family, rows in observed:
        if family >= 4:
            continue
        if len(rows) > 7:
            rows = np.sort(rng.choice(rows, rng.integers(5, 8), replace=False))
        signal.append(rows)
    p = np.array([c['p'] for c in candidates])
    lengths = np.array(list(map(len, dates)))
    background = (lengths >= 5) & (lengths <= 7) & (p <= min(.85, np.quantile(p, .80)))
    result = np.zeros(len(p))
    if len(signal) < 200 or background.sum() < 55:
        return result
    fold = rng.integers(0, 3, len(p))
    months = np.floor((visits['t']-visits['t'].min())*(13./365.25))
    for raw, weight in ((groups, .35), (nights, .30), (months, .35)):
        _, code = np.unique(raw, return_inverse=True)
        source = frequencies(signal, code)
        for j in range(3):
            train = np.flatnonzero(background & (fold != j))
            test = np.flatnonzero((lengths >= 5) & (lengths <= 9) & (fold == j))
            if len(train) < 30 or not len(test):
                continue
            pair, noise, prior = pair_odds(source, [dates[i] for i in train], code)
            mass = 32. + .14*len(train)
            smooth = mass*prior + .023
            normal = mass + .023*pair.size
            odds = np.log((len(signal)*pair+smooth)/(len(signal)+normal))
            odds -= np.log((len(train)*noise+smooth)/(len(train)+normal))
            np.clip(odds, -3.5, 3.5, out=odds)
            for i in test:
                q = code[dates[i]]
                a, b = np.triu_indices(len(q), 1)
                result[i] += weight * odds[q[a], q[b]].mean()
    return result


def apply(G, outdir, deadline):
    sample = (G.get('survey_control') or {}).get('visible_nights')
    if not sample or time.monotonic() >= deadline:
        return 0
    path = Path(outdir)/'candidates.json'
    candidates = json.loads(path.read_text())
    if len(candidates) < 60:
        return 0
    root = Path(G['scen'])
    exposure, snr = [np.load(root/('det_'+name+'.npy'), mmap_mode='r') for name in ('exp', 'snr')]
    evidence = adjust(candidates, exposure, snr, *sample)
    changed = 0
    for c, score in zip(candidates, evidence):
        # The calibration population represents disk and detached bodies.
        # Its date distribution carries little information about very slow sources.
        r = c.get('r', 42.)
        taper = np.clip((420.-r)/210., 0., 1.) if np.isfinite(r) else 0.
        if not taper or not score or c.get('stellar_parallax') or c.get('annual_parallax'):
            continue
        delta = .78*taper*np.clip(score, -2.5, 2.5)
        c['cadence_log_odds'] = round(float(delta), 4)
        c['p'] = float(expit(logit(np.clip(c['p'], 1.e-14, 1.-1.e-14)) + delta))
        changed += 1
    if changed and time.monotonic() < deadline:
        candidates.sort(key=lambda c: -c['p'])
        tmp = path.with_suffix('.cadence')
        tmp.write_text(json.dumps(candidates))
        os.replace(tmp, path)
        return changed
    return 0
