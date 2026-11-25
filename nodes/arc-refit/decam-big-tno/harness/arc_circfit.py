#!/usr/bin/env python3
"""Assumed-circular distance range and H for a card whose own places fix no orbit (2026-11-23, asked by TRKSUB01 after the
u20drop review: ten u20 cards keep one season of a few nights; harness/arc_refit_nolone.py --drop-mjd finds no bound orbit or
one whose "now" place is uncertain by degrees, so build.py shows no orbit and no H).

A few nights still fix the distance roughly through the parallax of Earth's motion. For each r on a grid (15-60 au step 1,
60-100 step 2, 100-200 step 5) fit a circular orbit of radius r: free parameters are the direction of the body at the epoch
(2 angles) and the direction of its motion (1 angle), i.e. orbit plane (inclination, node) and phase; the speed is the
circular speed at r. The orbit is about the solar-system barycentre with the pipeline fitter's mass (Sun + planets,
planet-search/linker/orbit_fit.py GMT), observer = DECam at CTIO (orbit_fit.observer_bary "W84", topocentric), light time
included; weights and the rms (arcsec, per coordinate) as in lead_check.fit_lead. Barycentric and heliocentric r differ by
<= 0.01 au. The allowed range is every r with rms <= max(1.5 x best rms, best rms + 0.05"). The rms curve can have narrow
valleys (on arcs of weeks the circular r is pinned to < 0.01 au) and two branches (away from opposition), so every local
minimum of the grid is refined (golden section) and every in/out transition is bisected; "intervals" lists the allowed
pieces, r_lo / r_hi are the outermost ends.

Sanity test (2026-11-23, C### e 0.17 and C### e 0.01): the circular range is a precision range under the assumption; it
misses the fitted r of an eccentric body (C###: 43.0-43.8 against 45.16 on its 27-day arc; on 1-2 day pieces the circular
r can be off by more). So the same scan is also run for bound orbits with e <= E_MAX (default 0.3; free radial and
transverse speed through e and true anomaly at the epoch, 5 parameters) and written as "bound_emax": that range holds the
true r of any body with e <= E_MAX.

Short arcs (2-4 nights near opposition) also fit a RETROGRADE circle at ~1.35 x the prograde r (the reflex rate
(v_Earth + v_body) / Delta matches (v_Earth - v_body) / Delta further out; checked: i 175-178 deg on that branch). Retrograde
TNOs are very rare, so the card's range (top-level fields) and bound_emax are prograde (ecliptic i < 90 deg);
"circular_any_i" keeps the scan over all inclinations. fit_ok is false when neither the circle nor e <= E_MAX fits to 0.30"
(the arc_extend joint-fit limit): then the places are not one bound body and the range means nothing.

H at each r uses the card H formula of lead_check.py (median over detections of V - 5 log10(r_h Delta) - 0.15 phase angle;
V from magzero, flux and the band offsets) with r_h and Delta from that r's fit. a, e and i are NOT determined.

Places: the kept rows of runs/<RUN>/arc/<id>.drop.json (= its base file's rows minus "dropped"), read on the scenario host as
arc_refit_nolone.py does (arc_extend.remote; DBT_SITE=nersc for the 5 % search regions). Writes runs/<RUN>/arc/<id>.circ.json
unless --print-only or --mjd-range (tests). Never touches any other file.
Usage: DBT_SITE=nersc OMP_NUM_THREADS=1 ~/anaconda3/bin/python -I harness/arc_circfit.py RUN ID_SUBSTRING [...] [--print-only]
       [--mjd-range LO,HI] [--emax 0.3]
"""
import argparse, datetime, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "harness"))
import lead_check as LC, arc_extend as AX                       # noqa: E402
from lead_check import O, radec_to_unit, ARCSEC                # noqa: E402
from astropy.time import Time                                  # noqa: E402
from astropy.coordinates import get_body_barycentric           # noqa: E402

GRID = list(range(15, 61)) + list(range(62, 101, 2)) + list(range(105, 201, 5))
NOTE = ("ASSUMED CIRCULAR orbit: r is the distance at which a prograde (i < 90 deg) circular orbit (barycentric, Sun + planets "
        "mass, CTIO observer, light time) fits the kept places; range = r with rms <= max(1.5 x best, best + 0.05\"); H from lead_check's formula at "
        "that r. a, e, i are not determined. The circular range is a precision range under the assumption: an eccentric body "
        "can lie outside it (see bound_emax).")


def frame(lon, lat):
    cl, sl, cb, sb = np.cos(lon), np.sin(lon), np.cos(lat), np.sin(lat)
    return np.array([cb * cl, cb * sl, sb]), np.array([-sl, cl, 0.0]), np.array([-sb * cl, -sb * sl, cb])


def state(q, r):
    """Barycentric state (au, au/d) at the epoch, distance r. q = (lon, lat, psi) circular, or (lon, lat, psi, e, f): e and
    true anomaly f set the radial and transverse speed (semi-latus rectum p = r (1 + e cos f))."""
    w, e_, n_ = frame(q[0], q[1]); vh = np.cos(q[2]) * e_ + np.sin(q[2]) * n_
    ecc, f = (q[3], q[4]) if len(q) > 3 else (0.0, 0.0)
    p = r * (1 + ecc * np.cos(f)); h = np.sqrt(O.GMT * p)
    return np.concatenate([r * w, (O.GMT / h) * ecc * np.sin(f) * w + (h / r) * vh])


def resid(q, r, t0, t, R, uo, e_, n_, s2):
    d = O.directions(state(q, r), t0, t, R) - uo
    return np.concatenate([(d * e_).sum(1), (d * n_).sum(1)]) / ARCSEC / s2


def incl(q, r):
    """Ecliptic inclination (deg) of the barycentric orbit."""
    s = state(q, r); h = np.cross(O.eq_to_ecl(s[:3]), O.eq_to_ecl(s[3:]))
    return float(np.degrees(np.arccos(np.clip(h[2] / np.linalg.norm(h), -1, 1))))


def fit_r(r, ctx, warm, base=None, emax=0.0, prograde=False):
    """Best fit at distance r: (cost, rms arcsec, q). Circular (emax = 0): starts at the body's direction from the observer
    nearest the epoch at distance r with 8 directions of motion, plus warm. Eccentric: starts from the circular solution at r
    (base) with e in {1e-3, emax/2, 0.95 emax} x f in {0, 90, 180, 270} deg, plus warm. prograde: keep only solutions with
    ecliptic i < 90 deg (short arcs also fit a retrograde circle at ~1.35 x the prograde r; 2026-11-23)."""
    t0, t, R, uo, sig = ctx["t0"], ctx["t"], ctx["R"], ctx["uo"], ctx["sig"]
    e_, n_ = O.tangent(uo); s2 = np.concatenate([sig, sig])
    if emax <= 0:
        k = int(np.argmin(abs(t - t0))); Rk, uk = R[k], uo[k]
        b = Rk @ uk; rho = -b + np.sqrt(max(b * b - Rk @ Rk + r * r, 0.0)); x0 = (Rk + rho * uk) / r
        lon, lat = np.arctan2(x0[1], x0[0]), np.arcsin(np.clip(x0[2], -1, 1))
        starts = [np.array([lon, lat, psi]) for psi in np.radians(np.arange(0, 360, 45))]
        kw = dict(method="lm", xtol=1e-13, ftol=1e-13)
    else:
        starts = [np.array([base[0], base[1], base[2], e0, f0]) for e0 in (1e-3, 0.5 * emax, 0.95 * emax)
                  for f0 in np.radians([0, 90, 180, 270])]
        lb = [-np.inf, -np.pi / 2, -np.inf, 0.0, -np.inf]; ub = [np.inf, np.pi / 2, np.inf, emax, np.inf]
        kw = dict(method="trf", bounds=(lb, ub), xtol=1e-12, ftol=1e-12, diff_step=1e-7)
    if warm is not None:
        starts.append(warm)
    best = None
    for q0 in starts:
        q0 = np.array(q0, float)
        if emax > 0:
            q0[1] = np.clip(q0[1], -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6); q0[3] = np.clip(q0[3], 1e-6, emax - 1e-6)
        try:
            res = least_squares(resid, q0, args=(r, t0, t, R, uo, e_, n_, s2), **kw)
        except Exception:                                          # noqa: BLE001
            continue
        rms = float(np.sqrt(((res.fun * s2) ** 2).mean()))
        if prograde and incl(res.x, r) >= 90:
            continue
        if best is None or res.cost < best[0]:
            best = (float(res.cost), rms, res.x)
    return best


def H_at(q, r, ctx):
    """lead_check.lead_fit's H: median over detections of V - 5 log10(r_h Delta) - 0.15 alpha (no light time, as there)."""
    s = state(q, r); p = O.propagate(s[:3], s[3:], ctx["t"] - ctx["t0"]); R, sun = ctx["R"], ctx["sun"]
    rh = np.linalg.norm(p - sun, axis=1); delta = np.linalg.norm(p - R, axis=1); es = np.linalg.norm(R - sun, axis=1)
    alpha = np.degrees(np.arccos(np.clip((rh ** 2 + delta ** 2 - es ** 2) / (2 * rh * delta), -1, 1)))
    H = ctx["V"] - 5 * np.log10(rh * delta) - 0.15 * alpha
    return float(np.nanmedian(H)) if np.isfinite(H).any() else None


def scan(ctx, emax=0.0, circ=None, prograde=False):
    """rms(r) over GRID, every local minimum refined, every in/out transition bisected. Returns the summary dict."""
    S = {}                                                      # r -> (rms, q)

    def base_at(r):                                             # circular solution nearest r (eccentric starts)
        return None if circ is None else circ[min(circ, key=lambda x: abs(x - r))][1]

    def ev(r, warm):
        r = round(float(r), 4)
        if r not in S:
            b = fit_r(r, ctx, warm, base_at(r), emax, prograde)
            if b is not None:
                S[r] = (b[1], b[2])
        return S.get(r)
    warm = None
    for r in GRID:
        b = ev(r, warm)
        warm = b[1] if b else warm
    if not S:
        return None
    rg = sorted(S); rm = [S[r][0] for r in rg]
    mins = [k for k in range(len(rg)) if (k == 0 or rm[k] <= rm[k - 1]) and (k == len(rg) - 1 or rm[k] <= rm[k + 1])]
    for k in mins:                                              # golden section between the neighbours of each grid minimum
        lo_b, hi_b = rg[max(k - 1, 0)], rg[min(k + 1, len(rg) - 1)]
        for _ in range(30):
            if hi_b - lo_b < 0.005:
                break
            m1, m2 = hi_b - 0.618 * (hi_b - lo_b), lo_b + 0.618 * (hi_b - lo_b)
            f1, f2 = ev(m1, S[rg[k]][1]), ev(m2, S[rg[k]][1])
            if f1 is None or f2 is None:
                break
            if f1[0] <= f2[0]:
                hi_b = m2
            else:
                lo_b = m1
    r_best = min(S, key=lambda x: S[x][0]); best = S[r_best][0]; lim = max(1.5 * best, best + 0.05)
    rg = sorted(S)
    for r_a, r_b in zip(rg[:-1], rg[1:]):                       # bisect each crossing of the limit
        ia, ib = S[r_a][0] <= lim, S[r_b][0] <= lim
        if ia == ib:
            continue
        r_in, r_out = (r_a, r_b) if ia else (r_b, r_a)
        for _ in range(12):
            if abs(r_out - r_in) < 0.01:
                break
            m = round(0.5 * (r_in + r_out), 4); b = ev(m, S[r_in][1])
            if b is not None and b[0] <= lim:
                r_in = m
            else:
                r_out = m
    rg = sorted(S); iv, cur = [], None
    for r in rg:
        if S[r][0] <= lim:
            cur = [r, r] if cur is None else [cur[0], r]
        elif cur is not None:
            iv.append(cur); cur = None
    if cur is not None:
        iv.append(cur)
    lo, hi = iv[0][0], iv[-1][1]
    H = lambda r: (lambda h: None if h is None else round(h, 2))(H_at(S[r][1], r, ctx))
    return dict(r_best=round(r_best, 2), r_lo=round(lo, 2), r_hi=round(hi, 2), edge=bool(lo <= GRID[0] or hi >= GRID[-1]),
                intervals=[[round(a, 2), round(b, 2)] for a, b in iv], rms_best=round(best, 3), rms_limit=round(lim, 3),
                H_best=H(r_best), H_at_r_lo=H(lo), H_at_r_hi=H(hi),
                grid=[[float(r), round(S[r][0], 4)] for r in rg], _S=S)


def load(RUN, run, sub):
    fs = [f for f in glob.glob(os.path.join(RUN, "arc", f"dbt-{run}-*.json"))
          if sub in os.path.basename(f) and not f.endswith((".refit.json", ".nolone.json", ".drop.json", ".circ.json"))]
    if len(fs) != 1:
        sys.exit(f"{sub}: {len(fs)} arc files match")
    i = json.load(open(fs[0]))["id"]
    dp = os.path.join(RUN, "arc", i + ".drop.json")
    if not os.path.exists(dp):
        sys.exit(f"{i}: no .drop.json")
    Dr = json.load(open(dp)); B = json.load(open(os.path.join(RUN, "arc", Dr["base_file"])))
    gone = {int(d["row"]) for d in Dr["dropped"]}
    rows = [int(r) for r in B["rows"] if int(r) not in gone]
    notes = [] if rows == [int(r) for r in Dr["rows"]] else ["kept rows differ from the .drop.json rows list"]
    return i, rows, Dr, notes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("ids", nargs="+")
    ap.add_argument("--print-only", action="store_true", help="do not write .circ.json (sanity tests on cards with an orbit)")
    ap.add_argument("--mjd-range", default="", help="LO,HI: test only, keep places inside this UTC MJD window (implies --print-only)")
    ap.add_argument("--emax", type=float, default=0.3, help="eccentricity limit of the bound_emax scan (0 = skip it)")
    ap.add_argument("--places-json", default="", help="one lead with no arc file: its places (mjd UTC, ra, dec, mag, band, snr); ids = the lead id")
    ap.add_argument("--rows-json", default="", help="test only: a saved arc_extend.remote reply instead of reading the host")
    a = ap.parse_args()
    a.print_only = a.print_only or bool(a.mjd_range)
    RUN = os.path.join(HERE, "runs", a.run); cache = os.path.join(RUN, "lead_check")
    if a.places_json:                                           # a lead with no arc file (C###, a split-off season, 2026-11-23)
        P = json.load(open(a.places_json)); P = P["places"] if isinstance(P, dict) else P
        i = a.ids[0]; t_utc = np.array([float(x["mjd"]) for x in P]); ra = np.array([float(x["ra"]) for x in P])
        dec = np.array([float(x["dec"]) for x in P]); snr = np.array([float(x.get("snr") or 5) for x in P])
        sig = np.maximum(0.08 + 0.6 / np.maximum(snr, 3.0), 0.1)
        V = np.array([float(x["mag"]) - LC.COL.get(x.get("band"), 0.0) for x in P])
        solve(a, RUN, i, t_utc, ra, dec, sig, V, [f"places from {os.path.basename(a.places_json)}"], {}, list(range(len(P))))
        return
    led = {x["id"]: x for x in json.load(open(os.path.join(RUN, "leads_sent.json")))}
    cards = [load(RUN, a.run, s) for s in a.ids]
    q = {}
    for i, rows, _, _ in cards:
        q.setdefault(led[i]["sid"], {"rows": []})["rows"] += rows
    REM = json.load(open(a.rows_json)) if a.rows_json else AX.remote(q)   # one read on the scenario host for all cards
    for i, rows, Dr, notes in cards:
        sid = led[i]["sid"]; E = LC.exposures(sid, cache); allt = np.array([e["mjd_mid"] for e in E])
        D = {int(k): v for k, v in REM[sid]["rows"].items()}
        if a.mjd_range:                                         # test only: keep the places inside a UTC MJD window
            w0, w1 = (float(v) for v in a.mjd_range.split(","))
            rows = [r for r in rows if w0 <= allt[int(D[r][2])] <= w1]
            notes.append(f"TEST: places restricted to MJD {w0}-{w1}")
        M = np.array([D[r] for r in rows]); ex = M[:, 2].astype(int)
        t_utc, ra, dec = allt[ex], M[:, 0], M[:, 1]
        sig = np.maximum(0.08 + 0.6 / np.maximum(M[:, 5], 3.0), 0.1)
        mag = np.array([E[e]["magzero"] if E[e].get("magzero") is not None else np.nan for e in ex], float) \
            - 2.5 * np.log10(np.maximum(M[:, 3], 1e-9))
        V = mag - np.array([LC.COL.get(E[e]["band"], 0.0) for e in ex])
        solve(a, RUN, i, t_utc, ra, dec, sig, V, notes, Dr, rows)


def solve(a, RUN, i, t_utc, ra, dec, sig, V, notes, Dr, rows):
    """the scans and the .circ.json for one card (split out of main 2026-11-23 for --places-json)"""
    R, t = O.observer_bary(t_utc, "W84"); uo = radec_to_unit(np.radians(ra), np.radians(dec)); t0 = float(np.median(t))
    sun = get_body_barycentric("sun", Time(t, format="mjd", scale="tdb")).xyz.to("au").value.T
    ctx = dict(t0=t0, t=t, R=R, uo=uo, sig=sig, sun=sun, V=V)
    k = int(np.argmin(abs(t - t0))); su = sun[k] - R[k]
    elong = float(np.degrees(np.arccos(np.clip(su @ uo[k] / np.linalg.norm(su), -1, 1))))
    C = scan(ctx, prograde=True)                            # the card's range: circular, prograde (i < 90 deg)
    if C is None:
        print(i, "no prograde circular solution at any r", flush=True); return
    SC = C.pop("_S")
    C["H_grid"] = [[r, (lambda h: None if h is None else round(h, 2))(H_at(SC[r][1], r, ctx))] for r, _ in C["grid"]
                   if float(r) in GRID]
    if len(C["intervals"]) > 1:
        notes.append(f"circular prograde: {len(C['intervals'])} separate allowed intervals {C['intervals']}")
    CA = scan(ctx); CA.pop("_S")                            # any inclination: shows the retrograde branch
    CA = dict(note="circular, any inclination; intervals above the prograde range are retrograde (i > 90 deg) circles", **CA)
    Bm = None
    if a.emax > 0:
        Bm = scan(ctx, a.emax, SC, prograde=True)
        if Bm is not None:
            Bm.pop("_S"); Bm = dict(e_max=a.emax, **Bm)
            Bm["note"] = (f"bound prograde orbits with e <= {a.emax} (barycentric e; free radial and transverse speed): holds the "
                          f"true r of any prograde body with e <= {a.emax}; same rms criterion")
    ok = C["rms_best"] <= 0.30 or (Bm is not None and Bm["rms_best"] <= 0.30)
    if not ok:
        notes.append(f"NO FIT: best circular rms {C['rms_best']}\"" + (f", best e <= {a.emax} rms {Bm['rms_best']}\"" if Bm else "")
                     + " (> 0.30\"): the kept places do not fit one bound orbit, so the range means nothing")
    nights = int(len(np.unique(np.floor(t_utc - 0.5))))
    out = dict(id=i, made=datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
               assumed="circular orbit, prograde (i < 90 deg)", fit_ok=bool(ok),
               n=len(rows), nights=nights, arc_days=round(float(t_utc.max() - t_utc.min()), 2), elong_deg=round(elong, 1),
               **{k_: C[k_] for k_ in ("r_best", "r_lo", "r_hi", "edge", "rms_best", "H_best", "H_at_r_lo", "H_at_r_hi",
                                       "intervals", "rms_limit", "grid", "H_grid")},
               note=NOTE + (" " + "; ".join(notes) if notes else ""), bound_emax=Bm, circular_any_i=CA,
               drop_fit=dict(fit=Dr.get("fit"), r=Dr.get("r"), a=Dr.get("a"), e=Dr.get("e"), rms=Dr.get("rms"),
                             grid_fit=Dr.get("grid_fit")))
    line = (f"{i}: n {out['n']} / {nights} nights / {out['arc_days']} d, elong {out['elong_deg']}; CIRC-PRO r_best {C['r_best']:g} "
            f"(rms {C['rms_best']}\"), range {C['r_lo']:g}-{C['r_hi']:g} {C['intervals'] if len(C['intervals']) > 1 else ''}"
            f"{' EDGE' if C['edge'] else ''}; H {C['H_at_r_lo']} / {C['H_best']} / {C['H_at_r_hi']}"
            f" | CIRC-ANY {CA['intervals']}")
    if Bm:
        line += (f" | e<={a.emax} pro: r_best {Bm['r_best']:g} (rms {Bm['rms_best']}\"), range {Bm['r_lo']:g}-{Bm['r_hi']:g}"
                 f" {Bm['intervals'] if len(Bm['intervals']) > 1 else ''}"
                 f"{' EDGE' if Bm['edge'] else ''}; H {Bm['H_at_r_lo']} / {Bm['H_best']} / {Bm['H_at_r_hi']}")
    line += f" | drop fit r {Dr.get('r')} e {Dr.get('e')} rms {Dr.get('rms')}" + (f" NOTE {notes}" if notes else "")
    print(line, flush=True)
    if not a.print_only:
        p = os.path.join(RUN, "arc", i + ".circ.json")
        json.dump(out, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)



if __name__ == "__main__":
    main()
