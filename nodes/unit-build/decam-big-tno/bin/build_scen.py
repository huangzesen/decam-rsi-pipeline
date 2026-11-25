#!/usr/bin/env python3
"""decam-big-tno: one scenario (a 5 % search region, all years) from the archive5 detection lists, with planted bodies,
and its sealed truth. Runs on Vista next to the data.

Scenario folder SCEN/ (what the program reads; no truth):
  det_ra.npy det_dec.npy (float64 deg), det_flux.npy det_ferr.npy det_snr.npy det_sharp.npy det_x.npy det_y.npy (float32),
  det_ccd.npy (int16), det_flag.npy (uint8), det_exp.npy (int32, row of the exposure table); rows sorted by exposure,
  shuffled within each exposure, planted rows mixed in.
  exposures.json: list of {stem, mjd_mid, band, exptime, magzero, fwhm, airmass, ra0, dec0, obs_xyz (heliocentric CTIO,
  equatorial au)}; region.json: {centre, radius, seasons}.
Truth TRUTH/<sid>.npz: det_body (int32: -1, known body k, or 100000 + planted k), known table (name, H, weight),
  planted table (a, e, i, H, r at mid-span, weight), and per body the nights with a detection.

Known bodies: Horizons positions (archive5/link/known/<rid>.json), detections within 2" in that exposure.
Planted: --planted N bodies; a third each at 30-50, 50-100 and 100-1,000 au (log-uniform within); V 20.5-23.6 at a random epoch of the data, H from it (<= 7.5); e < 0.6, q >= 25 au; i 0-40 deg; placed so they sit
inside the region at a random epoch of the data; two-body + light time; seen when inside 1.05 deg of an exposure's centre
(8 % lost to chip gaps) and S/N >= 4.5, with S/N = 5 x 10^(-0.4 (m - m5)), m5 = the exposure's 5-sigma point-source
magnitude measured from its detections, observed S/N = true + N(0, 1); astrometric noise 0.08" + 0.6"/S/N; sharp and flag
copied from a real known-body detection of similar S/N (donor); colours V-band: g +0.6, r -0.15, i -0.4, z -0.5, Y -0.55,
VR +0.1. Planted weight 1 (< 50 au), 2 (50-100), 4 (> 100); known weight 2 for H <= 7, 1 above.
Usage: python3 -I build_scen.py RID SID STEMS.json KNOWN.json SCEN TRUTH DET [DET ...] [--planted 60] [--seed N]
Sample (unlabelled, for the agents): KNOWN = a missing file, TRUTH = a scratch folder deleted afterwards,
  --exclude-rid of the sealed regions and --box RA0 RA1 DEC0 DEC1 to keep one part of the region.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

AS = np.pi / 180 / 3600; GM = 2.959122082855911e-4; C_AU = 173.1446326846693; EPS = np.radians(23.4392911)
ap = argparse.ArgumentParser()
ap.add_argument("rid"); ap.add_argument("sid"); ap.add_argument("stems"); ap.add_argument("known")
ap.add_argument("scen"); ap.add_argument("truth"); ap.add_argument("det", nargs="+")
ap.add_argument("--planted", type=int, default=0)   # 0 since 2026-11-12
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exclude-rid", default="")   # sample: drop exposures that belong to these regions (comma list)
ap.add_argument("--box", type=float, nargs=4, default=None, metavar=("RA0", "RA1", "DEC0", "DEC1"))   # sample: pointing centres inside
a = ap.parse_args()
t0 = time.time(); say = lambda *x: print(f"[{time.time() - t0:7.1f}s]", *x, flush=True)
rng = np.random.default_rng(a.seed or int(hashlib.sha1(a.sid.encode()).hexdigest()[:8], 16))   # str hash() changes per process
allst = json.load(open(a.stems)); stems = allst[a.rid] if isinstance(allst, dict) else allst
if a.exclude_rid:
    drop = set().union(*(set(allst[r]) for r in a.exclude_rid.split(",")))
    say(f"excluding {sum(s in drop for s in stems)} exposures shared with {a.exclude_rid}"); stems = [s for s in stems if s not in drop]

# ---- load detection lists
C = ("ra", "dec", "flux", "ferr", "snr", "sharp", "x", "y", "ccd", "flag")
cols = {k: [] for k in C}; ex = []; E = []
for s in stems:
    p = next((os.path.join(d, s[4:10], s + ".npz") for d in a.det if os.path.exists(os.path.join(d, s[4:10], s + ".npz"))), None)
    if p is None:
        continue
    z = np.load(p); m = json.loads(str(z["meta"]))
    if m["n"] == 0 or (a.box and not (a.box[0] <= m["ra0"] <= a.box[1] and a.box[2] <= m["dec0"] <= a.box[3])):
        continue
    k = len(E); E.append(m)
    for c in C:
        cols[c].append(z[c])
    ex.append(np.full(len(z["ra"]), k, np.int32))
D = {c: np.concatenate(v) for c, v in cols.items()}; D["exp"] = np.concatenate(ex)
ne, nd = len(E), len(D["ra"])
say(f"{a.rid}: {ne} exposures, {nd:,} detections")
# pointing: the detection lists carry NaN ra0/dec0 (header parse), so use the mean direction of each exposure's detections
_r, _d = np.radians(D["ra"]), np.radians(D["dec"])
_acc = np.stack([np.bincount(D["exp"], weights=w, minlength=ne) for w in (np.cos(_d) * np.cos(_r), np.cos(_d) * np.sin(_r), np.sin(_d))], 1)
_acc /= np.linalg.norm(_acc, axis=1, keepdims=True); del _r, _d
for k, m in enumerate(E):
    if not (np.isfinite(m.get("ra0", np.nan)) and np.isfinite(m.get("dec0", np.nan))):
        m["ra0"] = round(float(np.degrees(np.arctan2(_acc[k, 1], _acc[k, 0])) % 360), 5); m["dec0"] = round(float(np.degrees(np.arcsin(_acc[k, 2]))), 5)
et = np.array([m["mjd_mid"] for m in E]); night = np.floor(et - 0.5).astype(np.int64)
band = [m["band"].split()[0] for m in E]

# observer positions
from astropy.time import Time
from astropy.utils import iers
iers.conf.auto_download = False; iers.conf.iers_degraded_accuracy = "warn"
from astropy.coordinates import get_body_barycentric, EarthLocation
import astropy.units as u
import warnings; warnings.simplefilter("ignore")
CTIO = EarthLocation.from_geodetic(-70.80653 * u.deg, -30.16596 * u.deg, 2215 * u.m)
tt = Time(et, format="mjd", scale="utc")
obs = (get_body_barycentric("earth", tt).xyz.to(u.au).value.T - get_body_barycentric("sun", tt).xyz.to(u.au).value.T
       + CTIO.get_gcrs(tt).cartesian.xyz.to(u.au).value.T)

# per-exposure 5-sigma depth from its point-like detections: m5 = magzero - 2.5 log10(5 * median ferr)
m5 = np.full(ne, np.nan)
order = np.argsort(D["exp"], kind="stable"); st = np.searchsorted(D["exp"][order], np.arange(ne + 1))
for k in range(ne):
    j = order[st[k]:st[k + 1]]
    j = j[(D["flag"][j] == 0) & (D["snr"][j] > 4.5) & (D["snr"][j] < 20)]
    if len(j) and np.isfinite(E[k]["magzero"]):
        m5[k] = E[k]["magzero"] - 2.5 * np.log10(5 * np.median(D["ferr"][j]))
say(f"m5 median {np.nanmedian(m5):.2f}, 10-90 % {np.nanpercentile(m5, 10):.2f}-{np.nanpercentile(m5, 90):.2f}")

def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)

ue = unit(np.array([m["ra0"] for m in E]), np.array([m["dec0"] for m in E]))
cen = ue.mean(0); cen /= np.linalg.norm(cen)
rad = float(np.degrees(np.arccos(np.clip(ue @ cen, -1, 1))).max())

# ---- known bodies: label detections within 2" of the Horizons place
det_body = -np.ones(nd, np.int32)
K = json.load(open(a.known))["bodies"] if os.path.exists(a.known) else []
stem_k = {m["stem"]: k for k, m in enumerate(E)}
known = []
for bi, b in enumerate(K):
    nts = set()
    for s, (bra, bdec) in b["pos"].items():
        k = stem_k.get(s)
        if k is None:
            continue
        j = order[st[k]:st[k + 1]]
        dd = np.hypot((D["ra"][j] - bra) * np.cos(np.radians(bdec)), D["dec"][j] - bdec) * 3600
        hit = j[dd < 2.0]
        if len(hit):
            det_body[hit[np.argmax(D["snr"][hit])]] = bi; nts.add(int(night[k]))
    known.append(dict(name=b["name"], H=b["H"], weight=2 if b["H"] <= 7 else 1, nights=len(nts)))
donor = np.flatnonzero(det_body >= 0)
say(f"known: {len(K)} bodies, {sum(k['nights'] >= 5 for k in known)} with >= 5 nights; {len(donor)} donor detections")
if len(donor) < 50:      # too few real body detections to copy columns from: fall back to point-like faint detections
    donor = np.flatnonzero((D["flag"] == 0) & (D["snr"] < 10))[:200000]

# ---- planted bodies
def ecl2eq(v):
    c, s = np.cos(EPS), np.sin(EPS)
    return np.stack([v[..., 0], c * v[..., 1] - s * v[..., 2], s * v[..., 1] + c * v[..., 2]], -1)

def eq2ecl(v):
    c, s = np.cos(EPS), np.sin(EPS)
    return np.stack([v[..., 0], c * v[..., 1] + s * v[..., 2], -s * v[..., 1] + c * v[..., 2]], -1)

def kepler(r0, v0, dt):
    """universal-variable two-body propagation of one state over an array of dt (days)"""
    out = np.empty((len(dt), 3))
    r0n = np.linalg.norm(r0); vr0 = r0 @ v0 / r0n; alpha = 2 / r0n - v0 @ v0 / GM; sq = np.sqrt(GM)
    for i, t in enumerate(dt):
        x = sq * abs(alpha) * t if alpha > 0 else sq * t / r0n
        for _ in range(50):
            z = alpha * x * x
            if z > 1e-8:
                Cz = (1 - np.cos(np.sqrt(z))) / z; Sz = (np.sqrt(z) - np.sin(np.sqrt(z))) / z ** 1.5
            else:
                Cz, Sz = 0.5 - z / 24, 1 / 6 - z / 120
            F = r0n * vr0 / sq * x * x * Cz + (1 - alpha * r0n) * x ** 3 * Sz + r0n * x - sq * t
            dF = r0n * vr0 / sq * x * (1 - alpha * x * x * Sz) + (1 - alpha * r0n) * x * x * Cz + r0n
            dx = F / dF; x -= dx
            if abs(dx) < 1e-10:
                break
        z = alpha * x * x
        Cz, Sz = ((1 - np.cos(np.sqrt(z))) / z, (np.sqrt(z) - np.sin(np.sqrt(z))) / z ** 1.5) if z > 1e-8 else (0.5, 1 / 6)
        f = 1 - x * x / r0n * Cz; g = t - x ** 3 / sq * Sz
        out[i] = f * r0 + g * v0
    return out

COL = {"g": 0.6, "r": -0.15, "i": -0.4, "z": -0.5, "Y": -0.55, "VR": 0.1}
planted = []; rows = {c: [] for c in C}; rows_exp = []; rows_body = []
tries = 0
while len(planted) < a.planted and tries < 50 * a.planted:
    tries += 1
    lo, hi = [(30, 50), (50, 100), (100, 1000)][len(planted) % 3]       # equal numbers in three distance bands
    r = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))); Vt = float(rng.uniform(20.5, 23.6))
    k0 = int(rng.integers(ne)); t_ref = et[k0]
    # sky place inside the region at t_ref, then heliocentric position at distance r from the Sun
    dirn = ue[k0] + rng.normal(0, 0.01, 3); dirn /= np.linalg.norm(dirn)
    o = obs[k0]; bq = 2 * dirn @ o; cq = o @ o - r * r
    rho = (-bq + np.sqrt(bq * bq - 4 * cq)) / 2
    R = o + rho * dirn
    Re = eq2ecl(R)
    # velocity: bound, prograde-ish, q >= 25 au, inclination <= 40 deg
    ok = False
    for _ in range(200):
        e = rng.uniform(0, 0.6); inc = np.radians(rng.uniform(0, 40))
        a_ = r / (1 + e * rng.uniform(-1, 1)) if e > 0 else r
        if a_ * (1 - e) < 25 or a_ * (1 - e) > r + 1e-6 or a_ * (1 + e) < r - 1e-6:
            continue
        vmag = np.sqrt(GM * (2 / r - 1 / a_))
        # tangent basis in the ecliptic frame at Re; tilt the in-plane direction by the inclination
        rh = Re / r; zh = np.array([0, 0, 1.0]); t1 = np.cross(zh, rh); t1 /= np.linalg.norm(t1); t2 = np.cross(rh, t1)
        sgn = 1 if rng.random() < 0.97 else -1
        phi = rng.uniform(-inc, inc)
        h_dir = sgn * (np.cos(phi) * t1 + np.sin(phi) * t2)
        # radial part from e and r (energy and angular momentum)
        hmag = np.sqrt(GM * a_ * (1 - e * e)); vt = hmag / r
        if vt > vmag:
            continue
        vr = np.sqrt(max(0.0, vmag * vmag - vt * vt)) * (1 if rng.random() < 0.5 else -1)
        Ve = vt * h_dir + vr * rh; ok = True; break
    if not ok:
        continue
    H = Vt - 5 * np.log10(r * rho)                  # brightness set by the apparent magnitude at t_ref: far bodies are big
    if H > 7.5:
        continue
    Req, Veq = ecl2eq(Re), ecl2eq(Ve)
    P = kepler(Req, Veq, et - t_ref)
    for _ in range(2):                              # light time
        lt = np.linalg.norm(P - obs, axis=1) / C_AU
        P = kepler(Req, Veq, et - lt - t_ref)
    g = P - obs; dist = np.linalg.norm(g, axis=1); ug = g / dist[:, None]
    inside = (np.einsum("ij,ij->i", ug, ue) > np.cos(np.radians(1.05))) & (rng.random(ne) > 0.08)
    if not inside.any():
        continue
    rh_ = np.linalg.norm(P, axis=1)
    V = H + 5 * np.log10(rh_ * dist)
    mag = V + np.array([COL.get(b, 0.0) for b in band])
    snr_t = 5 * 10 ** (-0.4 * (mag - m5)); snr_o = snr_t + rng.normal(0, 1, ne)
    seen = inside & np.isfinite(snr_o) & (snr_o >= 4.5)
    nts = np.unique(night[seen])
    if len(nts) < 3:
        continue
    pid = len(planted)
    weight = 1 if r < 50 else 2 if r < 100 else 4
    planted.append(dict(r=round(r, 2), H=round(H, 2), a=round(float(a_), 2), e=round(float(e), 3), inc_deg=round(float(np.degrees(inc)), 1),
                        weight=weight, nights=int(len(nts)), detections=int(seen.sum())))
    for k in np.flatnonzero(seen):
        sig = (0.08 + 0.6 / snr_o[k]) * AS
        d0 = ug[k] + rng.normal(0, sig, 3); d0 /= np.linalg.norm(d0)
        dn = donor[rng.integers(len(donor))]
        ferr = float(np.median(D["ferr"][order[st[k]:st[k + 1]]][:2000]))
        rows["ra"].append(np.degrees(np.arctan2(d0[1], d0[0])) % 360); rows["dec"].append(np.degrees(np.arcsin(d0[2])))
        rows["flux"].append(snr_o[k] * ferr); rows["ferr"].append(ferr); rows["snr"].append(snr_o[k])
        rows["sharp"].append(D["sharp"][dn]); rows["flag"].append(D["flag"][dn]); rows["ccd"].append(D["ccd"][dn])
        rows["x"].append(rng.uniform(30, 2016)); rows["y"].append(rng.uniform(30, 4064))
        rows_exp.append(k); rows_body.append(100000 + pid)
say(f"planted {len(planted)} bodies ({tries} tries), {len(rows_exp)} detections; weights {np.bincount([p['weight'] for p in planted])}")

# ---- merge, order by exposure, shuffle inside each exposure
for c in C:
    D[c] = np.concatenate([D[c], np.array(rows[c], dtype=D[c].dtype)])
D["exp"] = np.concatenate([D["exp"], np.array(rows_exp, np.int32)])
det_body = np.concatenate([det_body, np.array(rows_body, np.int32)])
o = np.lexsort((rng.random(len(D["exp"])), D["exp"]))
os.makedirs(a.scen, exist_ok=True); os.makedirs(a.truth, exist_ok=True)
for c in C:
    np.save(os.path.join(a.scen, f"det_{c}.npy"), D[c][o])
np.save(os.path.join(a.scen, "det_exp.npy"), D["exp"][o])
exps = [dict(stem=m["stem"], mjd_mid=float(et[k]), band=band[k], exptime=m["exptime"], magzero=m["magzero"], fwhm=m["fwhm"],
             airmass=m["airmass"], ra0=m["ra0"], dec0=m["dec0"], m5=None if not np.isfinite(m5[k]) else round(float(m5[k]), 3),
             obs_xyz=[float(x) for x in obs[k]]) for k, m in enumerate(E)]
json.dump(exps, open(os.path.join(a.scen, "exposures.json"), "w"))
json.dump(dict(sid=a.sid, centre_radec=[float(np.degrees(np.arctan2(cen[1], cen[0])) % 360), float(np.degrees(np.arcsin(cen[2])))],
               radius_deg=round(rad + 1.1, 2), n_exposures=ne, n_detections=int(len(o)), first_mjd=float(et.min()), last_mjd=float(et.max())),
          open(os.path.join(a.scen, "region.json"), "w"))
np.savez_compressed(os.path.join(a.truth, a.sid + ".npz"), det_body=det_body[o], det_night=(night - night.min())[D["exp"][o]].astype(np.int32),
                    known=json.dumps(known), planted=json.dumps(planted), rid=a.rid)
say(f"wrote {a.scen} ({len(o):,} rows) and truth; findable (>= 5 nights): known {sum(k['nights'] >= 5 for k in known)}, "
    f"planted {sum(p['nights'] >= 5 for p in planted)}")
