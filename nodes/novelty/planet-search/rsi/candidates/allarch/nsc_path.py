"""NSC DR2 along each body's whole path, 2012-09 to 2019-11 (2026-11-19 request: "try harder", wider windows, no download).
NSC DR2 holds every public DECam (c4d), Mayall Mosaic3 (k4m, MzLS) and Bok 90Prime (ksb, BASS) exposure of those years, reduced
independently of our runs. Per 60-day block of the path (prediction 1-sigma <= 30", NSC_SMAX):
 1. exposures (nsc_dr2.exposure) whose centre lies within the field (c4d 1.1 deg, k4m 0.3, ksb 0.55) of the prediction at their mid
    time; the expected S/N there from the exposure's 10-sigma depth and the body's V from H (V - r 0.3, V - g -0.45, ...);
 2. single-epoch objects (nsc_dr2.object, ndet <= 3) inside the box around the path, kept when within max(2", 3 sigma) of the
    prediction at the object's own time; then their measurements (nsc_dr2.meas by objectid);
 3. a measurement counts as a candidate when its exposure is not in our runs' lists (dbt_used_exposures.txt), its magnitude is
    within 1.5 mag of the prediction and no Legacy Survey source lies within 1.5" (dr10_cones.json, filled as needed).
Identity: candidates of one night form a tracklet; each is tested like an ITF tracklet (itf_joint.py: one orbit through it with the
arc as a prior; shifted copies of 5", 15", 45" must fail); the nights that pass individually are then fitted together.
Writes nsc_path.json {id: {...}}. Resumable. Usage: ~/anaconda3/bin/python -I nsc_path.py BODIES.json [ID_SUBSTRING ...]"""
import ast, json, math, os, sys, time, urllib.parse, urllib.request
import numpy as np
from scipy.optimize import least_squares
import astropy.units as u
from astropy.coordinates import EarthLocation, get_body_barycentric
from astropy.time import Time
H = os.path.dirname(os.path.abspath(__file__)); C = os.path.dirname(H)
sys.path.insert(0, os.path.expanduser("$ALE_HOME/planet-search/linker"))
import orbit_fit as OF
from orbits import radec_to_unit, unit_to_radec
for k, (lo, la, h) in {"695": (-111.5997, 31.9633, 2120), "691": (-111.6000, 31.9633, 2070)}.items():
    OF.SITES.setdefault(k, EarthLocation.from_geodetic(lo * u.deg, la * u.deg, h * u.m))
SITE = {"c4d": "W84", "k4m": "695", "ksb": "691"}; FOV = {"c4d": 1.1, "k4m": 0.3, "ksb": 0.55}
VM = {"g": -0.45, "r": 0.3, "i": 0.6, "z": 0.8, "Y": 0.9, "VR": 0.1, "u": -1.6}
OURS = {l.strip()[:17] for l in open(os.environ.get("USED_LIST", C + "/dbt_used_exposures.txt")) if l.strip()}  # USED_LIST: the list as it was when the body's own run searched (2026-11-20)
import bisect, calendar
OM = sorted(calendar.timegm(time.strptime(x[4:17], "%y%m%d_%H%M%S")) / 86400 + 40587 for x in OURS if x.startswith("c4d") and len(x) >= 17)


def ours(name, m):
    """In our runs' lists: by name for c4d_ files; by start time (within 100 s) for NSC's early DECam 'tu' names (2012-2013)."""
    if name.startswith("c4d"):
        return name[:17] in OURS
    if name.startswith("tu"):
        k = bisect.bisect_left(OM, m - 0.00116); return k < len(OM) and abs(OM[k] - m) < 0.00116
    return False


inst_of = lambda name: "c4d" if name.startswith(("c4d", "tu")) else name[:3]
TOK = {"User-Agent": "archive-cover/1.0", "X-DL-AuthToken": "anonymous.0.0.anon_access"}
SMAX = float(os.environ.get("NSC_SMAX", 30))  # 1-sigma cap (2026-11-19: 600" made one body take > 40 min; larger ones go to nsc_pairs.py)
OUTF = os.environ.get("NSC_OUT", H + "/nsc_path.json"); res = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
sys.path.insert(0, H); import conecache as CC
CONES = H + "/dr10_cones.json"; cones = CC.load(CONES); CC.register(CONES, cones)
lit = lambda v: ast.literal_eval(v) if isinstance(v, str) else v
HH = {o["id"]: o.get("H") for o in json.load(open(os.path.expanduser("$ALE_HOME/rsi/decam-big-tno/progress/objects/objects.json")))["decam_big_tno"]}


def q(sql, tries=3):
    for a in range(tries):
        try:
            t = urllib.request.urlopen(urllib.request.Request("https://datalab.noirlab.edu/query/query?" + urllib.parse.urlencode(dict(sql=sql, ofmt="csv", **{"async": "false"})), headers=TOK),
                                       timeout=600).read().decode().strip().splitlines()
            hd = t[0].split(","); return [dict(zip(hd, r.split(","))) for r in t[1:]]
        except Exception as e:
            err = e; time.sleep(15 * (a + 1))
    raise RuntimeError(repr(err)[:120])


def dr10_clean(ra, de):
    cd = math.cos(math.radians(de))
    for c in cones:
        if math.hypot((c["ra"] - ra) * cd, c["dec"] - de) * 3600 < 60:
            break
    else:
        c = None
        for tab in ("ls_dr10.tractor", "ls_dr9.tractor"):
            rows = q(f"SELECT ra, dec, mag_r FROM {tab} WHERE q3c_radial_query(ra, dec, {ra}, {de}, {150 / 3600:.6f})")
            if rows:
                c = dict(ra=ra, dec=de, table=tab, rows=[[float(r["ra"]), float(r["dec"]), float(r["mag_r"]) if r["mag_r"] not in ("", "NaN", "Infinity") else 99.] for r in rows]); break
        c = c or dict(ra=ra, dec=de, table=None, rows=[]); cones.append(c); CC.save(CONES, cones)
    if c["table"] is None:
        return None
    return all(math.hypot((a - ra) * cd, b - de) * 3600 >= 1.5 for a, b, m in c["rows"])


class Orbit:
    def __init__(self, arc):
        A = json.load(open(arc)); self.p = np.array(lit(A["p"]), float); self.cov = np.array(lit(A["cov"]), float); self.t0 = float(A["epoch_tdb"])

    def at(self, m, site="W84"):
        m = np.atleast_1d(np.asarray(m, float)); R, tdb = OF.observer_bary(m, site); u0, s = OF.predict_sigma(self.p, self.cov, self.t0, tdb, R); ra, de = unit_to_radec(u0)
        X = OF.propagate(self.p[:3], self.p[3:], tdb - self.t0); S = get_body_barycentric("sun", Time(m, format="mjd")).xyz.value.T
        return np.degrees(ra) % 360, np.degrees(de), np.atleast_1d(s), np.linalg.norm(X - S, axis=1), np.linalg.norm(X - R, axis=1)

    def test(self, pts):  # pts: [(mjd, ra, dec, site)]; as itf_joint.test, sigma 0.15" per coordinate (NSC DR2 positions are Gaia-tied)
        m = np.array([x[0] for x in pts]); R = np.zeros((len(pts), 3)); tdb = np.zeros(len(pts))
        for k, x in enumerate(pts):
            r_, t_ = OF.observer_bary(np.atleast_1d(x[0]), x[3]); R[k] = r_[0]; tdb[k] = np.atleast_1d(t_)[0]
        uo = radec_to_unit(np.radians([x[1] for x in pts]), np.radians([x[2] for x in pts])); so = np.full(len(pts), .15)
        w, U = np.linalg.eigh(self.cov); W = (U / np.sqrt(np.maximum(w, 1e-30))).T; sc = np.array([1, 1, 1, 1e-3, 1e-3, 1e-3])
        f = lambda qq: np.concatenate([OF.residuals(qq * sc, self.t0, tdb, R, uo, so), W @ (qq * sc - self.p)])
        r = least_squares(f, self.p / sc, method="lm", xtol=1e-14, ftol=1e-14)
        rs = OF.residuals(r.x * sc, self.t0, tdb, R, uo, so).reshape(2, -1).T * so[:, None]; pr = W @ (r.x * sc - self.p)
        return float(np.sqrt(np.mean(np.sum(rs ** 2, 1)))), float(np.max(np.hypot(rs[:, 0], rs[:, 1]))), float(pr @ pr)

    def identity(self, pts):
        rms, mx, pen = self.test(pts); good = rms <= .35 and mx <= 1.0 and pen <= 20
        if not good:
            return dict(rms=round(rms, 3), max=round(mx, 3), prior_chi2=round(pen, 2), verdict="NOT THE SAME BODY" if rms > 1 or pen > 50 else "BORDERLINE")
        nfit = 0; fits = []
        for d in (5, 15, 45):
            for ang in (0, 90, 180, 270):
                dx, dy = d * math.cos(math.radians(ang)), d * math.sin(math.radians(ang))
                qq = [(m, ra + dx / 3600 / math.cos(math.radians(de)), de + dy / 3600, s) for m, ra, de, s in pts]
                r2, m2, p2 = self.test(qq); ok_ = (r2 <= .35 and m2 <= 1 and p2 <= 20); nfit += ok_
                if ok_:
                    fits.append(dict(shift=d, angle=ang, rms=round(r2, 3), prior_chi2=round(p2, 2)))
        return dict(rms=round(rms, 3), max=round(mx, 3), prior_chi2=round(pen, 2), shifted_fitting=nfit, shifts_that_fit=fits, verdict="SAME BODY" if nfit == 0 else "CONSISTENT, UNTESTABLE")


want = sys.argv[2:]
for b in json.load(open(sys.argv[1])):
    i = b["id"]
    if (want and not any(w in i for w in want)) or i in res:
        continue
    t_start = time.time(); O = Orbit(b["arc"]); Hb = b.get("H") or HH.get(i)
    g = np.arange(56190., 58810., 1.0); ra, de, sg, rh, dl = O.at(g)
    ok = sg <= SMAX; blocks = []; k = 0; ra_u = np.degrees(np.unwrap(np.radians(ra)))
    gat = lambda m: (np.interp(m, g, ra_u) % 360, np.interp(m, g, de), np.interp(m, g, sg), np.interp(m, g, rh), np.interp(m, g, dl))  # daily grid, < 0.5" off
    gsep = lambda a, d, pa, pd: np.hypot(((a - pa + 180) % 360 - 180) * np.cos(np.radians(pd)), d - pd) * 3600
    while k < len(g):
        j = min(k + 60, len(g)); sel = np.arange(k, j)[ok[k:j]]
        if len(sel):
            blocks.append(sel)
        k = j
    nexp = {"c4d": 0, "k4m": 0, "ksb": 0}; exp_snr = []; cands = []; nobj = 0; errs = []
    for sel in blocks:
        r0, r1 = ra[sel], de[sel]; pad = (3 * sg[sel].max() + 2) / 3600
        if np.ptp(r0) > 180:
            continue  # wraps 0h: skipped (rare)
        cdm = math.cos(math.radians(float(np.mean(r1))))
        try:
            E = q(f"SELECT exposure, instrument, filter, mjd, exptime, depth10sig, ra, dec FROM nsc_dr2.exposure WHERE ra BETWEEN {r0.min() - 1.2 / cdm:.4f} AND {r0.max() + 1.2 / cdm:.4f} "
                  f"AND dec BETWEEN {r1.min() - 1.2:.4f} AND {r1.max() + 1.2:.4f} AND mjd BETWEEN {g[sel[0]] - .5:.2f} AND {g[sel[-1]] + 1.5:.2f}")
        except RuntimeError as e:
            errs.append(str(e)); continue
        for e in E:
            m = float(e["mjd"]) + float(e["exptime"]) / 172800; a_, d_, s_, rh_, dl_ = (np.atleast_1d(x) for x in gat(m))
            sep = float(gsep(float(e["ra"]), float(e["dec"]), a_[0], d_[0])) / 3600
            if sep > FOV.get(e["instrument"], 1.0) or s_[0] > 600:
                continue
            nexp[e["instrument"]] = nexp.get(e["instrument"], 0) + 1
            if Hb is not None and e["depth10sig"] not in ("", "NaN"):
                V = Hb + 5 * math.log10(rh_[0] * dl_[0]); mb = V - VM.get(e["filter"], 0.3)
                exp_snr.append((e["exposure"], e["filter"], round(m, 4), round(10 * 10 ** (-0.4 * (mb - float(e["depth10sig"]))), 1), ours(e["exposure"], float(e["mjd"]))))
        if not E:
            continue
        try:
            Ob = q(f"SELECT id, ra, dec, mjd, ndet FROM nsc_dr2.object WHERE ra BETWEEN {r0.min() - pad / cdm:.5f} AND {r0.max() + pad / cdm:.5f} AND dec BETWEEN {r1.min() - pad:.5f} "
                   f"AND {r1.max() + pad:.5f} AND ndet <= 3 AND mjd BETWEEN {g[sel[0]] - .5:.2f} AND {g[sel[-1]] + 1.5:.2f}")
        except RuntimeError as e:
            errs.append(str(e)); continue
        nobj += len(Ob); keep = []
        if Ob:
            om = np.array([float(o["mjd"]) for o in Ob]); pa, pd, ps, _, _ = gat(om)
            sp = gsep(np.array([float(o["ra"]) for o in Ob]), np.array([float(o["dec"]) for o in Ob]), pa, pd)
            keep = [o["id"] for o, x, s3 in zip(Ob, sp, ps) if x <= max(2.0, 3 * s3) + 10]
        for k0 in range(0, len(keep), 200):
            ids = ",".join(f"'{x}'" for x in keep[k0:k0 + 200])
            try:
                M = q(f"SELECT objectid, exposure, filter, mjd, ra, dec, mag_auto, magerr_auto FROM nsc_dr2.meas WHERE objectid IN ({ids})")
            except RuntimeError as e:
                errs.append(str(e)); continue
            for r in M:
                inst = inst_of(r["exposure"]); m = float(r["mjd"]); a_, d_, s_, rh_, dl_ = (np.atleast_1d(x) for x in gat(m))
                if float(gsep(float(r["ra"]), float(r["dec"]), a_[0], d_[0])) > max(2.0, 3 * s_[0]) + 1:
                    continue
                a_, d_, s_, rh_, dl_ = O.at(m, SITE.get(inst, "W84"))
                sep = math.hypot((float(r["ra"]) - a_[0]) * math.cos(math.radians(d_[0])), float(r["dec"]) - d_[0]) * 3600
                if sep > max(2.0, 3 * s_[0]):
                    continue
                mb = None if Hb is None else Hb + 5 * math.log10(rh_[0] * dl_[0]) - VM.get(r["filter"], 0.3); mag = float(r["mag_auto"]) if r["mag_auto"] not in ("", "NaN") else None
                cands.append(dict(objectid=r["objectid"], exposure=r["exposure"], inst=inst, filter=r["filter"], mjd=m, ra=float(r["ra"]), dec=float(r["dec"]), mag=mag,
                                  mag_pred=None if mb is None else round(mb, 2), sep=round(sep, 2), sigma=round(float(s_[0]), 2), ours=ours(r["exposure"], m)))
    # candidates: not ours, brightness, no fixed source
    good = []
    for c in cands:
        if c["ours"]:
            continue
        if c["mag"] is not None and c["mag_pred"] is not None and abs(c["mag"] - c["mag_pred"]) > 1.5:
            c["why_out"] = "brightness"; continue
        cl = dr10_clean(c["ra"], c["dec"])
        if cl is False:
            c["why_out"] = "Legacy Survey source within 1.5\""; continue
        good.append(c)
    nights = {}
    for c in good:
        nights.setdefault((c["inst"], int(c["mjd"] - (0.75 if c["inst"] != "c4d" else 0.5))), []).append(c)
    tests = []
    for (inst, n), L_ in sorted(nights.items()):
        pts = [(c["mjd"], c["ra"], c["dec"], SITE.get(inst, "W84")) for c in L_]
        try:
            r = O.identity(pts)
        except Exception as e:
            r = dict(verdict="FIT ERROR", error=repr(e)[:80])
        tests.append(dict(inst=inst, night=n, n=len(L_), exposures=sorted({c["exposure"] for c in L_}), filters=sorted({c["filter"] for c in L_}), **r))
    passing = [t for t in tests if t["verdict"] in ("SAME BODY", "CONSISTENT, UNTESTABLE")]
    joint = None
    if len(passing) >= 2:
        pts = [(c["mjd"], c["ra"], c["dec"], SITE.get(c["inst"], "W84")) for c in good for t in passing if (c["inst"], int(c["mjd"] - (0.75 if c["inst"] != "c4d" else 0.5))) == (t["inst"], t["night"])]
        joint = O.identity(pts); joint["nights"] = len(passing)
    res[i] = dict(id=i, C=b.get("C"), exposures=nexp, expected_snr=exp_snr, n_objects_scanned=nobj, n_meas_near=len(cands), n_candidates=len(good), candidates=good[:200],
                  out=[c for c in cands if c.get("why_out")][:50], tests=tests, joint=joint, errors=errs[:5], seconds=round(time.time() - t_start), checked=time.strftime("%Y-%m-%d %H:%M"))
    res = CC.put(OUTF, i, res[i])
    nx = sum(1 for x in exp_snr if not x[4] and x[3] >= 5)
    print(f"{(b.get('C') or ''):5s} {i[4:48]:44s} exp {nexp} (S/N>=5, not ours: {nx}) objects {nobj} near {len(cands)} cand {len(good)} nights {len(tests)} pass "
          f"{[(t['inst'], t['night'], t['verdict'][:9]) for t in passing][:4]} joint {joint and joint['verdict']} {round(time.time() - t_start)} s", flush=True)
print("wrote", OUTF)
