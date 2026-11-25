"""Is a candidate already known? The routine check for every lead before it is called new (2026-11-09).

Lesson of KNOWN_BODY_F1 (r2 gen002's F1 lead): a body known only from a short arc (a Rubin discovery of 2026, say) can sit arcminutes from its catalogued
prediction in our older images, so "no known body within 10 arcsec" proves nothing. The test that works: anything catalogued that moves like the
candidate within a generous radius gets one orbit fitted through both data sets; one body fits both, two bodies do not.

Input SPEC.json, either
  {"id": ..., "run": "r2", "gen": "gen002", "sid": "scen05", "p": 0.98885, "r": 56.27}   (a program candidate: its orbit at the field's exposures
      gives one place per night, sigma 0.3"), or
  {"id": ..., "points": [[mjd_utc, ra, dec, sigma_arcsec, site], ...]}   (measured places; site "DECam", "CFHT" or "Subaru").
Steps (MPC files: the newest rsi/data/mpc/<date>/, refreshed by novelty_watch.py once a day):
  1. MPCORB, every body with a > 3.5 au (5.5 until 2026-11-19, when a Jupiter Trojan slipped through) or a one-opposition orbit, two-body with light
     time at the middle date of each season of the arc (nights more than 60 days apart start a new season; until 2026-11-19 only at the arc's
     middle date, which missed a known body that matched one season of a mis-linked arc), with that season's own rate: flagged if its sky motion is
     within 1.5 "/h of the candidate's in both axes and it lies within 2' (good orbits: U <= 4 and several oppositions) or 90' (all others).
     Power check for any SAME or borderline result: the fast screen with our places moved by 5 fixed sky offsets (2-5'); if 2 or more of those
     fit within max(1", 2 x the real rms), the link cannot be tested (two short arcs far apart join loosely; KNOWN_BODY_F3B "matched" TRKSUB01's
     4 nights at 0.63"): reported as a weak link, not as KNOWN or CHECK BY HAND.
  1b. (2026-11-22, novelty_fix) all-planet n-body places (allarch/nbody_screen.py's integrator: Sun + 8 planets from JPL Horizons vectors, RK4
     2-day steps, Hermite interpolation, light time) of every MPCORB orbit beyond 3.5 au and of every body flagged so far, at each of our points:
     flagged as in step 1 at each season's middle (Trojans, Hildas, Centaurs that two-body places miss: C### = (591789) 2014 EM90, C### =
     2015 HX238, C### = (459971) 2014 ON6 were never flagged), and every flagged body gets its offset from our place night by night.
  2. ITF: tracklets within 0.6 day of a candidate night, within 1' of its place there (carried by its rate), rate within 1.5 "/h (or single): flagged for a look by hand.
  3. (rewritten 2026-11-22, novelty_fix) Every flagged body (no cap; until 2026-11-22 only the closest 20, and only those a barycentric two-body
     orbit through both sets fitted to <= 3" got the n-body test, which rejected true long-gap identities: C### vs 2014 WB626 is 12.7" two-body,
     0.106" n-body; C### vs (666634) 2010 RJ226 4.2"). Bodies whose ephemeris passes the test below come first, then by distance; time budget TBUDGET.
     a. ephemeris test: our places within EPH_TOL (2") of its all-planet n-body ephemeris on >= 2 nights and >= half the nights -> KNOWN, unless
        2 or more of 5 copies of our places shifted by 2-5' also find some body of the n-body set that way (-> PROBABLY). This decides the Trojans
        and Centaurs whose long arcs the fit of (b) cannot follow (their own observations leave 0.6-1.1"), as SkyBoT does.
     b. the n-body fit (cand9_orbit.py, Sun + giant planets, grid 1990-2027): its observations alone, then both sets; SAME BODY if the joint rms
        <= 0.3" and no point is over 1"; borderline (rms <= 1") is reported for a look by hand. Every body that (a) does not decide is fitted:
        the two-body screen is only recorded (C### vs 2015 DR249, a 10-day arc: 679" two-body, 0.146" n-body, so no pre-screen limit is safe),
        except for a <= 3.5 au (main belt, which this fit cannot model over years): fitted only if the screen <= INNER_SCREEN (100") or near.
        If the whole card does not fit: the nights within EPH_SUB (10") of its ephemeris alone (a mislinked night: one card's night
        was 290.8" off 2014 WB626), and each season alone when it was flagged within 3' there. SAME on >= 2 nights and >= half of them ->
        KNOWN (the other nights are named in the verdict); SAME on fewer -> UNSURE (mixed card). When its own observations do not fit to
        <= ALONE_OK (0.3"), a borderline or failed fit says nothing: then (a) decides, and a good orbit (U <= 4) more than EPH_SUB off on every
        night is not the same body.
     c. power check for every SAME or borderline fit: the n-body fit with our places moved by 5 fixed sky offsets (until 2026-11-22 the
        two-body screen); 2 or more fitting within max(1", 2 x the real rms) make it weak (PROBABLY / weak link).
  4. SkyBoT at the first and last night (DECam site, 10'), distant bodies listed (it integrates orbits; a second opinion, not the test).
  4b. (2026-11-22) SkyBoT per night, every class, 72" cones on up to 8 nights: one object within max(3", min(2 err, 20")) on >= 3/5 of
     the nights (>= 2) is KNOWN (catches perturbed Trojans, Hildas, Centaurs that two-body MPCORB misses); on fewer nights: CHECK BY HAND.
     (novelty_fix: Setebos, Uranus XIX, sat 2.2-2.5" from C### on 5 of 5 nights: inside the 3" floor, kept;
     a 2" floor would miss it. Numbered asteroids are now named "(number) designation", satellites by name: SkyBoT numbers Setebos 719.)
  5. If the candidate's own arc is over 300 days: its orbit is carried to every 4th day of 2024-01..today and step 1 is repeated there (Rubin era),
     and (2026-11-17) every ITF observation 2013..today within 30" of the carried orbit at its own time is reported (CHECK BY HAND),
     radius 5' for poor orbits too (both orbits are good in that era).
  0. Our own filed bodies (own_bodies.json: filed positions in mpc_report/*.psv, not yet in the MPC's files while SARC holds them): an n-body orbit per
     body (cached in own_bodies_fit.json), predicted at the candidate's dates; within 30" -> the identity fit as in step 3.
Verdict: OURS <name> | KNOWN <designation> (fit good and shifted copies fail; or the ephemeris test) | PROBABLY <designation> (fit good, shifted
copies fit too: a third epoch must decide) | UNSURE: <name> is the same body on some nights only (mixed card) | CHECK BY HAND (borderline or failed
fit on a flagged body within 10', or an ITF tracklet; since 2026-11-22 always resolved at once by resolve_hand.py into KNOWN / UNSURE: <why> /
NOT IN MPC FILES) | NOT IN MPC FILES OF <date>.
Optional in SPEC: "state", "epoch" (barycentric, au, au/day, MJD TDB) as the start for step 5.
Caches (novelty_fix): NOVELTY_CACHE (default candidates/novelty_cache/): mpc_obs/ (MPC observations, OBS_AGE hours), planets/ (Horizons vectors).
Sites (2026-11-22, obscode): every MPC observation is fitted from its own observatory (the MPC table NOVELTY_OBSCODES, default
candidates/mpc_obscodes.json; until then all codes but 8 sat at DECam's site); observations from space (no parallax constants) are left out.
Spacecraft and roving lines (2026-11-23, TRKSUB01 satobs; numbers in allarch/doublecheck/satobs.md): until then get_obs dropped the second lines
('s', 'v') and obs80 then dropped the first lines of codes without parallax constants, so a body seen only from space (2010 OT53: 13 WISE lines,
"0 usable observations") was never tested, and HST's lines of 2003 BF91 / BG91 / BH91 (all their observations) were dropped (before 2026-11-22 18:44
they were fitted from DECam's site). Now get_obs keeps every line (a cache file without the second lines is fetched again), and obs80 fits an 'S' line
from the spacecraft's geocentric position in its 's' line (MPC 80-column satellite format: column 33 unit 1 = km, 2 = au; X, Y, Z with signs in
columns 34-45, 46-57, 58-69) and a 'V' line from the observer's place in its 'v' line (east longitude, latitude in degrees, altitude in metres).
Only lines whose second line is missing or unreadable, and other codes without constants, are still left out (counted in OBS80_SKIPPED).
Tests that did not run (2026-11-22, identity crashfix): result key "untested" [{body, reason}] (fit crash, HTTP error or timeout, unreadable
reply, < 3 usable observations, a failed season fit, the time budget); a lead with any is UNSURE (n identity tests did not run: ...), never NOT IN MPC.
(2026-11-22, openitems) A failed screening step is listed there too, under the step's name: step 1b (nbody_error), step 0 (own filed bodies),
step 5 (Rubin-era screen, ITF path scan), and SkyBoT 4b when nights failed and fewer than 2, or fewer than half, of those asked were answered.
Usage, from candidates/: python -I novelty_check.py SPEC.json OUT.json"""
import glob, gzip, json, math, os, re, subprocess, sys, time, urllib.parse, urllib.request
import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, get_body_barycentric_posvel, solar_system_ephemeris
import astropy.units as u
H = os.path.expanduser("$ALE_HOME/planet-search/rsi"); C = H + "/candidates"
sys.path.insert(0, C); sys.path.insert(0, H + "/harness"); sys.path.insert(0, os.path.dirname(H) + "/linker")
import orbits as O, mpc_check as M
CAU = 173.1446326846693; RATE_TOL = 1.5; R_GOOD = 2.; R_POOR = 90.; R_ITF = 1.; SCREEN = 3.; GMB = 2.9591220828559115e-04 * 1.001341
# 2026-11-22 (TRKSUB01 novelty_fix; numbers behind each in allarch/doublecheck/novelty_fix.handback.md): no MAXTEST; the identity rests on the n-body
# fit and the all-planet ephemeris; the two-body screen is kept for the record only (no limit is safe: C### vs 2015 DR249 is 679" two-body, 0.146" n-body).
# Classes (record): tno q >= Q_TNO au; pert a > 3.5 au and q < Q_TNO; inner a <= 3.5 au
Q_TNO = 25.; INNER_SCREEN = 100.; EPH_TOL = 2.0; EPH_SUB = 10.; ALONE_OK = 0.3; TBUDGET = 1200.; OBS_AGE = 12.
NB_P0, NB_NP = 55800., 6401          # the all-planet integration's planet table: MJD (TDB) 55800-62200 (2011-09 to 2029-01), daily rows
SHIFTS = ((180, 60), (-240, 120), (120, -200), (-90, -260), (300, -90))
CACHE = os.environ.get("NOVELTY_CACHE", C + "/novelty_cache")
_src = open(C + "/cand9_orbit.py").read()
for old, new in (('Time("2026-01-01").mjd', 'Time("2027-06-01").mjd'), ('Time("2012-06-01").mjd', 'Time("1990-01-01").mjd')):
    assert _src.count(old) == 1, old; _src = _src.replace(old, new)
# 2026-11-22 (TRKSUB01 identity, crashfix): cand9's traj() built no forward (backward) integration when no point lay after (before) the fit epoch,
# yet light time (or a wild optimizer step) then asks for that side: AttributeError("'NoneType' object has no attribute 'sol'"). The crashed test
# was dropped and the lead reported NOT IN MPC (that hid C### = 2006 TM130). Both sides are now always built: identical numbers wherever the old
# one did not crash (traj's callers pass lo <= ep <= hi). The same two-line change for cand9_orbit.py: allarch/doublecheck/cand9_traj.diff.
if " if hi > ep else None" in _src:
    for old in (" if hi > ep else None", " if lo < ep else None"):
        assert _src.count(old) == 1, old; _src = _src.replace(old, "")
NB = {"__name__": "cand9_orbit_ext", "__file__": C + "/cand9_orbit.py"}; exec(compile(_src, "cand9_orbit.py", "exec"), NB)
SITES = {"W84": "DECam", "807": "DECam", "X05": "DECam", "I11": "DECam", "T14": "CFHT", "568": "CFHT", "T09": "Subaru", "T12": "CFHT"}
# 2026-11-22 (TRKSUB01 obscode; numbers in allarch/doublecheck/obscode.handback.md): SITES above is no longer used for MPC lines. Until then every
# code outside those 8 was placed at DECam's site, which moves a body at 40 au by up to ~0.3", at 5 au by ~2.5" (Kitt Peak, Pan-STARRS, Mauna Kea,
# La Palma, Paranal...). Now every ground code is its own site from the MPC's parallax constants (OBSCODES: the MPC table ObsCodes.html as JSON
# {code: [east longitude deg, rho cos phi', rho sin phi', name]}), added to cand9's SITES under the code itself; codes without constants (space
# telescopes 250 / C51 / C57..., roving 247) are left out of the fits (obs80) and placed at the geocentre in predictions. Our own points keep
# "DECam" / "CFHT" / "Subaru" (the 8 codes' MPC sites are within 11 km of those: < 0.004" at 5 au).
OBSCODES = os.environ.get("NOVELTY_OBSCODES", C + "/mpc_obscodes.json")
try:
    _OBSC = json.load(open(OBSCODES))
except OSError as _ex:
    raise SystemExit(f"novelty_check: the MPC observatory table {OBSCODES} is missing ({_ex}); see allarch/doublecheck/obscode.handback.md")
OBS80_SKIPPED = {}      # code -> number of MPC lines left out of fits because the code has no parallax constants
SPACE = {}              # (satobs) site key "SAT ..." -> geocentric position (au, equatorial J2000) of a spacecraft at one observation
OBS80_SPACE = {}        # (satobs) code -> number of spacecraft / roving lines fitted from their own place
_observer0 = NB["observer"]


def _observer(m, site):
    """cand9's observer, plus spacecraft sites (satobs): the Earth's barycentric place (as cand9) + the 's' line's geocentric vector."""
    if site in SPACE:
        t = Time(m, format="mjd", scale="utc")
        with solar_system_ephemeris.set("builtin"):
            return get_body_barycentric("earth", t).xyz.to(u.au).value + SPACE[site]
    return _observer0(m, site)


NB["observer"] = _observer


def site_of(code, space=None):
    """Observer site for an MPC observatory code: the code itself (EarthLocation from its MPC parallax constants, made once and kept in
    cand9's SITES) for a ground site; `space` (None: leave the line out of a fit) for codes without constants or not in the table."""
    if code in NB["SITES"]:
        return code
    c = _OBSC.get(code)
    if not c or c[0] is None:
        return space
    from astropy.coordinates import EarthLocation
    lon = math.radians(c[0]); R_E = 6378.137      # km; the MPC's constants are in units of the Earth's equatorial radius
    NB["SITES"][code] = EarthLocation.from_geocentric(c[1] * math.cos(lon) * R_E, c[1] * math.sin(lon) * R_E, c[2] * R_E, unit=u.km)
    return code


import datetime
_MJD0 = datetime.date(1858, 11, 17).toordinal()


def fast_obs_date(l):
    """MJD (UTC) of an 80-column line (mpc_check.obs_date without astropy, which made the ITF pass take minutes)."""
    return datetime.date(int(l[15:19]), int(l[20:22]), 1).toordinal() - _MJD0 + float(l[23:32]) - 1


def fast_epoch(s):
    c = {"I": 1800, "J": 1900, "K": 2000}[s[0]] + int(s[1:3]); f = lambda ch: int(ch) if ch.isdigit() else ord(ch) - 55
    return datetime.date(c, f(s[3]), f(s[4])).toordinal() - _MJD0


def mpc_dir():
    return sorted(glob.glob(H + "/data/mpc/20*-*-*/"))[-1].rstrip("/")


def spec_points(S):
    if S.get("points"):
        return [tuple(p[:4]) + (p[4] if len(p) > 4 else "DECam",) for p in S["points"]]
    sys.path.insert(0, H + "/harness"); import grade as G
    P = json.load(open(f"{H}/private/scenarios/{S['sid']}.json")); E = sorted(P["all_exposures"], key=lambda e: e["mjd_mid"])
    Cs = json.load(open(f"{H}/runs/{S['run']}/gens/{S['gen']}/ls6_runs/{S['sid']}/candidates.json"))["candidates"]
    c = [c for c in Cs if abs(c["p"] - S["p"]) < 2e-5 and ("r" not in S or abs(float(np.linalg.norm(c["state"][:3])) - S["r"]) < .06)]
    assert len(c) == 1, f"{len(c)} candidates match p {S['p']} r {S.get('r')}"
    c = c[0]; nights = {}
    for e in E:
        q = G.positions(dict(state=np.array(c["state"], float), epoch=c["epoch_mjd"]), [e["mjd_mid"]], [np.array([e["obs_x"], e["obs_y"], e["obs_z"]])])
        if q is not None:
            nights.setdefault(int(math.floor(e["mjd_mid"] - .5)), []).append((e["mjd_mid"], float(q[0][0]), float(q[1][0])))
    return [(float(np.mean([x[0] for x in v])), float(np.degrees(np.arctan2(np.mean(np.sin(np.radians([x[1] for x in v]))), np.mean(np.cos(np.radians([x[1] for x in v]))))) % 360),
             float(np.mean([x[2] for x in v])), .3, "DECam") for _, v in sorted(nights.items())]


def load_mpcorb(d):
    rows = []
    with gzip.open(d + "/MPCORB.DAT.gz", "rt", errors="replace") as fh:
        for l in fh:
            if len(l) < 160 or not l[0:1].strip():
                continue
            try:
                a = float(l[92:103]); one = "days" in l[127:136]
                if a > 3.5 or one:
                    rows.append((float(l[26:35]), float(l[37:46]), float(l[48:57]), float(l[59:68]), float(l[70:79]), float(l[80:91]), a, fast_epoch(l[20:25].strip()),
                                 l[166:194].strip(), l[105:106].strip(), l[127:136].strip(), one))
            except ValueError:
                continue
    A = {k: np.array([r[i] for r in rows]) for i, k in enumerate(("M", "om", "Om", "inc", "e", "n", "a", "ep"))}
    for k in ("M", "om", "Om", "inc", "n"):
        A[k] = np.radians(A[k])
    A["raw"] = [r[:8] for r in rows]; A["name"] = [r[8] for r in rows]; A["U"] = [r[9] for r in rows]; A["arc"] = [r[10] for r in rows]
    A["poor"] = np.array([r[11] or r[9] in ("", "E", "D", "F", "5", "6", "7", "8", "9") for r in rows]); A["ok"] = (A["e"] < 1) & (A["a"] > 0)
    return A


def eh(m):
    with solar_system_ephemeris.set("builtin"):
        t = Time(m, format="mjd", scale="tdb"); return (get_body_barycentric("earth", t).xyz - get_body_barycentric("sun", t).xyz).to(u.au).value


def places(A, m):
    E = eh(m); Mm = A["M"] + A["n"] * (m - A["ep"]); r, _ = O.elements_to_rv(A["a"], A["e"], A["inc"], A["Om"], A["om"], Mm); g = O.ecl_to_eq(r) - E; d = np.linalg.norm(g, axis=1)
    r, _ = O.elements_to_rv(A["a"], A["e"], A["inc"], A["Om"], A["om"], Mm - A["n"] * d / CAU); g = O.ecl_to_eq(r) - E; d = np.linalg.norm(g, axis=1)
    return np.degrees(np.arctan2(g[:, 1], g[:, 0])) % 360, np.degrees(np.arcsin(g[:, 2] / d))


def flag_at(A, m, ra0, de0, rate0, r_poor=None):
    ra, de = places(A, m); ra1, de1 = places(A, m + 1); cd = math.cos(math.radians(de0))
    sep = np.hypot(((ra - ra0 + 180) % 360 - 180) * cd, de - de0) * 60; rr = np.c_[((ra1 - ra + 180) % 360 - 180) * np.cos(np.radians(de)) * 150, (de1 - de) * 150]
    dr = np.max(np.abs(rr - np.asarray(rate0)[None]), axis=1); lim = np.where(A["poor"], R_POOR if r_poor is None else r_poor, R_GOOD)
    k = np.where(A["ok"] & (dr < RATE_TOL) & (sep < lim))[0]
    return [dict(idx=int(i), name=A["name"][i], sep_arcmin=round(float(sep[i]), 2), rate_diff=round(float(dr[i]), 2), a=round(float(A["a"][i]), 2), e=round(float(A["e"][i]), 3),
                 i=round(float(np.degrees(A["inc"][i])), 2), U=A["U"][i], arc=A["arc"][i], date=Time(m, format="mjd").iso[:10]) for i in k[np.argsort(sep[k])]]


def cache_dir(sub):
    d = os.path.join(CACHE, sub); os.makedirs(d, exist_ok=True); return d


def get_obs(des):
    """The body's 80-column MPC observations (MPC get-obs API, one request at a time), kept in CACHE/mpc_obs/ for OBS_AGE hours (2026-11-22: every
    flagged body is now fetched, and resolve_hand asks again for the bodies a verdict names)."""
    f = os.path.join(cache_dir("mpc_obs"), re.sub(r"[^A-Za-z0-9]+", "_", re.sub(r"^\(\d+\)\s*", "", des)).strip("_") + ".json")
    if os.path.exists(f) and time.time() - os.path.getmtime(f) < OBS_AGE * 3600:
        L = json.load(open(f))
        if _pairs_ok(L):          # (satobs) a cache file written before 2026-11-23 lacks the 's' / 'v' second lines: fetch again
            return L
    o = subprocess.run(["curl", "-s", "--max-time", "90", "-X", "GET", "-H", "Content-Type: application/json", "-d", json.dumps({"desigs": [re.sub(r"^\(\d+\)\s*", "", des)], "output_format": ["OBS80"]}),
                        "https://data.minorplanetcenter.net/api/get-obs"], capture_output=True, text=True).stdout
    L = [l for l in json.loads(o)[0]["OBS80"].splitlines() if len(l) >= 80]      # (satobs) second lines 's' / 'v' kept for obs80
    json.dump(L, open(f + ".tmp", "w")); os.replace(f + ".tmp", f); time.sleep(0.3)
    return L


def _pairs_ok(L):
    """(satobs) True when every 'S' / 'V' first line is followed by its 's' / 'v' second line (a complete get-obs reply)."""
    return all(k + 1 < len(L) and L[k + 1][14:15] == l[14].lower() for k, l in enumerate(L) if l[14:15] in ("S", "V"))


def _second(l, nxt):
    """(satobs) The observer of a satellite ('S') or roving ('V') line from its second line nxt: a site key, or None when unreadable."""
    if len(nxt) < 80 or nxt[14] != l[14].lower() or nxt[15:32].split()[:3] != l[15:32].split()[:3] or nxt[77:80] != l[77:80]:
        return None
    try:
        if l[14] == "S":
            v = np.array([float(nxt[a:b].replace(" ", "")) for a, b in ((33, 45), (45, 57), (57, 69))])
            v = v / 149597870.7 if nxt[32] == "1" else v if nxt[32] == "2" else None
            if v is None or not 1e-5 < np.linalg.norm(v) < 0.2:      # beyond 0.2 au (Kepler, Spitzer, New Horizons...): not read here
                return None
            key = "SAT " + l[77:80] + " " + nxt[32:69]; SPACE[key] = v; return key
        lon, lat, alt = (float(x) for x in nxt[32:77].split()[:3])
        if not (0 <= lon <= 360 and -90 <= lat <= 90 and -500 < alt < 6000):
            return None
        key = f"ROV {lon:.5f} {lat:.5f} {alt:.0f}"
        if key not in NB["SITES"]:
            from astropy.coordinates import EarthLocation
            NB["SITES"][key] = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, alt * u.m)
        return key
    except (ValueError, IndexError):
        return None


def obs80(L):
    out = []
    for k, l in enumerate(L):
        if l[14:15] in ("s", "v"):      # (satobs) a second line: read with its first line
            continue
        try:
            m = fast_obs_date(l)
            ra = sum(float(v) / 60 ** q for q, v in enumerate(l[32:44].split())) * 15; de = (-1 if l[44] == "-" else 1) * sum(float(v) / 60 ** q for q, v in enumerate(l[45:56].split()))
        except ValueError:
            continue
        if NB["GRID"][0] + 1 < m < NB["GRID"][-1] - 1:
            s = site_of(l[77:80])
            if s is None and l[14] in "SV":      # (satobs) spacecraft / roving observer: its own place from the second line
                s = _second(l, L[k + 1] if k + 1 < len(L) else "")
                if s is not None:
                    OBS80_SPACE[l[77:80]] = OBS80_SPACE.get(l[77:80], 0) + 1
            if s is None:       # no parallax constants and no readable second line: left out (counted)
                OBS80_SKIPPED[l[77:80]] = OBS80_SKIPPED.get(l[77:80], 0) + 1; continue
            out.append((m, ra, de, .2, s))
    return out


def start_state(pts):
    """A barycentric state through the candidate's places at 45 au with its sky motion (a start only; the fits move it)."""
    m0 = pts[len(pts) // 2][0]; o = NB["observer"](m0, pts[len(pts) // 2][4]); a, d = math.radians(pts[len(pts) // 2][1]), math.radians(pts[len(pts) // 2][2])
    uv = np.array([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)]); b = o @ uv; rho = -b + math.sqrt(b * b - o @ o + 45. ** 2); x = o + rho * uv
    v = np.cross([0, 0, 1.], x); v = v / np.linalg.norm(v) * math.sqrt(2.9591220828559115e-04 / 45.); return np.r_[x, v], m0


def mpcorb_state(A, i, m_near):
    """Barycentric state from the MPCORB elements; if their epoch is outside the ephemeris grid, at m_near instead (two-body: the fit corrects it)."""
    Mdeg, om, Om, inc, e, n, a, ep = A["raw"][i]
    if not NB["GRID"][0] + 1 < ep < NB["GRID"][-1] - 1:
        Mdeg = (Mdeg + n * (m_near - ep)) % 360; ep = m_near
    s = np.array(M.state(a, e, inc, Om, om, Mdeg))
    with solar_system_ephemeris.set("builtin"):
        sp, sv = get_body_barycentric_posvel("sun", Time(ep, format="mjd", scale="tdb"))
    return s + np.r_[sp.xyz.to(u.au).value, sv.xyz.to(u.au / u.day).value], ep


def tb_elements(x, v, mu):
    rn = np.linalg.norm(x); h = np.cross(x, v); hn = np.linalg.norm(h); inc = math.acos(max(-1, min(1, h[2] / hn))); Om = math.atan2(h[0], -h[1])
    ev = np.cross(v, h) / mu - x / rn; e = float(np.linalg.norm(ev)); a = 1 / (2 / rn - v @ v / mu)
    if not (0 <= e < 1 and a > 0):
        return None
    nv = np.array([math.cos(Om), math.sin(Om), 0.]); hz = h / hn; om = math.atan2(np.cross(nv, ev) @ hz, nv @ ev); f = math.atan2(np.cross(ev, x) @ hz, ev @ x)
    E = 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(f / 2)); return a, e, inc, Om, om, E - e * math.sin(E)


def tb_resid(state, ep, P):
    el = tb_elements(state[:3], state[3:], GMB)
    if el is None:
        return np.full(2 * len(P), 1e5)
    a, e, inc, Om, om, M0 = el; n = math.sqrt(GMB / a ** 3); out = []
    for m, ra, de, sg, site in P:
        o = NB["obs_cached"](m, site); tau = 0.
        for _ in range(2):
            r, _ = O.elements_to_rv(np.array([a]), np.array([e]), np.array([inc]), np.array([Om]), np.array([om]), np.array([M0 + n * (m - tau - ep)])); g = r[0] - o; tau = np.linalg.norm(g) / CAU
        ra_ = math.degrees(math.atan2(g[1], g[0])) % 360; de_ = math.degrees(math.asin(g[2] / np.linalg.norm(g)))
        out += [((ra_ - ra + 180) % 360 - 180) * math.cos(math.radians(de)) * 3600, (de_ - de) * 3600]
    return np.array(out)


def screen(pts, theirs, s0, ep):
    """Fast: one barycentric two-body orbit (frame: equatorial, as cand9_orbit) through both sets; returns rms (arcsec)."""
    from scipy.optimize import least_squares
    P = pts + theirs; sc = np.array([1e-3] * 3 + [1e-5] * 3)
    f = least_squares(lambda q: tb_resid(s0 + q * sc, ep, P), np.zeros(6), diff_step=1e-3, x_scale=1.0, max_nfev=200)
    r = tb_resid(s0 + f.x * sc, ep, P); return float(np.sqrt(np.mean(r ** 2)))


def identity(pts, theirs, s0, ep):
    """Their observations alone (from their catalogued orbit), then both sets. Returns dict."""
    return identity_f1(pts, theirs, s0, ep)[0]


def identity_f1(pts, theirs, s0, ep, f1=None):
    """identity(), also returning the fit of their observations alone (f1) so that subsets and shifted copies re-use it (2026-11-22)."""
    try:
        f1 = f1 or NB["fit"](theirs, s0, ep, max_nfev=120)
        f2 = NB["fit"](pts + theirs, np.array(f1["state"]), f1["epoch"], max_nfev=150); rs = np.hypot(*np.array(f2["resid"]).T)
    except Exception as ex:
        return dict(error=repr(ex)[:120]), f1
    o, t = rs[:len(pts)], rs[len(pts):]
    return dict(n_theirs=len(theirs), theirs_alone_rms=round(f1["rms"], 3), joint_rms=round(f2["rms"], 3), ours_rms=round(float(np.sqrt(np.mean(o ** 2))), 3), ours_max=round(float(o.max()), 3),
                theirs_max=round(float(t.max()), 3), same=bool(f2["rms"] <= .3 and rs.max() <= 1), borderline=bool(f2["rms"] <= 1 and not (f2["rms"] <= .3 and rs.max() <= 1))), f1


def power(P, theirs, f1, rms):
    """Can the n-body fit tell? The joint fit with our places P moved by 5 fixed sky offsets (nothing real there). Returns (rms list, number fitting
    within max(1", 2 x rms)). (Until 2026-11-22 this used the two-body screen, which fits a shifted copy as badly as the real long-gap link.)"""
    fk = []
    for ox, oy in SHIFTS:
        sh = [(m, ra + ox / 3600 / math.cos(math.radians(de)), de + oy / 3600, sg, site) for m, ra, de, sg, site in P]
        try:
            fk.append(round(NB["fit"](sh + theirs, np.array(f1["state"]), f1["epoch"], max_nfev=100)["rms"], 2))
        except Exception:
            fk.append(None)
    ref = max(1.0, 2 * rms); return fk, sum(1 for v in fk if v is not None and v <= ref)


NS = None


def _ns():
    """allarch/nbody_screen.py, loaded once (its RK4 integrator, Hermite interpolation with light time, two-body start states, Horizons planet
    vectors; nothing new is integrated here). Its planets() keeps the Horizons vectors in CACHE/planets/. Single-threaded."""
    global NS
    if NS is None:
        import importlib.util
        os.environ.setdefault("NUMBA_NUM_THREADS", "1")
        sp = importlib.util.spec_from_file_location("nbody_screen", C + "/allarch/nbody_screen.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
        m.OUT = cache_dir("planets"); NS = m
    return NS


def nb_places(el6, ep, targets):
    """All-planet n-body astrometric places of n MPCORB orbits (el6: rows M, peri, node, incl in deg, e, a; ep: their epochs, MJD) at targets
    [(mjd_utc, site)], as allarch/nbody_screen.py computes them (0.04-0.7" from Horizons/SkyBoT for good orbits): each orbit starts at the 2-day
    node nearest its epoch (two-body to it), is carried back and forth with the Sun and 8 planets, and is read at each target with light time.
    Returns ra, dec (deg), shape (n, len(targets)); NaN outside the planet table."""
    ns = _ns(); PP, VV = ns.planets(NB_P0 + 2400000.5, NB_NP, 1.0); HS = 2.0; J = (NB_NP - 1) // 2
    el6 = np.asarray(el6, float); ep = np.asarray(ep, float); n, nt = len(el6), len(targets)
    tt = np.atleast_1d(Time([t[0] for t in targets], format="mjd", scale="utc").tdb.mjd)
    Ro = np.array([NB["obs_cached"](m, s) for m, s in targets], float)
    jt = np.floor((tt - NB_P0) / HS).astype(int); tj = [j for j in range(nt) if 1 <= jt[j] <= J - 2]
    k0 = np.clip(np.rint((ep - NB_P0) / HS).astype(int), 1, J - 1)
    r = np.empty((n, 3)); v = np.empty((n, 3))
    for k in np.unique(k0):
        s = k0 == k; rr, vv = ns.helio_state(el6[s], NB_P0 + HS * k - ep[s]); r[s] = rr + PP[2 * k, 0]; v[s] = vv + VV[2 * k, 0]
    good = np.isfinite(r).all(1) & np.isfinite(v).all(1); r[~good] = 1e6; v[~good] = 0.0
    U = np.full((n, nt, 3), np.nan); ut = np.array([1.0, 0.0, 0.0])
    for d_ in (-1, 1):
        if not tj:
            break
        order = np.argsort(-k0 if d_ < 0 else k0, kind="stable"); ks = k0[order]; rs = r[order].copy(); vs = v[order].copy(); r1 = np.empty_like(rs); v1 = np.empty_like(vs)
        if d_ < 0:   # step k -> k-1 covers [t(k-1), t(k)): targets with jt == k-1, orbits starting at k or later
            k, stop = int(ks.max()), min(jt[j] for j in tj)
            while k > stop:
                na = int((ks >= k).sum())
                if na:
                    ns.rk4(rs, vs, na, PP[2 * k], PP[2 * k - 1], PP[2 * k - 2], ns.GM, -HS, r1, v1)
                    for j in tj:
                        if jt[j] == k - 1:
                            idx, res = ns.match(rs, vs, r1, v1, na, -HS, (tt[j] - (NB_P0 + HS * k)) / -HS, Ro[j], ut, -2.0, -2.0, na); U[order[idx], j] = res[:, 1:4]
                    rs[:na] = r1[:na]; vs[:na] = v1[:na]
                k -= 1
        else:        # step k -> k+1 covers [t(k), t(k+1)): targets with jt == k, orbits starting at k or earlier
            k, stop = int(ks.min()), max(jt[j] for j in tj)
            while k <= stop:
                na = int((ks <= k).sum())
                if na:
                    ns.rk4(rs, vs, na, PP[2 * k], PP[2 * k + 1], PP[2 * k + 2], ns.GM, HS, r1, v1)
                    for j in tj:
                        if jt[j] == k:
                            idx, res = ns.match(rs, vs, r1, v1, na, HS, (tt[j] - (NB_P0 + HS * k)) / HS, Ro[j], ut, -2.0, -2.0, na); U[order[idx], j] = res[:, 1:4]
                    rs[:na] = r1[:na]; vs[:na] = v1[:na]
                k += 1
    U[~good] = np.nan
    return np.degrees(np.arctan2(U[..., 1], U[..., 0])) % 360, np.degrees(np.arcsin(np.clip(U[..., 2], -1, 1)))


def night_of(m):
    return int(m - 0.6)          # as allarch/nbody_screen.py: a DECam night runs from about MJD x.6 to x+1.6


def eph_offsets(ra, de, pts, shift=(0, 0)):
    """Per-night median offset (arcsec) of n places (ra, de: (n, len(pts))) from our places, optionally moved by shift (east, north arcsec)."""
    P = np.array([p[1:3] for p in pts], float); cd = np.cos(np.radians(P[:, 1]))
    pa = P[:, 0] + shift[0] / 3600 / cd; pd = P[:, 1] + shift[1] / 3600
    s = np.hypot(((ra - pa + 180) % 360 - 180) * cd, de - pd) * 3600; nk = np.array([night_of(p[0]) for p in pts]); keys = sorted(set(nk))
    return keys, np.stack([np.median(s[:, nk == k], axis=1) for k in keys], 1)


def body_class(A, i):
    a = float(A["a"][i]); q = a * (1 - float(A["e"][i]))
    return "inner" if a <= 3.5 else "pert" if q < Q_TNO else "tno"


def psv_points(path):
    hd = None; out = []
    for l in open(path):
        if l.startswith(("trkSub", "permID", "provID")):
            hd = [x.strip() for x in l.split("|")]; continue
        if l.startswith(("#", "!")) or "|" not in l or not hd:
            continue
        r = dict(zip(hd, [x.strip() for x in l.split("|")]))
        out.append((Time(r["obsTime"].rstrip("Z"), format="isot", scale="utc").mjd, float(r["ra"]), float(r["dec"]), max(float(r["rmsRA"] or .1), .05), site_of(r["stn"], "DECam")))
    return sorted(out)


def own_check(pts):
    """Our own filed bodies: predict at the candidate's dates; identity fit when within 30"."""
    import hashlib
    reg = json.load(open(C + "/own_bodies.json")); cache_p = C + "/own_bodies_fit.json"; cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}; out = []
    for b in reg:
        P = sum((psv_points(C + "/" + f) for f in b["psv"]), []); key = hashlib.md5(json.dumps(P).encode()).hexdigest()
        if cache.get(b["name"], {}).get("key") != key:
            s0, ep = start_state(P); f = NB["fit"](P, s0, ep, max_nfev=200); cache[b["name"]] = dict(key=key, state=f["state"], epoch=f["epoch"], rms=f["rms"]); json.dump(cache, open(cache_p, "w"))
        c = cache[b["name"]]; pr = NB["predict"](c["state"], c["epoch"], [(p[0], p[4]) for p in pts])
        sep = max(math.hypot(((p[1] - q[0] + 180) % 360 - 180) * math.cos(math.radians(p[2])), p[2] - q[1]) * 3600 for p, q in zip(pts, pr))
        r = dict(name=b["name"], max_sep_arcsec=round(sep, 1), fit_rms=round(c["rms"], 3))
        if sep < 30:
            r.update(identity(pts, P, np.array(c["state"]), c["epoch"]))
        out.append(r)
    return out


def skybot(m, ra, de):
    q = {"-mime": "text", "-ra": f"{ra:.5f}", "-dec": f"{de:.5f}", "-rd": "0.17", "-ep": f"{m + 2400000.5:.5f}", "-loc": "W84", "-output": "all", "-objFilter": "110", "-filter": "0", "-refsys": "EQJ2000", "-from": "archive-cover"}
    try:
        t = urllib.request.urlopen(urllib.request.Request("https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?" + urllib.parse.urlencode(q), headers={"User-Agent": "archive-cover/1.0"}), timeout=120).read().decode("utf-8", "replace")
    except Exception as ex:
        return dict(error=repr(ex)[:80])
    rows = [l for l in t.splitlines() if "|" in l and not l.startswith("#")]
    far = [l[:140] for l in rows if not re.search(r"MB>|Hungaria|NEA|Mars-Crosser|Trojan|Hilda", l)]
    return dict(n=len(rows), distant=far)


def skybot_nights(pts, nmax=8):
    """2026-11-22 (TRKSUB01: C### = (591789) 2014 EM90 and C### = 2015 HX238, Jupiter Trojans, and C### = (459971) 2014 ON6
    passed as NOT IN MPC FILES: two-body MPCORB propagation misses perturbed Trojans by more than the match radius, and the
    old SkyBoT step dropped Trojan/Hilda/main-belt classes and was never part of the verdict). SkyBoT integrates orbits:
    a 72" cone at our place on up to nmax nights spread over the arc, every class kept; an object counts on a night when it
    sits within max(3", min(2 x its own uncertainty, 20")) of our place."""
    nights = {}
    for q in pts:
        nights.setdefault(int(q[0] + 0.5), q)
    keys = sorted(nights)
    if len(keys) > nmax:
        keys = [keys[round(i * (len(keys) - 1) / (nmax - 1))] for i in range(nmax)]
    out = []
    for k in keys:
        m, ra, de = nights[k][0], nights[k][1], nights[k][2]
        q = {"-mime": "text", "-ra": f"{ra:.5f}", "-dec": f"{de:.5f}", "-rd": "0.02", "-ep": f"{m + 2400000.5:.5f}", "-loc": "W84",
             "-output": "basic", "-filter": "0", "-refsys": "EQJ2000", "-from": "archive-cover"}
        try:
            t = urllib.request.urlopen(urllib.request.Request("https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?" + urllib.parse.urlencode(q),
                                                              headers={"User-Agent": "archive-cover/1.0"}), timeout=120).read().decode("utf-8", "replace")
        except Exception as ex:
            out.append(dict(mjd=round(m, 4), error=repr(ex)[:80])); continue
        hits = []
        for l in t.splitlines():
            if l.startswith("#") or "|" not in l:
                continue
            c = [x.strip() for x in l.split("|")]
            try:
                err, d_ = float(c[6]), float(c[7])
            except (ValueError, IndexError):
                continue
            if d_ <= max(3.0, min(2 * err, 20.0)):
                hits.append(dict(name=((c[0] + " ") if c[0] not in ("", "-") else "") + c[1], des=skybot_des(c), cls=c[4], d=d_, err=err))
        out.append(dict(mjd=round(m, 4), hits=hits)); time.sleep(1)
    return out


def skybot_des(c):
    """A designation from a SkyBoT row (novelty_fix): "(591789) 2014 EM90" for a numbered asteroid, the name alone for a planet or satellite
    (SkyBoT numbers Setebos 719, which is not the asteroid (719)), "29P/Name" for a numbered comet."""
    if c[0] in ("", "-") or re.search(r"Planet|Satellite|Moon", c[4]):
        return c[1]
    return f"{c[0]}/{c[1]}" if "omet" in c[4] else f"({c[0]}) {c[1]}"


def skybot_known(nights):
    ok = [x for x in nights if "hits" in x]
    cnt = {}
    for x in ok:
        for h in x["hits"]:
            c = cnt.setdefault(h["name"], dict(n=0, dmax=0.0, des=h["des"], cls=h["cls"]))
            c["n"] += 1; c["dmax"] = max(c["dmax"], h["d"])
    need = max(2, -(-3 * len(ok) // 5)) if len(ok) >= 2 else 99
    best = sorted(cnt.items(), key=lambda kv: -kv[1]["n"])
    known = [(n, c) for n, c in best if c["n"] >= need]
    partial = [(n, c) for n, c in best if 0 < c["n"] < need]
    return known, partial, len(ok)


def test_body(pts, f, A, eph, seasons, eph_fakes):
    """Step 3 (2026-11-22, novelty_fix) for one flagged body f; eph: its per-night n-body ephemeris offsets {"nights", "off"} or None.
    Returns the test record; "decision": EPHEMERIS | SAME | SUBSET (the same body on >= 2 and >= half of our nights) | MINORITY (on fewer) |
    BORDERLINE | NO | ERROR, "weak" from the power checks, "why" for the verdict."""
    r = dict(name=f["name"], cls=body_class(A, f["idx"]), flagged_by=f.get("how", "two-body"), sep_arcmin=f["sep_arcmin"])
    allk = sorted({night_of(p[0]) for p in pts}); nn = len(allk); need = max(2, -(-nn // 2))
    if eph is not None:
        ok = [o for o in eph["off"] if o == o]
        r["eph_offsets"] = {str(k): (None if o != o else round(float(o), 2)) for k, o in zip(eph["nights"], eph["off"])}
        r["eph_within"] = sum(1 for o in ok if o <= EPH_TOL); r["eph_near"] = sum(1 for o in ok if o <= EPH_SUB)
        if r["eph_within"] >= need:      # (a) decisive by itself: our places on its all-planet ephemeris
            other = [f"{k} ({o:.1f}\" off)" if o == o else f"{k}" for k, o in zip(eph["nights"], eph["off"]) if not o <= EPH_TOL]
            r.update(decision="EPHEMERIS", eph_fakes=eph_fakes, weak=eph_fakes >= 2,
                     why=f"n-body ephemeris of its MPCORB orbit, all planets: our places within {max(o for o in ok if o <= EPH_TOL):.2f}\" of it on {r['eph_within']} of {nn} nights"
                     + (f"; not this body on night(s) MJD {', '.join(other)}" if other else ""))
            return r
    try:
        th = obs80(get_obs(f["name"]))
    except Exception as ex:
        r.update(decision="ERROR", error=repr(ex)[:80]); return r
    if len(th) < 3:
        r.update(decision="ERROR", error=f"{len(th)} usable observations"); return r
    s0, ep0 = mpcorb_state(A, f["idx"], th[len(th) // 2][0]); r["n_theirs"] = len(th)
    try:
        rs = screen(pts, th, s0, ep0); r["screen_rms"] = round(rs, 2)
    except Exception:
        rs = None
    near = r.get("eph_near", 0) > 0     # (the two-body screen decides nothing beyond 3.5 au: a short arc's 2026 state often does not converge to our nights)
    if r["cls"] == "inner" and rs is not None and rs > INNER_SCREEN and not near:   # main belt: the Sun + giant planets fit cannot model it over years
        r.update(same=False, decision="NO", why=f"a <= 3.5 au: two-body screen {rs:.0f}\" and its ephemeris more than {EPH_SUB:.0f}\" off on every night"); return r
    q, f1 = identity_f1(pts, th, s0, ep0); r.update(q)
    if "error" in q:
        r["decision"] = "ERROR"; return r
    fit_ok = q["theirs_alone_rms"] <= ALONE_OK           # can this fit follow the body's own arc at all?
    if q["same"] or (q["borderline"] and fit_ok):
        r["fake_joint_rms"], r["fakes_fitting"] = power(pts, th, f1, q["joint_rms"]); r["weak"] = r["fakes_fitting"] >= 2
        if q["same"]:
            r.update(decision="SAME", why=f"joint rms {q['joint_rms']}\""); return r
    # (b) the whole card does not fit: the nights near its ephemeris alone, and each season alone where it was flagged within 3'
    subs = []
    if eph is not None and 2 <= r["eph_near"] < nn:
        keep = {k for k, o in zip(eph["nights"], eph["off"]) if o <= EPH_SUB}; subs.append(("ephemeris", [p for p in pts if night_of(p[0]) in keep]))
    if len(seasons) > 1 and ((f.get("season_mjd") and f["sep_arcmin"] < 3) or near):
        subs += [("season", sp) for sp in seasons if len({night_of(p[0]) for p in sp}) >= 2 and len(sp) < len(pts)]
    r["subsets"] = []
    for kind, P in subs:
        q2, _ = identity_f1(P, th, s0, ep0, f1); kn = sorted({night_of(p[0]) for p in P})
        rec = dict(kind=kind, mjd=[round(P[0][0], 1), round(P[-1][0], 1)], nights=len(kn), night_list=kn,
                   **{k: q2.get(k) for k in ("joint_rms", "ours_rms", "ours_max", "theirs_max", "same", "borderline", "error")})
        if q2.get("same"):
            rec["fake_joint_rms"], rec["fakes_fitting"] = power(P, th, f1, q2["joint_rms"]); rec["weak"] = rec["fakes_fitting"] >= 2
        r["subsets"].append(rec)
        if q2.get("same") and not rec.get("weak") and len(kn) >= need:
            break
    r["season_tested"] = any(s["kind"] == "season" for s in r["subsets"])
    good = [s for s in r["subsets"] if s.get("same")]
    if good:
        s = min(good, key=lambda s: (bool(s.get("weak")), -s["nights"])); others = [k for k in allk if k not in s["night_list"]]
        eo = r.get("eph_offsets") or {}
        txt = (f"joint rms {s['joint_rms']}\" with our places on {s['nights']} of {nn} nights, ours max {s['ours_max']}\"; not this body on night(s) MJD "
               + ", ".join(f"{k}" + (f" ({eo[str(k)]:.1f}\" off its ephemeris)" if eo.get(str(k)) is not None else "") for k in others))
        r.update(decision="SUBSET" if s["nights"] >= need else "MINORITY", weak=bool(s.get("weak")), fakes_fitting=s.get("fakes_fitting"), why=txt)
        return r
    if fit_ok:
        r["decision"] = "BORDERLINE" if q["borderline"] or any(s.get("borderline") for s in r["subsets"]) else "NO"; return r
    # its own observations do not fit (Trojans, Centaurs, old photographic arcs): a borderline or failed joint fit says nothing by itself
    alone = q["theirs_alone_rms"]
    if q["joint_rms"] > max(3 * alone, alone + 2):
        r.update(decision="NO", why=f"joint rms {q['joint_rms']}\" against {alone}\" for its own observations"); return r
    if eph is not None and not A["poor"][f["idx"]] and r["eph_near"] == 0:
        r.update(decision="NO", why=f"good orbit, its n-body ephemeris more than {EPH_SUB:.0f}\" off on every night"); return r
    r.update(decision="BORDERLINE", why=f"its own observations leave {alone}\" in this fit: it cannot tell"); return r


def main(spec, out):
    S = json.load(open(spec)); t0 = time.time(); d = mpc_dir(); pts = sorted(spec_points(S)); R = dict(id=S.get("id"), spec=S, mpc_files=os.path.basename(d), points=pts)
    mid = pts[len(pts) // 2]; arc = pts[-1][0] - pts[0][0]
    if len(pts) >= 2 and arc > 0:
        k = (pts[-1][1] - pts[0][1] + 180) % 360 - 180; rate = [k * math.cos(math.radians(mid[2])) * 3600 / (arc * 24), (pts[-1][2] - pts[0][2]) * 3600 / (arc * 24)]
    else:
        rate = [0., 0.]
    R["rate"] = [round(v, 3) for v in rate]; R["arc_days"] = round(arc, 2)
    try:
        R["own"] = own_check(pts)
    except Exception as ex:
        R["own"] = [dict(error=repr(ex)[:120])]
    A = load_mpcorb(d)
    seasons = [[pts[0]]]
    for p_ in pts[1:]:
        if p_[0] - seasons[-1][-1][0] > 60:
            seasons.append([p_])
        else:
            seasons[-1].append(p_)
    flags = []; R["seasons"] = len(seasons); season_rates = []
    for sp in seasons:
        ms_ = sp[len(sp) // 2]
        # the rate from the points within 6 days of the season's middle point (2026-11-19: a mean rate over months differs from the
        # instantaneous one by more than the 1.5"/h tolerance, so the Centaur 2010 FB49 was never flagged); the whole season if those span < 0.02 d
        loc = [q for q in sp if abs(q[0] - ms_[0]) <= 6] 
        loc = loc if loc[-1][0] - loc[0][0] > 0.02 else sp; arc_s = loc[-1][0] - loc[0][0]
        if arc_s > 0.02:
            k_ = (loc[-1][1] - loc[0][1] + 180) % 360 - 180; rate_s = [k_ * math.cos(math.radians(ms_[2])) * 3600 / (arc_s * 24), (loc[-1][2] - loc[0][2]) * 3600 / (arc_s * 24)]
        elif len(seasons) == 1:
            rate_s = rate
        else:
            continue  # one point, no rate of its own
        season_rates.append((ms_, rate_s))
        for f in flag_at(A, ms_[0], ms_[1], ms_[2], rate_s):
            if f["name"] not in {x["name"] for x in flags}:
                f["season_mjd"] = round(ms_[0], 1); flags.append(f)
    fits = None
    if arc > 300:  # long arc: also the Rubin era
        try:
            s0, ep = (np.array(S["state"], float), S["epoch"]) if S.get("state") else start_state(pts); fits = NB["fit"](pts, s0, ep, max_nfev=150)
            for m in np.arange(Time("2024-01-01").mjd, min(Time.now().mjd, NB["GRID"][-1] - 2), 4.):
                p = NB["predict"](fits["state"], fits["epoch"], [(m, "DECam"), (m + 1, "DECam")]); r_ = [((p[1][0] - p[0][0] + 180) % 360 - 180) * math.cos(math.radians(p[0][1])) * 150, (p[1][1] - p[0][1]) * 150]
                for f in flag_at(A, m, p[0][0], p[0][1], r_, r_poor=5.):  # in the Rubin era both orbits are good: 5' is generous
                    if f["name"] not in {x["name"] for x in flags}:
                        flags.append(f)
        except Exception as ex:
            R["rubin_era_error"] = repr(ex)[:120]
    # 1b (2026-11-22, novelty_fix): all-planet n-body places of every MPCORB orbit beyond 3.5 au and of every body flagged so far, at each of our
    # points and one day after each season's middle point: flags as step 1 does, and gives each flagged body its offsets night by night
    EPH = {}; eph_fakes = 0; t_nb = time.time()
    try:
        sel = np.union1d(np.where(A["ok"] & (A["a"] > 3.5))[0], np.array([f["idx"] for f in flags], int)).astype(int)
        raw = np.array([A["raw"][i] for i in sel], float); npt = len(pts); pidx = {id(p): j for j, p in enumerate(pts)}
        ra_, de_ = nb_places(raw[:, [0, 1, 2, 3, 4, 6]], raw[:, 7], [(p[0], p[4]) for p in pts] + [(sm[0] + 1., sm[4]) for sm, _ in season_rates])
        have = {x["name"] for x in flags}; nbf = 0
        with np.errstate(invalid="ignore"):
            for k_, (sm, rs_) in enumerate(season_rates):
                j0, j1 = pidx[id(sm)], npt + k_; cd = math.cos(math.radians(sm[2]))
                sep = np.hypot(((ra_[:, j0] - sm[1] + 180) % 360 - 180) * cd, de_[:, j0] - sm[2]) * 60
                rr = np.c_[((ra_[:, j1] - ra_[:, j0] + 180) % 360 - 180) * np.cos(np.radians(de_[:, j0])) * 150, (de_[:, j1] - de_[:, j0]) * 150]
                dr = np.max(np.abs(rr - np.asarray(rs_)[None]), axis=1); lim = np.where(A["poor"][sel], R_POOR, R_GOOD)
                for j in np.where((dr < RATE_TOL) & (sep < lim))[0][np.argsort(sep[(dr < RATE_TOL) & (sep < lim)])]:
                    i = int(sel[j])
                    if A["name"][i] not in have:
                        have.add(A["name"][i]); nbf += 1
                        flags.append(dict(idx=i, name=A["name"][i], sep_arcmin=round(float(sep[j]), 2), rate_diff=round(float(dr[j]), 2), a=round(float(A["a"][i]), 2), e=round(float(A["e"][i]), 3),
                                          i=round(float(np.degrees(A["inc"][i])), 2), U=A["U"][i], arc=A["arc"][i], date=Time(sm[0], format="mjd").iso[:10], season_mjd=round(sm[0], 1), how="n-body"))
            keys, off = eph_offsets(ra_[:, :npt], de_[:, :npt], pts); pos = {int(i): j for j, i in enumerate(sel)}
            need = max(2, -(-len(keys) // 2))
            for j in np.where(np.sum(off <= EPH_TOL, axis=1) >= need)[0]:     # on its ephemeris on enough nights, whatever step 1's rate says
                i = int(sel[j])
                if A["name"][i] not in have:
                    have.add(A["name"][i]); nbf += 1
                    flags.append(dict(idx=i, name=A["name"][i], sep_arcmin=round(float(np.nanmin(off[j])) / 60, 2), rate_diff=None, a=round(float(A["a"][i]), 2), e=round(float(A["e"][i]), 3),
                                      i=round(float(np.degrees(A["inc"][i])), 2), U=A["U"][i], arc=A["arc"][i], date=Time(pts[0][0], format="mjd").iso[:10], how="n-body ephemeris"))
            for f in flags:
                if f["idx"] in pos:
                    EPH[f["name"]] = dict(nights=keys, off=[float(x) for x in off[pos[f["idx"]]]])
            # control for the ephemeris test: copies of our places shifted by 2-5' (nothing real there); how many find some body of this set on it
            for sh in SHIFTS:
                eph_fakes += int(np.any(np.sum(eph_offsets(ra_[:, :npt], de_[:, :npt], pts, sh)[1] <= EPH_TOL, axis=1) >= need))
        R["nbody"] = dict(orbits=len(sel), flagged=nbf, ephemeris_fakes=eph_fakes, seconds=round(time.time() - t_nb))
    except (Exception, SystemExit) as ex:     # (nbody_screen.planets exits when Horizons fails)
        R["nbody_error"] = repr(ex)[:150]
    flags.sort(key=lambda f: f["sep_arcmin"]); R["flagged"] = flags[:40]; R["n_flagged"] = len(flags)
    # ITF tracklets near our nights
    itf = {}; months = set()
    for p in pts:
        for dd in (-1, 0, 1):
            t_ = datetime.date.fromordinal(int(p[0] + dd) + _MJD0); months.add(f"{t_.year:04d} {t_.month:02d}")
    with gzip.open(d + "/itf.txt.gz", "rt", errors="replace") as fh:
        for l in fh:
            if len(l) < 80 or l[14] in "sv" or l[15:22] not in months:
                continue
            try:
                m = fast_obs_date(l)
            except Exception:
                continue
            for p in pts:
                if abs(m - p[0]) < .6:
                    ra, de = M.radec(l); m0 = m - p[0]; a0 = p[1] + rate[0] * m0 * 24 / 3600 / math.cos(math.radians(p[2])); d0 = p[2] + rate[1] * m0 * 24 / 3600
                    if math.hypot(((ra - a0 + 180) % 360 - 180) * math.cos(math.radians(d0)), de - d0) * 60 < R_ITF:
                        itf.setdefault(l[5:12].strip() or l[:12].strip(), []).append(l.rstrip())
                    break
    trk = []
    for k, v in itf.items():
        v.sort(key=fast_obs_date); m1, m2 = fast_obs_date(v[0]), fast_obs_date(v[-1])
        if len(v) < 2 or m2 - m1 <= 0:
            trk.append(dict(trk=k, n=len(v), first=v[0][15:32], rate=None, lines=v)); continue
        a1, d1 = M.radec(v[0]); a2, d2 = M.radec(v[-1]); rr = [((a2 - a1 + 180) % 360 - 180) * math.cos(math.radians(d1)) * 3600 / ((m2 - m1) * 24), (d2 - d1) * 3600 / ((m2 - m1) * 24)]
        if max(abs(rr[0] - rate[0]), abs(rr[1] - rate[1])) < RATE_TOL:
            trk.append(dict(trk=k, n=len(v), first=v[0][15:32], rate=[round(x, 2) for x in rr], lines=v))
    # 2026-11-17: the ITF along the carried orbit as well, 2013 to now (LEAD12's 2025 Rubin tracklets RUBIN_TRKSUB were missed by the
    # 0.6-day window above); any observation within 30" of the long-arc orbit's place at its own time is reported for a look by hand
    if fits is not None:
        try:
            g = np.arange(max(Time("2013-01-01").mjd, NB["GRID"][0] + 2), min(Time.now().mjd, NB["GRID"][-1] - 2), 2.)
            G = np.array(NB["predict"](fits["state"], fits["epoch"], [(m, "DECam") for m in g]), float)
            gr, gd = (G[:, 0], G[:, 1]) if G.shape[0] == len(g) else (G[0], G[1])
            cand = []
            with gzip.open(d + "/itf.txt.gz", "rt", errors="replace") as fh:
                for l in fh:
                    if len(l) < 80 or l[14] in "sv" or not l[15:19].isdigit() or int(l[15:19]) < 2013:
                        continue
                    try:
                        m = fast_obs_date(l); k = min(max(int(np.searchsorted(g, m)), 1), len(g) - 1); ra, de = M.radec(l)
                    except Exception:
                        continue
                    if abs(de - gd[k]) < .3 and abs(((ra - gr[k] + 180) % 360 - 180) * math.cos(math.radians(de))) < .3:
                        cand.append((m, ra, de, l))
            if cand:
                Q = np.array(NB["predict"](fits["state"], fits["epoch"], [(c[0], site_of(c[3][77:80]) or site_of("500")) for c in cand]), float)
                qa, qd = (Q[:, 0], Q[:, 1]) if Q.shape[0] == len(cand) else (Q[0], Q[1]); path = {}
                for (m, ra, de, l), a_, d_ in zip(cand, qa, qd):
                    sep = math.hypot(((ra - a_ + 180) % 360 - 180) * math.cos(math.radians(d_)), de - d_) * 3600
                    if sep < 30:
                        path.setdefault(l[5:12].strip() or l[:12].strip(), []).append((round(sep, 2), l.rstrip()))
                for k_, v in path.items():
                    if k_ not in {t["trk"] for t in trk}:
                        trk.append(dict(trk=k_, n=len(v), first=v[0][1][15:32], rate=None, along_path=True, max_sep_arcsec=max(x[0] for x in v), lines=[x[1] for x in v]))
            R["itf_path_scanned"] = len(cand)
        except Exception as ex:
            R["itf_path_error"] = repr(ex)[:150]
    R["itf"] = trk
    # identity tests (2026-11-22, novelty_fix: every flagged body, no cap of 20; the ephemeris test first, then the n-body fit; see test_body)
    tests = []; t_id = time.time(); untested = []
    rank = lambda f: (not ((EPH.get(f["name"]) and sum(1 for o in EPH[f["name"]]["off"] if o <= EPH_TOL) >= max(2, -(-len(EPH[f["name"]]["nights"]) // 2)))),
                      not (EPH.get(f["name"]) and any(o <= EPH_SUB for o in EPH[f["name"]]["off"])), f["sep_arcmin"])
    for f in sorted(flags, key=rank):
        if time.time() - t_id > TBUDGET:
            untested.append(f["name"]); continue
        r = test_body(pts, f, A, EPH.get(f["name"]), seasons, eph_fakes)
        tests.append(r)
        print(S.get("id"), "vs", f["name"], {k: v for k, v in r.items() if k not in ("eph_offsets", "subsets")}, flush=True)
        if r.get("decision") in ("SAME", "SUBSET", "EPHEMERIS") and not r.get("weak"):
            break
    R["seconds_tests"] = round(time.time() - t_id)
    if untested:
        R["untested_time_budget"] = untested
    R["tests"] = tests
    # 2026-11-22 (TRKSUB01 identity, crashfix): every identity test that did not run (fit crash, HTTP error or timeout, unreadable reply, too few
    # usable observations, a failed season / ephemeris-night fit, the time budget) is listed; such a lead can never be NOT IN MPC FILES (verdict below)
    R["untested"] = [dict(body=t["name"], reason=str(t.get("error"))[:90]) for t in tests if t.get("decision") == "ERROR"]
    R["untested"] += [dict(body=t["name"], reason=f"{s['kind']} fit of MJD {s['mjd'][0]}-{s['mjd'][1]}: {s['error']}"[:90]) for t in tests for s in t.get("subsets") or [] if s.get("error")]
    R["untested"] += [dict(body=n, reason=f"time budget ({TBUDGET:.0f} s) used up") for n in untested]
    # 2026-11-22 (TRKSUB01 openitems): a screening step that failed is an untested identity too. Until then a failed step 1b (nbody_error: no
    # Trojan / Hilda / Centaur flags and no ephemeris test; on the VM it failed silently for numba reasons), a failed step 0 (own filed bodies),
    # a failed Rubin-era screen or ITF path scan (step 5), or SkyBoT 4b answering on too few nights, left a lead NOT IN MPC FILES without a word.
    for k_, what in (("nbody_error", "all-planet n-body screen (step 1b)"), ("rubin_era_error", "Rubin-era screen (step 5)"),
                     ("itf_path_error", "ITF along the carried orbit (step 5)")):
        if R.get(k_):
            R["untested"].append(dict(body=what, reason=str(R[k_])[:90]))
    _own_err = [o for o in R.get("own") or [] if isinstance(o, dict) and o.get("error")]
    if _own_err:
        R["untested"].append(dict(body="our own filed bodies (step 0)", reason=str(_own_err[0]["error"])[:90]))
    R["skybot"] = [dict(date=Time(p[0], format="mjd").iso[:10], **skybot(p[0], p[1], p[2])) for p in (pts[0], pts[-1])]
    try:
        R["skybot_nights"] = skybot_nights(pts)
        sk_known, sk_partial, sk_n = skybot_known(R["skybot_nights"])
    except Exception as ex:
        R["skybot_nights"] = [dict(error=repr(ex)[:100])]; sk_known, sk_partial, sk_n = [], [], 0
    # (openitems) SkyBoT per night: an HTTP 204 is an answer (an empty cone; urlopen returns an empty body); a night with an error is not.
    # Untested when nights failed and fewer than 2, or fewer than half of the nights asked, were answered (4b then cannot reach its 2-night bar).
    _sk_ask = len(R["skybot_nights"]); _sk_bad = [x for x in R["skybot_nights"] if "error" in x]
    if _sk_bad and (_sk_ask - len(_sk_bad) < 2 or 2 * (_sk_ask - len(_sk_bad)) < _sk_ask):
        R["untested"].append(dict(body="SkyBoT per night (step 4b)", reason=f"{len(_sk_bad)} of {_sk_ask} nights failed: {_sk_bad[0]['error']}"[:90]))
    R["n_untested"] = len(R["untested"])
    # verdict (2026-11-22, novelty_fix: from each test's "decision", see test_body)
    dec = lambda t: t.get("decision") or ("SAME" if t.get("same") else "BORDERLINE" if t.get("borderline") else "ERROR" if "error" in t else "NO")
    same = [t for t in tests if dec(t) in ("SAME", "EPHEMERIS", "SUBSET") and not t.get("weak")]; probably = [t for t in tests if dec(t) in ("SAME", "EPHEMERIS", "SUBSET") and t.get("weak")]
    minority = [t for t in tests if dec(t) == "MINORITY" and not t.get("weak")]
    border = [t for t in tests if (dec(t) == "BORDERLINE" and not t.get("weak")) or (dec(t) == "ERROR" and any(f["name"] == t["name"] and f["sep_arcmin"] < 10 for f in flags))]
    border += [dict(name="ITF " + t["trk"]) for t in trk]
    R["weak_links"] = [f"{t['name']} ({t.get('fakes_fitting', t.get('eph_fakes'))} of 5 shifted copies fit as well)" for t in tests if t.get("weak") and dec(t) not in ("SAME", "EPHEMERIS", "SUBSET")]
    tested = {t["name"] for t in tests}
    if not same:
        border += [dict(name=f["name"] + " (untested)") for f in flags if f["name"] not in tested and f["sep_arcmin"] < 10]
        # a body on its all-planet ephemeris on some nights but fewer than the test needs (one night of a mixed card; no commas: resolve_hand splits on them)
        border += [dict(name=f"{t['name']} (n-body ephemeris within {EPH_TOL:.0f}\" of our places on {t['eph_within']} of {len(t['eph_offsets'])} nights)")
                   for t in tests if t.get("eph_within") and dec(t) in ("NO", "ERROR") and not any(x is t for x in border)]
        if R.get("seasons", 1) > 1:  # 2026-11-19: a known body at one season of a multi-season arc (the arc may join it to other detections); since novelty_fix only when the seasons were not fitted
            border += [dict(name=f"{f['name']} (within {f['sep_arcmin']:.1f}' and moving alike in the season of {Time(f['season_mjd'], format='mjd').iso[:10]}; the whole arc does not fit it)")
                       for f in flags if f.get("season_mjd") and f["sep_arcmin"] < 3 and f["name"] in tested and not any(t["name"] == f["name"] and (dec(t) != "NO" or t.get("season_tested") or t.get("why", "").startswith("good orbit")) for t in tests)
                       and not any(str(x.get("name", "")).startswith(f["name"]) for x in border)]
    ours = [o for o in R["own"] if o.get("same")]
    if ours:
        R["verdict"] = f"OURS {ours[0]['name']} (our filing, joint rms {ours[0]['joint_rms']}\")"
    elif same:
        R["verdict"] = f"KNOWN {same[0]['name']} ({same[0].get('why') or 'joint rms ' + str(same[0].get('joint_rms')) + chr(34)})"
    elif sk_known:
        n_, c_ = sk_known[0]
        R["verdict"] = f"KNOWN {c_['des']} (SkyBoT: {c_['cls']}, within {c_['dmax']:.1f}\" of our places on {c_['n']} of {sk_n} nights)"
    elif probably:
        t = probably[0]; R["verdict"] = f"PROBABLY {t['name']} ({t.get('why') or 'joint rms ' + str(t.get('joint_rms')) + chr(34)}, but {t.get('fakes_fitting', t.get('eph_fakes'))} of 5 shifted copies fit too: needs a third epoch)"
    elif minority:
        t = minority[0]; R["verdict"] = f"UNSURE: {t['name']} is the same body on some nights only ({t['why']}): a mixed card"
    elif border or sk_partial:
        border += [dict(name=c_["des"]) for n_, c_ in sk_partial if not any(str(t.get("name", "")).startswith(c_["des"]) for t in border)]
        R["verdict"] = "CHECK BY HAND: " + ", ".join(t["name"] for t in border)
    else:
        R["verdict"] = f"NOT IN MPC FILES OF {os.path.basename(d)}"
    R["verdict"] = guard_untested(R)
    if OBS80_SKIPPED:       # (obscode) MPC lines from space / roving observers left out of the fits, by code
        R["space_obs_left_out"] = dict(OBS80_SKIPPED)
    if OBS80_SPACE:         # (satobs) spacecraft / roving lines fitted from their own place, by code
        R["space_obs_used"] = dict(OBS80_SPACE)
    R["seconds"] = round(time.time() - t0)
    json.dump(R, open(out, "w"), indent=0, default=float); print(S.get("id"), "->", R["verdict"], f"({len(flags)} flagged, {len(trk)} ITF tracklets, {R['seconds']} s)", flush=True)
    if R["verdict"].startswith("CHECK BY HAND"):  # 2026-11-22
        try:
            import resolve_hand
            R["verdict"] = resolve_hand.apply(out); print(S.get("id"), "-> resolved:", R["verdict"], flush=True)
        except Exception as ex:
            R["verdict"] = "UNSURE: automatic resolve failed " + repr(ex)[:100]; R["verdict_before_resolve"] = "CHECK BY HAND"
            json.dump(R, open(out, "w"), indent=0, default=float)
        if str(R["verdict"]).startswith("NOT IN MPC FILES") and R["untested"]:     # (crashfix) resolve_hand retests the bodies it names
            Rf = json.load(open(out)); done_ = {t.get("trk") for t in Rf.get("resolve") or [] if t.get("result") in ("SAME", "WEAK", "NOT SAME", "BORDERLINE")}
            Rf["untested"] = R["untested"] = [x for x in R["untested"] if x["body"] not in done_]; Rf["n_untested"] = len(R["untested"])
            Rf["verdict"] = R["verdict"] = guard_untested(Rf); tmp = out + ".tmp"; json.dump(Rf, open(tmp, "w"), indent=0, default=float); os.replace(tmp, out)
            print(S.get("id"), "-> after the untested check:", R["verdict"], flush=True)
    return R


def guard_untested(R):
    """(crashfix, 2026-11-22) A lead with an identity test that did not run is never NOT IN MPC FILES: UNSURE, naming the tests."""
    v, u_ = str(R.get("verdict", "")), R.get("untested") or []
    if not v.startswith("NOT IN MPC FILES") or not u_:
        return v
    R["verdict_before_untested"] = v
    return (f"UNSURE ({len(u_)} identity test{'s' if len(u_) > 1 else ''} did not run: " + "; ".join(f"{x['body']} ({x['reason']})" for x in u_)
            + f"; every other flagged body tested, none the same body, MPC files of {R.get('mpc_files', '?')})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
