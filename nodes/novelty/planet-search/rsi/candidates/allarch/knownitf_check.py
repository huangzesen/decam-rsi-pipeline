"""Cross-check of "KNOWN ITF <trk>" novelty tags against the card's best orbit (2026-11-22; <coordinator-session>: resolve_hand fits only the lead's
own spec points, so a short arc with a loose plane can absorb a tracklet off the track, as C###'s ITF C1M76CP). For every active card whose
novelty verdict is KNOWN ITF <trk>, the tracklet's ITF lines (MPC file of 2026-11-20) are tested with nsc_path's Orbit.identity against the
card's best orbit: allarch/joint/C###_joint.json, else the refit, else the arc (covariance as the prior). Prints AGREES / CONTRADICTS /
UNTESTABLE per card and writes knownitf_check.json. A CONTRADICTS (NOT THE SAME BODY with prior chi2 > 100) needs a hand verdict
(novelty_hand.json + apply_hand.py), with the cross-track offset from the nominal prediction as the evidence.
Usage: OMP_NUM_THREADS=1 ~/anaconda3/bin/python -I knownitf_check.py [C### ...]"""
import datetime, glob, gzip, json, os, re, sys
H = os.path.dirname(os.path.abspath(__file__)); R = "$ALE_HOME/rsi/decam-big-tno"; D = H + "/../novelty_queue/done/"
want = set(sys.argv[1:])
src = open(H + "/nsc_path.py").read().split("\nwant = sys.argv[2:]")[0]
NP = {"__name__": "nscp", "__file__": H + "/nsc_path.py"}; sys.argv = ["x", "bodies_u16.json"]; exec(compile(src, "nsc_path.py", "exec"), NP); Orbit = NP["Orbit"]
from astropy.coordinates import EarthLocation
import astropy.units as u
S_ = NP["OF"].SITES
for code, lon, lat, h in (("V00", -111.600, 31.963, 2120), ("695", -111.600, 31.958, 2120), ("F51", -156.257, 20.708, 3067), ("F52", -156.257, 20.708, 3067),
                          ("T09", -155.476, 19.826, 4163), ("T14", -155.469, 19.825, 4204), ("309", -70.404, -24.627, 2635), ("X05", -70.749, -30.244, 2663),
                          ("G96", -110.789, 32.443, 2790), ("703", -110.732, 32.442, 2510), ("I41", -116.862, 33.356, 1712), ("P07", -116.862, 33.356, 1712)):
    if code not in S_: S_[code] = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, h * u.m)
# (satobs gap, TRKSUB01 u20vet 2026-11-23) every ITF line is fitted from its own place: ground codes from the MPC's parallax constants
# (candidates/mpc_obscodes.json, as novelty_check.site_of), 'S' spacecraft lines from the geocentric vector of their 's' line (MPC 80-column
# satellite format), 'V' roving lines from their 'v' line; deleted / replaced discovery lines ('x', 'X') and radar lines ('R', 'r') are left out.
# Until now the ITF loop below dropped 'sSrRvxX' lines, the lines resolve_hand tested were all kept (deleted ones too), and codes missing from
# the short table above were fitted from DECam's site (W84).
import math as _m
_OBSC = json.load(open(H + "/../mpc_obscodes.json")); _OF = NP["OF"]; _SATV = {}; _ob0 = _OF.observer_bary
S_["GEO"] = EarthLocation.from_geocentric(0, 0, 0, unit=u.km)
def _observer_bary(m, site):
    if isinstance(site, str) and site in _SATV:
        R_, t_ = _ob0(m, "GEO"); return R_ + _SATV[site], t_
    return _ob0(m, site)
_OF.observer_bary = _observer_bary
def site_key(lines, j):
    """Observer key for line j of a tracklet (None: leave the line out)."""
    l = lines[j]; code, st = l[14], l[77:80]
    if code in "svxXrR": return None
    if code in "SV":
        nx = lines[j + 1] if j + 1 < len(lines) else ""
        if len(nx) < 80 or nx[14] != code.lower(): return None
        try:
            if code == "S":
                import numpy as _np
                v = _np.array([float(nx[a:b].replace(" ", "")) for a, b in ((33, 45), (45, 57), (57, 69))]); v = v / 149597870.7 if nx[32] == "1" else v
                k = "SAT " + st + " " + nx[32:69]; _SATV[k] = v; return k
            lon, lat, alt = (float(x) for x in nx[32:77].split()[:3]); k = f"ROV {lon:.5f} {lat:.5f} {alt:.0f}"
            S_.setdefault(k, EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, alt * u.m)); return k
        except (ValueError, IndexError):
            return None
    if st in S_: return st
    c = _OBSC.get(st)
    if not c or c[0] is None: return None
    lon = _m.radians(c[0]); S_[st] = EarthLocation.from_geocentric(c[1] * _m.cos(lon) * 6378.137, c[1] * _m.sin(lon) * 6378.137, c[2] * 6378.137, unit=u.km); return st
LEFT_OUT = {}
MJD0 = datetime.date(1858, 11, 17).toordinal(); ITF = H + "/../../data/mpc/2026-11-20/itf.txt.gz"
cards = []
for f in sorted(glob.glob(f"{R}/candidates/C[0-9][0-9][0-9]/card.json")):
    d = json.load(open(f)); C = f.split("/")[-2]
    if not d.get("active") or not d.get("key") or (want and C not in want): continue
    try: v = json.load(open(D + d["key"] + ".result.json"))["verdict"]
    except Exception: continue
    m = re.match(r"KNOWN ITF (\S+)", v)
    if m: cards.append((C, d["key"], (d.get("identities") or {}).get("run"), m.group(1)))
need = {c[3] for c in cards}; raw = {}; obs = {}
with gzip.open(ITF, "rt", errors="replace") as fh:
    for l in fh:
        if len(l) < 80 or l[5:12].strip() not in need: continue
        raw.setdefault(l[5:12].strip(), []).append(l.rstrip("\n"))
def to_points(lines, trk):
    P = []
    for j, l in enumerate(lines):
        if l[14] in "sv": continue
        k = site_key(lines, j)
        if k is None:
            LEFT_OUT.setdefault(trk, []).append(l[14] + " " + l[77:80]); continue
        try:
            mj = datetime.date(int(l[15:19]), int(l[20:22]), 1).toordinal() - MJD0 + float(l[23:32]) - 1
            ra = 15 * (int(l[32:34]) + int(l[35:37]) / 60 + float(l[38:44]) / 3600); de = (-1 if l[44] == "-" else 1) * (int(l[45:47]) + int(l[48:50]) / 60 + float(l[51:56]) / 3600)
        except ValueError: continue
        P.append((mj, ra, de, k))
    return P
for t_, L_ in raw.items(): obs[t_] = to_points(L_, t_)
out = {}
for C, k, run, trk in cards:
    j = os.environ.get("KI_JOINT_DIR", H + "/joint") + f"/{C}_joint.json" + os.environ.get("KI_SUFFIX", ""); rf = f"{R}/runs/{run}/arc/{k}.refit.json"; af = f"{R}/runs/{run}/arc/{k}.json"
    base = j if os.path.exists(j) else (rf if os.path.exists(rf) else af)
    P = []  # the exact lines resolve_hand tested (trkSubs are not unique across stations), else the ITF scan
    for x in json.load(open(D + k + ".result.json")).get("itf") or []:
        if x.get("trk") == trk:
            P += to_points(x["lines"], trk)
    if not P: P = obs.get(trk, [])
    stations = sorted({p[3] for p in P})
    if not P or not os.path.exists(base):
        out[C] = dict(trk=trk, result="UNTESTABLE (no ITF lines or no orbit)"); print(C, trk, out[C]["result"]); continue
    try:
        r = Orbit(base).identity(P)
    except Exception as ex:
        out[C] = dict(trk=trk, result=f"UNTESTABLE ({ex!r})"[:80]); print(C, trk, out[C]["result"]); continue
    res = "AGREES" if r["verdict"] == "SAME BODY" else ("UNTESTABLE (numerical failure: fit diverged)" if (r.get("rms") or 0) > 100 else ("CONTRADICTS" if r["verdict"].startswith("NOT THE SAME") and (r.get("prior_chi2") or 0) > 100 else "UNTESTABLE / BORDERLINE"))
    out[C] = dict(trk=trk, stations=stations, base=os.path.basename(base), n=len(P), verdict=r["verdict"], rms=r["rms"], max=r["max"], prior_chi2=r["prior_chi2"], result=res,
                  left_out=LEFT_OUT.get(trk, []), kinds=sorted({"spacecraft" if str(p[3]).startswith("SAT") else "roving" if str(p[3]).startswith("ROV") else "ground" for p in P}))
    print(C, trk, res, "|", r["verdict"], "rms", r["rms"], "prior", r["prior_chi2"], "| orbit", os.path.basename(base)[-26:], flush=True)
json.dump(out, open(H + "/knownitf_check.json", "w"), indent=1)
print({x: sum(1 for v in out.values() if v["result"].startswith(x)) for x in ("AGREES", "CONTRADICTS", "UNTESTABLE")})
