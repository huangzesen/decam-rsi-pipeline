"""Could a candidate be a body the MPC already has, whose orbit is too poor to place it in 2014 or 2018? For every a > 25 au body whose observed arc does not span the
candidate's year: is the candidate in its orbit plane at a sensible distance? If so, fit one orbit through the body's published observations and the candidate's
night positions. A body that fits is then looked for at that orbit's predicted places in other images, which neither side has used. Usage: python -I suspects.py DATADIR OUTDIR
"""
import gzip, json, math, os, re, subprocess, sys, time
import numpy as np
H = os.path.expanduser("$ALE_HOME/planet-search/rsi")
sys.path.insert(0, H + "/candidates")
import confirm as C, confirm2 as C2, mpc_check as M, scan as S
G = C.G
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
import astropy.units as u
from scipy.optimize import least_squares
ARC = C.ARC
RUNOFF = {"0": 1.0, "1": 4.4, "2": 19.6, "3": 86.5, "4": 382, "5": 1692, "6": 7488, "7": 33121, "8": 146502, "9": 648000}


def obs_b(m):
    """Barycentric observer (DECam's site is close enough for all sites at these distances: under 0.4"): TNOs orbit the barycentre, and over 12 years the Sun's wobble matters."""
    t = Time(m, format="mjd", scale="utc")
    with solar_system_ephemeris.set("builtin"):
        e = get_body_barycentric("earth", t).xyz.to(u.au).value
    return e + C.SITE.get_gcrs_posvel(t)[0].xyz.to(u.au).value


def get_obs(desig):
    d = re.sub(r"^\(\d+\)\s*", "", desig).strip()
    out = subprocess.run(["curl", "-s", "--max-time", "60", "-X", "GET", "-H", "Content-Type: application/json", "-d", json.dumps({"desigs": [d], "output_format": ["ADES_DF"]}), "https://data.minorplanetcenter.net/api/get-obs"], capture_output=True, text=True).stdout
    try:
        rows = json.loads(out)[0]["ADES_DF"]
    except Exception:
        return []
    return [(Time(r["obstime"].rstrip("Z"), format="isot", scale="utc").mjd, float(r["ra"]), float(r["dec"])) for r in rows if r.get("Obstype", "optical") == "optical" and not r.get("deprecated")]


def fit(q0, ep, P):
    mj = [p[0] for p in P]; ra = np.array([p[1] for p in P]); de = np.array([p[2] for p in P]); ob = [p[3] for p in P]
    def res(q):
        if not (q[0] > 20 and 0 <= q[1] < .98):
            return np.full(2 * len(P), 1e4)
        s = M.state(q[0], q[1], q[2], q[3], q[4], q[5] % 360)
        p = G.positions(dict(state=s, epoch=ep), mj, ob) if s is not None else None
        if p is None:
            return np.full(2 * len(P), 1e4)
        dr = ((np.array(p[0]) - ra + 180) % 360 - 180) * np.cos(np.radians(de)) * 3600
        return np.r_[dr, (np.array(p[1]) - de) * 3600]
    s = least_squares(res, q0, x_scale=[1, .05, .5, .5, 5, 5], diff_step=1e-4, max_nfev=80)
    return s.x, res(s.x)


def main():
    data, out = sys.argv[1:3]
    V = json.load(open(H + "/candidates/vet.json")); R2 = json.load(open(H + "/candidates/confirm2.json")); Rf = json.load(open(H + "/candidates/refit.json"))
    U = sorted([x for x in V if not x["known"] and x["smax"] is not None and x["smax"] < 3 and x["drop1"] >= 5 and x["p"] >= .9], key=lambda x: -x["p"])
    tn = []
    with gzip.open(os.path.join(data, "MPCORB.DAT.gz"), "rt", errors="replace") as fh:
        for l in fh:
            if len(l) < 160 or not l[0:1].strip():
                continue
            try:
                a = float(l[92:103])
                if a >= 25:
                    tn.append(dict(H=l[8:13].strip(), ep=M.unpack_epoch(l[20:25].strip()), M=float(l[26:35]), w=float(l[37:46]), O=float(l[48:57]), i=float(l[59:68]), e=float(l[70:79]), a=a, U=l[105:106].strip(), nobs=l[117:122].strip(), arc=l[127:136].strip(), name=l[166:194].strip()))
            except Exception:
                continue
    print(len(tn), "bodies", flush=True)
    cache = {}; res = []
    Ex = S.exposures()
    for x in U:
        f = next((q for q in Rf if q["f"] == x["f"] and q["p"] == x["p"] and not q["known"]), None) or next(q for q in R2 if q["f"] == x["f"] and q["p"] == x["p"] and not q["known"])
        obs = f["obs"]; rs = f["rs"]; rmid = np.median(rs); mid = obs[len(obs) // 2]; yr = Time(mid[0], format="mjd").jyear
        o = C.observer(mid[0]); ca, sa, cd, sd = math.cos(math.radians(mid[1])), math.sin(math.radians(mid[1])), math.cos(math.radians(mid[2])), math.sin(math.radians(mid[2]))
        uv = np.array([ca * cd, sa * cd, sd]); b_ = o @ uv; rho = -b_ + math.sqrt(b_ * b_ - o @ o + rmid ** 2); X = o + rho * uv; Xh = X / np.linalg.norm(X)
        tr = x["track"]; hrs = (tr[-1][0] - tr[0][0]) * 24; vc = ((tr[-1][1] - tr[0][1]) * cd * 3600 / hrs, (tr[-1][2] - tr[0][2]) * 3600 / hrs)
        K = []
        for t in tn:
            m = re.match(r"(\d{4})-(\d{4})", t["arc"])
            if m and int(m.group(1)) <= yr <= int(m.group(2)) + 1:      # its orbit is interpolated at this date: its place is known, and it is elsewhere
                continue
            s = M.state(t["a"], t["e"], t["i"], t["O"], t["w"], t["M"])
            if s is None:
                continue
            n = np.cross(s[:3], s[3:]); n /= np.linalg.norm(n); beta = abs(math.degrees(math.asin(float(Xh @ n))))
            short = t["U"] in ("E", "9", "8", "") or "days" in t["arc"]
            if beta > (4 if short else 2):
                continue
            # the MPC's U is the runoff along the orbit, in arcsec per decade: a body cannot be further from its predicted place than a few times that
            run = RUNOFF.get(t["U"], 1e7) * max(abs(mid[0] - t["ep"]) / 3652.5, .1) * 3
            pq = G.positions(dict(state=s, epoch=t["ep"]), [mid[0], mid[0] + 1], [o, C.observer(mid[0] + 1)])
            if pq is None:
                continue
            sep = math.hypot((float(pq[0][0]) - mid[1]) * cd, float(pq[1][0]) - mid[2]) * 3600
            vb = ((float(pq[0][1]) - float(pq[0][0])) * cd * 150, (float(pq[1][1]) - float(pq[1][0])) * 150)
            if sep > run or math.hypot(vb[0] - vc[0], vb[1] - vc[1]) > .7:
                continue
            P_ = s[:3] / np.linalg.norm(s[:3]); Q_ = np.cross(n, P_); th = math.atan2(float(Xh @ Q_), float(Xh @ P_))
            # distance the orbit allows anywhere near that direction: within [q, Q], which is generous for poor orbits
            q_, Q__ = t["a"] * (1 - t["e"]), t["a"] * (1 + t["e"])
            if rs[-1] * 1.15 < q_ or rs[0] * .85 > Q__:
                continue
            K.append((t, beta))
        print(f"== {x['f']} p={x['p']}: {len(K)} bodies with a poor enough orbit and the candidate near their plane", flush=True)
        rows = []
        for t, beta in K:
            if t["name"] not in cache:
                cache[t["name"]] = get_obs(t["name"]); time.sleep(.3)
            ob = cache[t["name"]]
            if len(ob) < 3:
                continue
            ob = ob if len(ob) <= 150 else ob[:: len(ob) // 150 + 1]
            P = [(m, r, d, obs_b(m)) for m, r, d in ob] + [(o_[0], o_[1], o_[2], obs_b(o_[0])) for o_ in obs]
            nc = len(obs); best = None
            for da in (1.0, .97, 1.03):
                try:
                    q, r_ = fit([t["a"] * da, t["e"], t["i"], t["O"], t["w"], t["M"]], t["ep"], P)
                except Exception:
                    continue
                n_ = len(P); ra_ = r_[:n_]; de_ = r_[n_:]; all_ = float(np.sqrt(np.mean(r_ ** 2))); cand = float(np.sqrt(np.mean(np.r_[ra_[-nc:], de_[-nc:]] ** 2))); old = float(np.sqrt(np.mean(np.r_[ra_[:-nc], de_[:-nc]] ** 2)))
                if best is None or all_ < best["all"]:
                    best = dict(name=t["name"], U=t["U"], H=t["H"], nobs=len(ob), arc=t["arc"], beta=round(beta, 2), all=round(all_, 2), cand=round(cand, 2), old=round(old, 2), a=round(float(q[0]), 1), e=round(float(q[1]), 3), i=round(float(q[2]), 1), a0=t["a"], e0=t["e"], i0=t["i"], q=[float(v) for v in q], ep=t["ep"])
            if best:
                rows.append(best)
        rows.sort(key=lambda r: r["all"])
        for r in rows[:10]:
            print("   ", {k: r[k] for k in ("name", "U", "H", "nobs", "arc", "beta", "all", "cand", "old", "a", "e", "i", "a0", "e0", "i0")}, flush=True)
        for r in [r for r in rows if r["all"] <= 1.5][:5]:
            s = M.state(*r["q"][:5], r["q"][5] % 360); by = {}
            used = {e["image"].split("_ooi_")[0] for e in json.load(open(f"{H}/private/scenarios/{x['sid']}.json"))["all_exposures"]}
            for t in Ex:
                if t["base"] in used:
                    continue
                p = G.positions(dict(state=s, epoch=r["ep"]), [t["m"]], [obs_b(t["m"])])
                if p is None:
                    continue
                ra, de = float(p[0][0]), float(p[1][0])
                if math.hypot((t["ra"] - ra) * math.cos(math.radians(de)), t["dec"] - de) < 1.3:
                    by.setdefault(t["cal"], []).append((t, ra, de))
            print(f"    look for {r['name']} (joint rms {r['all']}) in {len(by)} nights:", flush=True)
            for cal, L in sorted(by.items()):
                items = []; ex_ = []
                for t, ra, de in L:
                    e = G.Exposure(*t["paths"]); ex_.append(e)
                    if e.locate(ra, de) is not None:
                        items.append((e, ra, de))
                if items:
                    v = C2.small(items, 6.0, 2.0); print(f"       {cal} {'/'.join(sorted(set(t['band'] for t, _, _ in L)))} {len(items)} exp: S/N {v} within 2\"", flush=True)
                for e in ex_:
                    e.close()
        res.append(dict(f=x["f"], p=x["p"], rows=rows)); json.dump(res, open(os.path.join(out, "suspects.json"), "w"), indent=0)
    print("DONE")


if __name__ == "__main__":
    main()
