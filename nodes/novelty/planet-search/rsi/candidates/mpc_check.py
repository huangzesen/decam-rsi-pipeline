"""First check against the MPC's own files (downloaded to data/mpc, untrusted, plain parsing only).
(1) Isolated Tracklet File: any unlinked observation near each candidate's predicted place within 45 days of its arc.
(2) MPCORB: every body with a > 25 au, propagated (two-body) to the middle of the candidate's arc: those within 3 degrees with similar motion,
    the short-arc ones (poor orbits, large error back in time) being the ones that could be the same body.
Usage: python -I mpc_check.py DATADIR OUTDIR
"""
import gzip, json, math, os, sys
import numpy as np
H = os.path.expanduser("$ALE_HOME/planet-search/rsi")
sys.path.insert(0, H + "/candidates")
import confirm as C
G = C.G
from astropy.time import Time
GM = 2.959122082855911e-4; EPS = math.radians(23.439291)


def obs_date(l):
    y, mo, d = int(l[15:19]), int(l[20:22]), float(l[23:32]); return Time(f"{y}-{mo:02d}-01T00:00:00", scale="utc").mjd + d - 1


def radec(l):
    rh, rm, rs = float(l[32:34]), float(l[35:37]), float(l[38:44]); ds, dd, dm, dss = l[44], float(l[45:47]), float(l[48:50]), float(l[51:56])
    return 15 * (rh + rm / 60 + rs / 3600), (-1 if ds == "-" else 1) * (dd + dm / 60 + dss / 3600)


def unpack_epoch(s):
    c = {"I": 18, "J": 19, "K": 20}[s[0]]; y = c * 100 + int(s[1:3]); f = lambda ch: int(ch) if ch.isdigit() else ord(ch) - 55
    return Time(f"{y}-{f(s[3]):02d}-{f(s[4]):02d}T00:00:00", scale="tt").mjd


def state(a, e, inc, node, peri, M):
    """Heliocentric equatorial J2000 position and velocity from MPCORB elements (angles in degrees), au and au/day."""
    M = math.radians(M); E = M
    if e >= 1:
        return None
    for _ in range(30):
        E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    x, y = a * (math.cos(E) - e), a * math.sqrt(1 - e * e) * math.sin(E); r = a * (1 - e * math.cos(E)); n = math.sqrt(GM / a ** 3)
    vx, vy = -a * a * n * math.sin(E) / r, a * a * n * math.sqrt(1 - e * e) * math.cos(E) / r
    i, O, w = map(math.radians, (inc, node, peri))
    P = np.array([math.cos(O) * math.cos(w) - math.sin(O) * math.sin(w) * math.cos(i), math.sin(O) * math.cos(w) + math.cos(O) * math.sin(w) * math.cos(i), math.sin(w) * math.sin(i)])
    Q = np.array([-math.cos(O) * math.sin(w) - math.sin(O) * math.cos(w) * math.cos(i), -math.sin(O) * math.sin(w) + math.cos(O) * math.cos(w) * math.cos(i), math.cos(w) * math.sin(i)])
    Rx = np.array([[1, 0, 0], [0, math.cos(EPS), -math.sin(EPS)], [0, math.sin(EPS), math.cos(EPS)]])
    return np.r_[Rx @ (x * P + y * Q), Rx @ (vx * P + vy * Q)]


def main():
    data, out = sys.argv[1:3]
    V = json.load(open(H + "/candidates/vet.json")); R2 = json.load(open(H + "/candidates/confirm2.json")); Rf = json.load(open(H + "/candidates/refit.json"))
    U = sorted([x for x in V if not x["known"] and x["smax"] is not None and x["smax"] < 3 and x["drop1"] >= 5 and x["p"] >= .9], key=lambda x: -x["p"])
    B = []
    for x in U:
        f = next((q for q in Rf if q["f"] == x["f"] and q["p"] == x["p"] and not q["known"]), None); t = x["track"]; a, b, mid = t[0], t[-1], t[len(t) // 2]
        cd = math.cos(math.radians(mid[2])); hrs = (b[0] - a[0]) * 24
        B.append(dict(x=x, fit=f, m0=a[0], m1=(max(o[0] for o in f["obs"]) if f else b[0]), mid=mid, v=((b[1] - a[1]) * cd * 3600 / hrs, (b[2] - a[2]) * 3600 / hrs), ra=mid[1], dec=mid[2]))
    print("(1) isolated tracklets", flush=True)
    hits = {i: [] for i in range(len(B))}; n = 0
    with gzip.open(os.path.join(data, "itf.txt.gz"), "rt", errors="replace") as fh:
        for l in fh:
            n += 1
            if len(l) < 70 or l[14] in "sv":
                continue
            try:
                y = int(l[15:19])
            except ValueError:
                continue
            for i, b in enumerate(B):
                if abs(y - Time(b["m0"], format="mjd").jyear) > 1:
                    continue
                try:
                    m = obs_date(l); ra, de = radec(l)
                except Exception:
                    continue
                if not (b["m0"] - 45 <= m <= b["m1"] + 45):
                    continue
                if math.hypot((ra - b["ra"]) * math.cos(math.radians(b["dec"])), de - b["dec"]) < 1.5:
                    hits[i].append((m, ra, de, l[:12].strip(), l[65:70].strip(), l[77:80].strip()))
    print(n, "lines read", flush=True)
    for i, b in enumerate(B):
        x = b["x"]; f = b["fit"]; rows = []
        for m, ra, de, tid, mag, code in hits[i]:
            o = C.observer(m)
            if f:
                ps = [G.positions(dict(state=np.array(s), epoch=f["epoch"]), [m], [o]) for s in f["states"].values()]
                ps = [(float(p[0][0]), float(p[1][0])) for p in ps if p is not None]
            else:
                ps = [(x["track"][0][1] + (x["track"][-1][1] - x["track"][0][1]) * (m - x["track"][0][0]) / (x["track"][-1][0] - x["track"][0][0]), x["track"][0][2] + (x["track"][-1][2] - x["track"][0][2]) * (m - x["track"][0][0]) / (x["track"][-1][0] - x["track"][0][0]))]
            sep = min(math.hypot((p[0] - ra) * math.cos(math.radians(de)), p[1] - de) * 3600 for p in ps)
            rows.append((round(sep, 1), round(m, 3), tid, mag, code))
        rows.sort()
        print(f"== {x['f']} p={x['p']}: {len(rows)} unlinked observations within 1.5 deg and 45 d; nearest {rows[:6]}", flush=True)
    print("(2) MPCORB bodies with a > 25 au", flush=True)
    tn = []
    with gzip.open(os.path.join(data, "MPCORB.DAT.gz"), "rt", errors="replace") as fh:
        for l in fh:
            if len(l) < 160 or not l[0:1].strip():
                continue
            try:
                a = float(l[92:103])
                if a < 25:
                    continue
                tn.append(dict(H=l[8:13].strip(), ep=unpack_epoch(l[20:25].strip()), M=float(l[26:35]), w=float(l[37:46]), O=float(l[48:57]), i=float(l[59:68]), e=float(l[70:79]), a=a, U=l[105:106], nobs=l[117:122].strip(), arc=l[127:136].strip(), name=l[166:194].strip(), last=l[194:202].strip()))
            except Exception:
                continue
    print(len(tn), "bodies", flush=True)
    for b in B:
        x = b["x"]; m = b["mid"][0]; o = C.observer(m); o2 = C.observer(m + 1); rows = []
        for t in tn:
            s = state(t["a"], t["e"], t["i"], t["O"], t["w"], t["M"])
            if s is None:
                continue
            p = G.positions(dict(state=s, epoch=t["ep"]), [m, m + 1], [o, o2])
            if p is None:
                continue
            ra, de = float(p[0][0]), float(p[1][0]); cd = math.cos(math.radians(b["dec"])); sep = math.hypot((ra - b["ra"]) * cd, de - b["dec"])
            if sep > 3:
                continue
            v = ((float(p[0][1]) - ra) * cd * 3600 / 24, (float(p[1][1]) - de) * 3600 / 24); dv = math.hypot(v[0] - b["v"][0], v[1] - b["v"][1])
            rows.append(dict(sep=round(sep * 3600), dv=round(dv, 2), name=t["name"], H=t["H"], U=t["U"], nobs=t["nobs"], arc=t["arc"], last=t["last"], a=round(t["a"], 1), e=round(t["e"], 2), i=round(t["i"], 1)))
        rows.sort(key=lambda r: r["sep"])
        close = [r for r in rows if r["dv"] < .5]
        print(f"== {x['f']} p={x['p']}: {len(rows)} bodies within 3 deg, {len(close)} with motion within 0.5\"/h; poorest orbits among those (U >= 5 or short arc):", flush=True)
        poor = [r for r in close if r["U"] in "56789E" or "days" in r["arc"]]
        for r in poor[:12]:
            print("   ", r, flush=True)
        print("   nearest:", close[:3], flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
