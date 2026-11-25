"""Screen the unmatched candidates first seen after gen042 with p above PMIN (new_cands.json), as f1_60.py did for one, beside one known TNO per field:
forced flux in each exposure, per-night stacks, the in-field static test, the stack without its brightest exposure, sky motion, and the Legacy Surveys DR10
catalogue (Data Lab, anonymous; sky positions only) at each detected night's place. Rules, fixed before running: reject if fewer than 3 nights reach a stack
S/N of 4.5, if the in-field static test reaches 3 on any night, if a DR10 source brighter than r 24.5 lies within 1" of any detected night's place, or if
the stack without its brightest exposure falls below 5. Writes screen_new.json. Usage, from harness/: python -I ../candidates/screen_new.py PMIN [PMAX OUT.json F1,F2,...]"""
import json, math, os, sys, time, urllib.parse, urllib.request
from collections import OrderedDict
MAXCCD = int(os.environ.get("VET_MAXCCD", 30))
import numpy as np
H = os.path.expanduser("$ALE_HOME/planet-search/rsi"); sys.path.insert(0, H + "/harness")
import grade as G
PMIN = float(sys.argv[1]); PMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 9.; OUTJ = sys.argv[3] if len(sys.argv) > 3 else H + "/candidates/screen_new.json"
FIELDS = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None; cfg = json.load(open(H + "/runs/r1/config.json")); sid_of = {f"F{i + 1}": s for i, s in enumerate(cfg["eval"])}
O = [o for o in json.load(open(H + "/candidates/new_cands.json")) if o["first"] > 42 and PMIN < o["pmax"] <= PMAX and (FIELDS is None or o["field"] in FIELDS)]
print(len(O), "candidates", flush=True)
TOK = {"User-Agent": "archive-cover/1.0", "X-DL-AuthToken": "anonymous.0.0.anon_access"}


def ls(ra, de, rad=3.):
    sql = (f"SELECT ra, dec, type, mag_r, mag_z, q3c_dist(ra, dec, {ra:.6f}, {de:.6f}) * 3600 AS sep FROM ls_dr10.tractor "
           f"WHERE q3c_radial_query(ra, dec, {ra:.6f}, {de:.6f}, {rad / 3600:.6f}) ORDER BY sep")
    for k in range(3):
        try:
            t = urllib.request.urlopen(urllib.request.Request("https://datalab.noirlab.edu/query/query?" + urllib.parse.urlencode(dict(sql=sql, ofmt="csv", **{"async": "false"})), headers=TOK), timeout=180).read().decode()
            L = t.strip().splitlines(); hd = L[0].split(","); return [dict(zip(hd, l.split(","))) for l in L[1:]]
        except Exception as e:
            time.sleep(5 * (k + 1))
    return None


class Field:
    def __init__(self, sid):
        P = json.load(open(f"{H}/private/scenarios/{sid}.json")); self.E = sorted(P["all_exposures"], key=lambda e: e["mjd_mid"]); self.d = os.path.join(H, "data", str(P["window"]))
        self.mj = [e["mjd_mid"] for e in self.E]; self.ob = [np.array([e["obs_x"], e["obs_y"], e["obs_z"]]) for e in self.E]; self.night = [int(math.floor(m - .5)) for m in self.mj]
        e = self.E[len(self.E) // 2]; self.ref = (e["mjd_mid"], np.array([e["obs_x"], e["obs_y"], e["obs_z"]])); self.X = {}; self.held = OrderedDict()

    def meas(self, k, ra, de):
        if k not in self.X:
            e = self.E[k]; self.X[k] = G.Exposure(os.path.join(self.d, e["image"]), os.path.join(self.d, e["weight"]), os.path.join(self.d, e["mask"]))
        loc = self.X[k].locate(ra, de)
        if loc:
            fl, er, flag = G.phot(*self.X[k].data(loc[0]), loc[1], loc[2], loc[3])
            self.held[(k, loc[0])] = True; self.held.move_to_end((k, loc[0]))
            while len(self.held) > MAXCCD:  # as vet.py: one DECam CCD with weight and mask is about 0.13 GB
                (k0, c0), _ = self.held.popitem(last=False); self.X[k0].cache.pop(c0, None); del self.X[k0].h[c0].data
            if fl is not None and er and flag == "":
                return fl, er
        return None

    def close(self):
        for x in self.X.values():
            x.close()
        self.X = {}; self.held = OrderedDict()


def check(F, c):
    ra, de = G.positions(dict(state=np.array(c["state"], float), epoch=c.get("epoch_mjd")), F.mj, F.ob); per = []; byn = {}
    for k in range(len(F.E)):
        m = F.meas(k, ra[k], de[k]); per.append(None if m is None else round(m[0] / m[1], 1))
        if m:
            byn.setdefault(F.night[k], []).append((k, m))
    ns = {n: round(G.stack(*zip(*[x[1] for x in v])), 1) for n, v in byn.items()}; stat = {}; lsm = {}
    for n, v in byn.items():
        k0 = v[0][0]; ot = [j for j in range(len(F.E)) if F.night[j] != n]; ot = ot[::max(1, len(ot) // 10)]
        ms = [m for m in (F.meas(j, ra[k0], de[k0]) for j in ot) if m]; stat[n] = round(G.stack(*zip(*ms)), 1) if ms else None
        if ns[n] >= 4:
            q = ls(float(ra[k0]), float(de[k0])); lsm[n] = None if q is None else [(round(float(x["sep"]), 2), x["type"], x["mag_r"]) for x in q[:3]]
    allm = [x[1] for v in byn.values() for x in v]
    tot = G.stack(*zip(*allm)) if allm else 0.; s1 = sorted(allm, key=lambda m: m[0] / m[1]); drop = G.stack(*zip(*s1[:-1])) if len(s1) > 1 else 0.
    dt = (F.mj[-1] - F.mj[0]) * 24; dra = ((ra[-1] - ra[0] + 180) % 360 - 180) * math.cos(math.radians(de[0])) * 3600 / dt; dde = (de[-1] - de[0]) * 3600 / dt
    good = sum(v >= 4.5 for v in ns.values()); smax = max([v for v in stat.values() if v is not None], default=None)
    lsbad = [(n, x) for n, L in lsm.items() if L for x in L if x[0] < 1.0 and x[2] not in ("", "nan") and float(x[2]) < 24.5]
    why = []
    if good < 3: why.append(f"{good} nights >= 4.5")
    if smax is not None and smax >= 3: why.append(f"static {smax}")
    if lsbad: why.append(f"DR10 source {lsbad[0][1]} on night {lsbad[0][0]}")
    if drop < 5: why.append(f"without brightest {drop:.1f}")
    return dict(p=round(c["p"], 3), mag=c.get("mag"), r=round(float(np.linalg.norm(c["state"][:3])), 1), nights=ns, static=stat, ls=lsm, tot=round(tot, 1), drop=round(drop, 1),
                rate=[round(dra, 2), round(dde, 2)], per=per, ra=[float(x) for x in ra], dec=[float(x) for x in de], verdict="PASS" if not why else "reject: " + "; ".join(why))


out = []; ctl_done = set()
for o in sorted(O, key=lambda o: (o["field"], -o["pmax"])):
    sid = sid_of[o["field"]]; F = Field(sid); g = max(o["seen"], key=lambda x: x[1])[0]
    C = json.load(open(f"{H}/runs/r1/gens/{g}/ls6_runs/{sid}/candidates.json"))["candidates"]; m, obv = F.ref; best = None
    for c in C:
        q = G.positions(dict(state=np.array(c["state"], float), epoch=c.get("epoch_mjd")), [m], [obv])
        if q is not None and math.hypot((float(q[0][0]) - o["ra"]) * math.cos(math.radians(o["dec"])), float(q[1][0]) - o["dec"]) * 3600 < 5 and (best is None or c["p"] > best["p"]):
            best = c
    r = check(F, best); r.update(field=o["field"], gen=g, rounds=o["rounds"], first=o["first"], last=o["last"], kind="candidate"); out.append(r)
    print(f"{o['field']} r {r['r']:5.1f} p {r['p']:.3f} ({o['rounds']} rounds) nights {r['nights']} static {r['static']} all {r['tot']} w/o brightest {r['drop']} rate {r['rate']}\n   DR10 {r['ls']}\n   -> {r['verdict']}", flush=True)
    if o["field"] not in ctl_done:
        hits = [h for h in json.load(open(f"{H}/runs/r1/gens/{g}/grade/grade.json"))["hits"] if h["sid"] == sid]
        if hits:
            h = max(hits, key=lambda h: h["p"]); c = next((c for c in C if round(c["p"], 9) == round(h["p"], 9)), None)
            if c:
                r = check(F, c); r.update(field=o["field"], gen=g, kind="control", name=h["pdes"]); out.append(r)
                print(f"   CONTROL {h['pdes']} p {r['p']:.3f} nights {r['nights']} static {r['static']} all {r['tot']} DR10 {r['ls']} -> {r['verdict']}", flush=True)
        ctl_done.add(o["field"])
    F.close(); json.dump(out, open(OUTJ, "w"), indent=0)
c = [x for x in out if x["kind"] == "candidate"]; print("PASS:", [(x["field"], x["r"], x["p"]) for x in c if x["verdict"] == "PASS"], "| controls passing:", sum(x["verdict"] == "PASS" for x in out if x["kind"] == "control"), "of", sum(x["kind"] == "control" for x in out))
