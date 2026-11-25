"""Extend each remaining candidate night by night into images it never used. Known bodies run the same way as controls, and
fake tracks (the candidate's own orbit moved by a fixed sky offset, so nothing real follows it) measure how often the same chain
extends by chance. A night counts only if its peak is near the prediction and unlikely by chance, is not far brighter than the body, is seen in more than one exposure, is not at a static source, and fits the orbit. Same measures as confirm2.py.
Usage, from candidates/: python -I confirm_rest.py list TARGETS.json
                         python -I confirm_rest.py run TARGETS.json INDEX OUTDIR"""
import glob, json, math, os, sys
import numpy as np
H = os.path.expanduser("$ALE_HOME/planet-search/rsi")
sys.path.insert(0, H + "/candidates")
import confirm as C, confirm2 as C2
from scipy.signal import fftconvolve
from astropy.io import fits
G = C.G; ARC = C.ARC
FIELDS = ("F1", "F3", "F4", "F6", "F9"); NULLS = [(180, 60), (-240, 120), (120, -200), (-90, -260), (300, -90)]
RS = (26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 75, 80, 90); COLOR = dict(VR=-.1, r=-.2, g=.5, i=-.4, z=-.5, Y=-.5); MAXSPREAD = 120.; MAXPEAK = 3; TOOBRIGHT = 1.5; MINSNR = 5.5; MAXCHANCE = .15  # fixed from the known bodies in confirm2.log and one control test, before any candidate ran
ZHT = ("F3", 44.1)  # TRKSUB01, already confirmed


def listing(out):
    cfg = json.load(open(H + "/runs/r1/config.json")); sid = {f"F{i+1}": s for i, s in enumerate(cfg["eval"])}; U = []
    for f in sorted(glob.glob(H + "/candidates/vet_gen0*.json")):
        g = os.path.basename(f)[4:10]
        for x in json.load(open(f)):
            if x["f"] not in FIELDS:
                continue
            passed = x["smax"] is not None and x["smax"] < 3 and x["drop1"] >= 5
            if not x["known"] and not passed:
                continue
            m0, ra, de = x["track"][0][:3]; cd = math.cos(math.radians(de))
            same = [u for u in U if u["f"] == x["f"] and abs(u["m0"] - m0) < 1e-6 and math.hypot((u["ra"] - ra) * cd, u["de"] - de) * 3600 < 5]
            if same:
                same[0]["known"] |= x["known"]
                if x["tot"] > same[0]["x"]["tot"]:
                    same[0]["x"], same[0]["g"] = x, g
                continue
            U.append(dict(f=x["f"], m0=m0, ra=ra, de=de, x=x, g=g, known=x["known"]))
    T = []
    for u in sorted(U, key=lambda u: -u["x"]["tot"]):
        x = u["x"]
        if not u["known"] and u["f"] == ZHT[0] and abs(x["r"] - ZHT[1]) < .5:
            continue
        L = json.load(open(f"{H}/runs/r1/gens/{u['g']}/ls6_runs/{sid[u['f']]}/candidates.json"))["candidates"]; best = None; m_, ra_, de_ = x["track"][0][:3]; ob = C.observer(m_)
        for c in L:
            if round(c["p"], 3) != x["p"]:
                continue
            q = G.positions(dict(state=np.array(c["state"], float), epoch=float(c["epoch_mjd"])), [m_], [ob])
            if q is None:
                continue
            sep = math.hypot((float(q[0][0]) - ra_) * math.cos(math.radians(de_)), float(q[1][0]) - de_) * 3600
            if best is None or sep < best[0]:
                best = (sep, c)
        assert best and best[0] < 5, (u["f"], x["p"])
        name = f"{u['f']}_r{x['r']:.0f}_p{x['p']}"; base = dict(f=u["f"], sid=sid[u["f"]], track=x["track"], state=best[1]["state"], epoch=float(best[1]["epoch_mjd"]), name=name)
        T.append(dict(base, kind="control" if u["known"] else "candidate", off=[0, 0], id=name))
        if not u["known"]:
            T += [dict(base, kind="null", off=list(o), id=f"{name}_null{i}") for i, o in enumerate(NULLS)]
    json.dump(T, open(out, "w")); print(len(T), "runs:", sum(t["kind"] == "candidate" for t in T), "candidates,", sum(t["kind"] == "control" for t in T), "controls,", sum(t["kind"] == "null" for t in T), "fake tracks")
    for t in T:
        if t["kind"] != "null":
            print(" ", t["id"], t["kind"])


def targets(f):
    rows = []; seen = set()
    for name in ("files_confirm4.json", "files_confirm.json"):
        p = H + "/candidates/" + name
        if not os.path.exists(p):
            continue
        for b, e in json.load(open(p))["exposures"].items():
            if e["f"] != f or b in seen:
                continue
            ps = [os.path.join(H, "data/confirm", next(x["name"] for x in e["files"] if f"_oo{c}_" in x["name"])) for c in "iwd"]
            if all(os.path.exists(q) for q in ps):
                seen.add(b); rows.append(dict(base=b, cal=e["cal"], band=e["band"], t=e["t"], paths=ps))
    return rows


def family(obs, S0, ep):
    fam = {}
    for r in RS:
        try:
            fam[r] = C.fit_r(obs, S0, ep, r)
        except Exception:
            pass
    if not fam:
        return {}, 99.
    best = min(v[1] for v in fam.values()); kept = {r for r, v in fam.items() if v[1] <= max(.5, 2 * best)}
    if len(kept) == 1:  # one distance alone would mean no spread at all and a falsely tiny search radius: keep its neighbours too
        i = RS.index(next(iter(kept))); kept |= {RS[j] for j in (i - 1, i + 1) if 0 <= j < len(RS)}
    return {r: v for r, v in fam.items() if r in kept}, best


def run(t, out):
    S0 = np.array(t["state"], float); ep = t["epoch"]; P = json.load(open(f"{H}/private/scenarios/{t['sid']}.json")); E = {e["image"]: e for e in P["all_exposures"]}; d = os.path.join(H, "data", str(P["window"]))
    paths = lambda img: tuple(os.path.join(d, E[img][k]) for k in ("image", "weight", "mask"))
    nights = {}
    for m, ra, de, img in t["track"]:
        nights.setdefault(int(np.floor(m - .5)), []).append((m, ra, de, img))
    obs = []; dra, dde = t["off"]; fmags = []
    for n, tr in sorted(nights.items()):
        ex = [(G.Exposure(*paths(q[3])), q[1], q[2]) for q in tr]
        ax, Nm, Dm, used = C.comoving(ex, 6.0)
        if used:
            s, dx, dy = C.peak(ax, C.snr_map(Nm, Dm), 2.0); m = np.mean([q[0] for q in tr]); ra = np.mean([q[1] for q in tr]); de = np.mean([q[2] for q in tr])
            if s >= 4:
                obs.append((m, ra + (dx + dra) * ARC / math.cos(math.radians(de)), de + (dy + dde) * ARC, C.observer(m), s))
                for (e_, a_, d_), q in zip(ex, tr):  # the candidate's own brightness, always at the real track
                    l = e_.locate(a_ + dx * ARC / math.cos(math.radians(d_)), d_ + dy * ARC); mz = E[q[3]].get("magzero")
                    if l and mz:
                        fl, er, fg = G.phot(*e_.data(l[0]), l[1], l[2], l[3])
                        if fl is not None and er and fl > 2 * er:
                            fmags.append(-2.5 * math.log10(fl) + float(mz))
        for e_, _, _ in ex:
            e_.close()
    vfield = float(np.median(fmags)) if fmags else None; ex_cache = {}
    def cached(ps):
        return ex_cache.get(ps[0]) or ex_cache.setdefault(ps[0], G.Exposure(*ps))
    res = dict(id=t["id"], kind=t["kind"], f=t["f"], name=t["name"], field_nights=len(obs), vfield=vfield, new=[], missed=[], rms=None, arc_days=0, r_range=None)
    if len(obs) < 2:
        res["why"] = "fewer than 2 field nights"; json.dump(res, open(out, "w")); return
    if dra or dde:  # start the fit from the moved orbit
        o0 = C.observer(ep); g = S0[:3] - o0; rho = np.linalg.norm(g); uv = g / rho; e1 = np.cross(uv, [0, 0, 1.]); e1 /= np.linalg.norm(e1); e2 = np.cross(uv, e1)
        S0 = np.r_[S0[:3] - (dra * e1 + dde * e2) * rho * ARC * math.pi / 180, S0[3:]]
    fam, rms = family(obs, S0, ep)
    used_here = {i.split("_ooi_")[0] for i in E}; Tg = [x for x in targets(t["f"]) if x["base"] not in used_here]; by = {}  # never count an image the program saw
    for x in Tg:
        h0 = fits.getheader(x["paths"][0], 0); x["m"] = float(h0["MJD-OBS"]) + float(h0["EXPTIME"]) / 2 / 86400; x["mz"] = h0.get("MAGZERO"); by.setdefault(x["cal"], []).append(x)
    done = {}; own = [q[3] for q in t["track"]]
    for npass in range(3):
        gained = False; mid = np.mean([o[0] for o in obs])
        for cal, L in sorted(by.items(), key=lambda kv: abs(kv[1][0]["m"] - mid)):
            if cal in done and (done[cal] is True or done[cal] < 0):
                continue
            if not fam:
                break
            items = []
            for x in L:
                ob = C.observer(x["m"]); ps = [G.positions(dict(state=v[0], epoch=ep), [x["m"]], [ob]) for v in fam.values()]
                ps = [(float(p[0][0]), float(p[1][0])) for p in ps if p is not None]
                if not ps:
                    continue
                a0 = float(np.median([p[0] for p in ps])); d0 = float(np.median([p[1] for p in ps])); sp = max(math.hypot((p[0] - a0) * math.cos(math.radians(d0)), p[1] - d0) for p in ps) * 3600
                items.append((x, a0, d0, sp))
            if not items:
                continue
            spread = max(i[3] for i in items); R = max(1.5, spread + 1.0)
            if spread > MAXSPREAD:
                res["missed"].append(dict(cal=cal, why="too uncertain", spread=round(spread))); continue
            if cal in done and R > .7 * done[cal]:
                continue  # tried with about this radius already
            half = max(30., 2 * R + 10); exo = []; exx = []
            for x, a0, d0, sp in items:
                e_ = cached(x["paths"])
                if e_.locate(a0, d0) is not None:
                    exo.append((e_, a0, d0)); exx.append(x)
            if not exo:
                done[cal] = -1; continue
            ax, Nm, Dm, used = C.comoving(exo, half); done[cal] = R
            if not used:
                continue
            z = C.snr_map(Nm, Dm); X, Y = np.meshgrid(ax, ax); fp = C2.disk(R).astype(float); ok = (Dm > 0) & (np.hypot(X, Y) > 2 * R)
            m = np.mean([x["m"] for x in exx]); accepted = False; row = dict(cal=cal, band=L[0]["band"], n=len(exo), R=round(R, 1), spread=round(spread, 1), rejected=[])
            mzs = [float(x["mz"]) for x in exx if x.get("mz")]; c0 = len(ax) // 2  # how bright it should look here, from its own field brightness (rough, +-0.5 mag)
            if vfield is not None and mzs and Dm[c0, c0] > 0:
                row["exp"] = round(float(10 ** (.4 * (np.median(mzs) - vfield - COLOR.get(L[0]["band"], 0.)))) * math.sqrt(float(Dm[c0, c0])), 1)
            for tries in range(MAXPEAK):
                s, dx, dy = C.peak(ax, z, R); chance = float((fftconvolve(((z >= s) & (Dm > 0)).astype(float), fp, mode="same")[ok] > .5).mean()) if ok.any() else None
                row.update(snr=round(s, 1), chance=None if chance is None else round(chance, 4), dx=round(dx, 2), dy=round(dy, 2))
                if s < MINSNR:
                    break
                cd = math.cos(math.radians(exo[0][2])); mags = []
                per = [v for v in (C2.small([(e_, a0 + dx * ARC / cd, d0 + dy * ARC)]) for e_, a0, d0 in exo) if v is not None]
                for (e_, a0, d0), x in zip(exo, exx):
                    l = e_.locate(a0 + dx * ARC / cd, d0 + dy * ARC)
                    if l and x.get("mz"):
                        fl, er, fg = G.phot(*e_.data(l[0]), l[1], l[2], l[3])
                        if fl is not None and fl > 0:
                            mags.append(-2.5 * math.log10(fl) + float(x["mz"]))
                k_ = len(exo) // 2; sky = (exo[k_][1] + dx * ARC / cd, exo[k_][2] + dy * ARC)
                motion = math.hypot((exo[-1][1] - exo[0][1]) * cd, exo[-1][2] - exo[0][2]) * 3600
                stat_ = C2.small([(e_, sky[0], sky[1]) for e_, _, _ in exo]); other = []
                groups = [[t["paths"] for t in v] for k, v in by.items() if k != cal] + [[paths(i) for i in own]]
                for g_ in groups:
                    on = [(cached(p_), sky[0], sky[1]) for p_ in g_ if cached(p_).locate(*sky) is not None]
                    if on:
                        v = C2.small(on)
                        if v is not None:
                            other.append(v)
                mag = round(float(np.median(mags)), 2) if mags else None
                row.update(per=per, mag=mag, motion=round(motion, 1), static=stat_, other=other); why = None; sp_ = sorted(per)
                if chance is not None and chance > MAXCHANCE:
                    why = "chance"
                elif mag is not None and vfield is not None and mag < vfield - TOOBRIGHT:
                    why = "too bright"
                elif len(sp_) > 1 and sp_[-2] < 3:
                    why = "one exposure"
                elif other and max(other) >= 3:
                    why = "static"
                elif motion >= 3 and stat_ is not None and stat_ >= s:
                    why = "static"
                elif not other and not (motion >= 3 and stat_ is not None and stat_ < .7 * s):
                    why = "cannot test static"
                else:
                    mk = exx[k_]["m"]  # the place `sky` is exo[k_]'s, so its time is exx[k_]'s, not the mean (fixed 2026-11-09 23:20 PST: on i+g nights an hour apart the mean put the place ~2" off)
                    ob2 = obs + [(mk, sky[0], sky[1], C.observer(mk), s)]; row["t_place"] = mk; fam2, rms2 = family(ob2, S0, ep); row["rms_after"] = round(rms2, 2)
                    if not fam2 or rms2 > .6:
                        why = "off the orbit"
                if why is None:
                    obs, fam, rms = ob2, fam2, rms2; done[cal] = True; gained = True; accepted = True; res["new"].append(row)
                    S0 = min(fam.values(), key=lambda v: v[1])[0]; break
                row["rejected"].append([round(s, 1), why]); z[np.hypot(X - dx, Y - dy) <= 1.] = -9  # was 3": with R = 1.5" one rejected spike blanked the whole search disc (fixed 2026-11-10 00:25 PST: <lead> lost two nights, where the body sits at S/N 6.0 and 10.3 at the predicted place)
            for e_ in ex_cache.values():
                e_.cache.clear()
                for hd in list(e_.h):
                    hd.__dict__.pop("data", None)
            if not accepted:
                res["missed"].append(row)
        if not gained:
            break
    ms_ = [o[0] for o in obs]; res.update(rms=round(rms, 3), arc_days=round(max(ms_) - min(ms_), 1), r_range=sorted(fam)[:1] + sorted(fam)[-1:] if fam else None, tested=len(done), points=[[float(o[0]), float(o[1]), float(o[2]), float(o[4])] for o in obs], state=[float(v) for v in (min(fam.values(), key=lambda v: v[1])[0] if fam else S0)], epoch=ep)
    json.dump(res, open(out, "w"), indent=0)
    print(t["id"], "new nights", len(res["new"]), [(n["cal"], n["snr"]) for n in res["new"]], "arc", res["arc_days"], "rms", res["rms"], "tested", res["tested"], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "list":
        listing(sys.argv[2])
    else:
        T = json.load(open(sys.argv[2])); t = T[int(sys.argv[3])]; os.makedirs(sys.argv[4], exist_ok=True); out = os.path.join(sys.argv[4], t["id"] + ".json")
        if not (os.path.exists(out) and os.path.getsize(out)):
            run(t, out)
