"""Automatic confirmation of a planet-search lead in images its program never used (2026-11-09). The steps done by hand for r2 gen002's F1 lead (cand10_*), made general. Each stage writes into OUTDIR and is skipped when its
output exists, so a rerun resumes. Stages run in their own processes (they set module globals).

SPEC: as novelty_check.py's program form {"id", "run", "gen", "sid", "p", "r"}; the field tag (F1..F10) is found from runs/r1/config.json.
  pos     night places of the lead and of a control (the field's highest-p hit in that round's grade.json) by cand5_f3pos's comoving stacks on the
          field's visible exposures (held-back ones left out); the control's places against JPL Horizons at W84.        -> pos.json
  range   systematic ranging as cand10_range.py: every bound orbit through the lead's nights (peak >= 3), 400 draws, places every day of the field
          date +-150 d.                                                                                                     -> range.json, draws.npz
  list    public DECam InstCal exposures (NOIRLab; the archive gets a wide box only) within 60 d of the field's nights, >= 30 s, g r i z Y VR,
          P = share of draws within 1.15 deg of the pointing >= 0.3 and the draws' 99% spread there <= 120" (confirm_rest's MAXSPREAD);
          the field's visible exposures left out; nearest in time first, at most 30 exposures and 15 GB.                    -> files.json
  fetch   download to data/confirm (Crucial), md5-checked (bin/fetch_scenarios.py).                                         -> fetch.log
  check   confirm_rest.py's run() and rules unchanged (MINSNR 5.5, MAXCHANCE 0.15, MAXSPREAD 120", one-exposure, static, orbit tests), on the
          fetched exposures plus the field's held-back exposures (never shown to the program; confirm_rest's "never count an image the program saw"
          is kept by excluding only the visible ones). Runs: the lead, the control, 5 fake tracks (confirm_rest's NULLS).   -> check/*.json
  cfht    CFHT MegaCam: images from archive_cover/box_<F>.json, else a CADC ObsCore query (box +-1 deg around the path, field date +-60 d, 20 min
          limit); used where the family of the lead's nights (incl. any gained in check) has spread < 1" and the pointing is within 0.8 deg;
          known bodies from SkyBoT (568) in the same images as controls; run on NERSC (cfht_fetch.py with this sky's Gaia box, cfht_phot.py
          unchanged; debug CPU job) if the ControlMaster socket is alive, else skipped.                                       -> cfht/
  verdict one line.
Rules, fixed 2026-11-09 before any lead after F1 ran:
  DECam: a night counts as confirm_rest decides; the DECam test counts only if no fake track gains a night.
  CFHT FOUND: stack S/N at the prediction >= 5 and >= null median + 5 null sigma (robust), >= 2 nights with S/N >= 3 and peak within 0.6", and on
    every night the same sky on other nights < 3 sigma or < 25% of the lead's flux there (F1 test: a static source at 14% of the lead, S/N 5); counts only if >= 2/3 of the controls with V <= 24 are recovered at S/N >= 5 (or, with no
    V <= 24 control, any control at S/N >= 5). The F1 run's "no null as high" clause is dropped: with few images stars at fixed offsets beat every
    control too (F1 run: controls 1.6-13% of nulls as high; TRKSUB01's own CFHT run 5%).
  CFHT NOT SEEN: stack S/N < 2 with >= 6 images.  Otherwise UNDECIDED.
  Overall: SEEN IN OTHER IMAGES (DECam counts with >= 1 gained night, or CFHT FOUND) | NOT SEEN (CFHT NOT SEEN and no DECam night) | NO DECISIVE DATA.
Usage, from candidates/: python -I lead_confirm.py SPEC.json OUTDIR"""
import collections, datetime, glob, json, math, os, re, shutil, subprocess, sys, time, urllib.parse, urllib.request
import numpy as np
H = os.path.expanduser("$ALE_HOME/planet-search/rsi"); C = H + "/candidates"; PY = sys.executable
NERSC = "$NERSC_USER@perlmutter.nersc.gov"; SOCK = os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock"); NC = "$NERSC_SCRATCH/planet-cand"
sys.path.insert(0, C); sys.path.insert(0, H + "/harness")


def spec_field(S):
    cfg = json.load(open(H + "/runs/r1/config.json")); f = {s: f"F{i + 1}" for i, s in enumerate(cfg["eval"])}[S["sid"]]
    P = json.load(open(f"{H}/private/scenarios/{S['sid']}.json")); hb = set(P.get("held_back") or [])
    E = sorted([e for e in P["all_exposures"] if e["image"] not in hb], key=lambda e: e["mjd_mid"]); HB = sorted([e for e in P["all_exposures"] if e["image"] in hb], key=lambda e: e["mjd_mid"])
    return f, P, E, HB, os.path.join(H, "data", str(P["window"]))


def cands(S):
    return json.load(open(f"{H}/runs/{S['run']}/gens/{S['gen']}/ls6_runs/{S['sid']}/candidates.json"))["candidates"]


def pick(S):
    c = [c for c in cands(S) if abs(c["p"] - S["p"]) < 2e-5 and abs(float(np.linalg.norm(c["state"][:3])) - S["r"]) < .06]; assert len(c) == 1, len(c); return c[0]


def control(S):
    g = json.load(open(f"{H}/runs/{S['run']}/gens/{S['gen']}/grade/grade.json"))["hits"]; h = sorted([x for x in g if x["sid"] == S["sid"]], key=lambda x: (x["tier"] != 1, -x["p"]))
    for x in h:
        c = [c for c in cands(S) if round(c["p"], 9) == round(x["p"], 9)]
        if len(c) == 1:
            return c[0], x["pdes"]
    return None, None


# ---------------------------------------------------------------- pos
def st_pos(S, D):
    import confirm as Cf
    G = Cf.G; F, P, E, HB, d0 = spec_field(S); mj = [e["mjd_mid"] for e in E]; ob = [np.array([e["obs_x"], e["obs_y"], e["obs_z"]]) for e in E]; night = [int(math.floor(m - .5)) for m in mj]
    src = open(C + "/cand5_f3pos.py").read(); ns = {"__name__": "x", "C": Cf, "np": np, "math": math, "PS": .263, "ARC": 1 / 3600.}
    exec("from scipy.ndimage import convolve\n" + "def comoving" + src.split("def comoving")[1].split("\nS = json.load")[0], ns); comoving = ns["comoving"]
    lead = pick(S); ctl, pdes = control(S); X = {k: G.Exposure(os.path.join(d0, e["image"]), os.path.join(d0, e["weight"]), os.path.join(d0, e["mask"])) for k, e in enumerate(E)}; OUT = dict(field=F, control=pdes)
    for name, c in (("lead", lead), ("control", ctl)):
        if c is None:
            continue
        ra, de = G.positions(dict(state=np.array(c["state"], float), epoch=c.get("epoch_mjd")), mj, ob); out = []
        for nt in sorted(set(night)):
            ks = [k for k in range(len(E)) if night[k] == nt]; ax, N, Dn = comoving([(X[k], float(ra[k]), float(de[k])) for k in ks], 2.5); z_ = Cf.snr_map(N, Dn)
            pk, ox, oy = Cf.peak(ax, z_, 1.2); m = float(np.mean([mj[k] for k in ks])); r0, dd = float(np.mean(ra[ks])), float(np.mean(de[ks]))
            out.append(dict(night=nt, t=m, peak=float(pk), off=[float(ox), float(oy)], ra=r0 + ox / 3600 / math.cos(math.radians(dd)), dec=dd + oy / 3600, n=len(ks), bands=sorted({E[k]["band"] for k in ks})))
            for k in ks:
                X[k].cache.clear()
        if name == "control":
            try:
                cmd = f"'{pdes};'" if str(pdes).isdigit() else f"'DES={pdes};'"
                q = dict(format="json", COMMAND=cmd, OBJ_DATA="NO", MAKE_EPHEM="YES", EPHEM_TYPE="OBSERVER", CENTER="'W84@399'", QUANTITIES="'1'", ANG_FORMAT="DEG", EXTRA_PREC="YES", CSV_FORMAT="YES",
                         TLIST_TYPE="MJD", TIME_TYPE="UT", TLIST=" ".join(f"'{o['t']:.6f}'" for o in out))
                t = json.loads(urllib.request.urlopen(urllib.request.Request("https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q), headers={"User-Agent": "orbit-check/1.0"}), timeout=120).read())["result"]
                for o, l in zip(out, t.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()):
                    v = l.split(","); o["minus_horizons"] = [((o["ra"] - float(v[3]) + 180) % 360 - 180) * math.cos(math.radians(o["dec"])) * 3600, (o["dec"] - float(v[4])) * 3600]
            except Exception as ex:
                OUT["horizons_error"] = repr(ex)[:120]
        OUT[name] = out; print(name, [(o["night"], round(o["peak"], 1), [round(v, 2) for v in o["off"]], [round(v, 2) for v in o.get("minus_horizons", [])]) for o in out], flush=True)
    json.dump(OUT, open(D + "/pos.json", "w"), indent=0)


# ---------------------------------------------------------------- range
def st_range(S, D):
    import suspects as Su
    sys.path.insert(0, os.path.dirname(H) + "/linker"); import orbits as O
    O.GM = 2.959122082855911e-4 * 1.001341; GM = O.GM; CAU = O.C_AU; AS = math.pi / 180 / 3600
    F_ = [o for o in json.load(open(D + "/pos.json"))["lead"] if o["peak"] >= 3]
    if len(F_) < 2:
        json.dump(dict(error="fewer than 2 nights with peak >= 3"), open(D + "/range.json", "w")); return
    def attempt(F_, extra, write=True):
        T = np.array([o["t"] for o in F_]); RA = np.array([o["ra"] for o in F_]); DE = np.array([o["dec"] for o in F_]); SG = np.array([max(.1, .65 / o["peak"]) for o in F_]); EP = float(T.mean())
        OB = np.array([Su.obs_b(float(t)) for t in T]); O0 = Su.obs_b(EP)

        def unit(a, d):
            return np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], -1)

        def radec(X, V, t, obs):
            a, e, i, Om, om, M0 = O.rv_to_elements(X, V); n = np.sqrt(GM / a ** 3); tau = np.zeros((len(X), len(t)))
            for _ in range(3):
                r, _v = O.elements_to_rv(a[:, None], e[:, None], i[:, None], Om[:, None], om[:, None], M0[:, None] + n[:, None] * (t[None] - tau - EP)); g = r - obs[None]; dist = np.linalg.norm(g, axis=-1); tau = dist / CAU
            return np.arctan2(g[..., 1], g[..., 0]) % (2 * np.pi), np.arcsin(g[..., 2] / dist)
        k = np.polyfit(T - EP, np.unwrap(np.radians(RA)), 1); kd = np.polyfit(T - EP, np.radians(DE), 1); a0, d0 = k[1], kd[1]
        u0 = unit(a0, d0); e1 = np.array([-math.sin(a0), math.cos(a0), 0.]); e2 = np.cross(u0, e1)

        def states(rho, vl, q):
            u_ = u0[None] + (q[:, 0:1] * e1[None] + q[:, 1:2] * e2[None]) * AS; u_ /= np.linalg.norm(u_, axis=1, keepdims=True)
            return O0[None] + rho[:, None] * u_, vl[:, None] * u_ + 1e-3 * (q[:, 2:3] * e1[None] + q[:, 3:4] * e2[None])

        def resid(rho, vl, q):
            X, V = states(rho, vl, q); ok = np.einsum("ij,ij->i", V, V) < 2 * GM / np.linalg.norm(X, axis=1) * .9999; r_ = np.full((len(rho), 2 * len(T)), 1e4)
            if ok.any():
                ra, de = radec(X[ok], V[ok], T, OB); r_[ok] = np.c_[((ra - np.radians(RA)[None] + math.pi) % (2 * math.pi) - math.pi) * np.cos(np.radians(DE))[None] / AS / SG[None], (de - np.radians(DE)[None]) / AS / SG[None]]
            return r_
        RH, VL = [g.ravel() for g in np.meshgrid(np.arange(10., 300.01, .5), np.arange(-.01, .01001, 1e-4), indexing="ij")]
        rate = np.r_[k[0] * math.cos(d0), kd[0]]; Vobs = Su.obs_b(EP + .5) - Su.obs_b(EP - .5); Q = np.c_[np.zeros((len(RH), 2)), np.c_[RH * rate[0] + Vobs @ e1, RH * rate[1] + Vobs @ e2] * 1e3]; h = np.array([.01, .01, 1e-4, 1e-4])
        keep = np.einsum("ij,ij->i", *[states(RH, VL, Q)[1]] * 2) < 2 * GM / np.linalg.norm(states(RH, VL, Q)[0], axis=1) * 1.5; RH, VL, Q = RH[keep], VL[keep], Q[keep]
        for it in range(8):
            r0 = resid(RH, VL, Q); J = np.stack([(resid(RH, VL, Q + h[j] * np.eye(4)[j]) - r0) / h[j] for j in range(4)], -1); good = np.all(np.abs(r0) < 1e3, axis=1)
            A_ = np.einsum("nij,nik->njk", J, J); b = np.einsum("nij,ni->nj", J, r0); dq = np.zeros_like(Q)
            if good.any():
                dq[good] = -np.einsum("njk,nk->nj", np.linalg.pinv(A_[good]), b[good])
            Q = Q + dq
        r0 = resid(RH, VL, Q); chi = (r0 ** 2).sum(1); J = np.stack([(resid(RH, VL, Q + h[j] * np.eye(4)[j]) - r0) / h[j] for j in range(4)], -1)
        X, V = states(RH, VL, Q); bound = np.einsum("ij,ij->i", V, V) < 2 * GM / np.linalg.norm(X, axis=1) * .9999; dof = max(2 * len(T) - 4, 1)
        from scipy.stats import chi2 as CH
        acc = bound & np.isfinite(chi) & (chi <= CH.ppf(.9973, dof)); idx = np.where(acc)[0]
        c2min = float(chi[bound].min()) if bound.any() else None
        if not len(idx) or not write:
            return len(idx), c2min
        Xe, Ve = O.eq_to_ecl(X[acc]), O.eq_to_ecl(V[acc]); a, e, inc, Om, om, M = O.rv_to_elements(Xe, Ve); rB = np.linalg.norm(X[acc], axis=1)
        rng = np.random.default_rng(13); pick_ = rng.choice(idx, size=400, replace=len(idx) < 400); Qd = []
        for n in pick_:
            Qd.append(rng.multivariate_normal(Q[n], np.linalg.pinv(J[n].T @ J[n]) * max(1., chi[n] / dof)))
        Xd, Vd = states(RH[pick_], VL[pick_], np.array(Qd)); okd = np.einsum("ij,ij->i", Vd, Vd) < 2 * GM / np.linalg.norm(Xd, axis=1) * .9999
        dates = np.arange(EP - 150, EP + 150.01, 1.); obs = np.array([Su.obs_b(float(m)) for m in dates]); ra, de = radec(Xd[okd], Vd[okd], dates, obs); Pd = np.stack([np.degrees(ra), np.degrees(de)], -1)
        med = np.c_[np.degrees(np.arctan2(np.median(np.sin(ra), 0), np.median(np.cos(ra), 0))) % 360, np.degrees(np.median(de, 0))]
        sp = np.hypot(((Pd[:, :, 0] - med[None, :, 0] + 180) % 360 - 180) * np.cos(np.radians(med[None, :, 1])), Pd[:, :, 1] - med[None, :, 1]) * 3600; s99 = np.percentile(sp, 99, axis=0)
        np.savez(D + "/draws.npz", mjd=dates, P=Pd, med=med, s99=s99)
        R = dict(nights=len(T), epoch=EP, accepted=int(acc.sum()), r_bary=[float(rB.min()), float(rB.max())], a=[float(a.min()), float(np.median(a)), float(a.max())], e=[float(e.min()), float(np.median(e)), float(e.max())],
                 i=[float(np.degrees(inc).min()), float(np.degrees(np.median(inc))), float(np.degrees(inc).max())], chi2_min=float(chi[acc].min()), nights_used=[float(t) for t in T], **extra, spread99={str(dd): round(float(s99[150 + dd]), 1) for dd in (-60, -30, -15, -5, 5, 15, 30, 60)})
        json.dump(R, open(D + "/range.json", "w"), indent=0); print("range", json.dumps(R), flush=True)
        return len(idx), c2min

    n, c2 = attempt(F_, {}, write=False)
    if n:
        attempt(F_, {}); return
    # No bound orbit through every night (<lead>: chi2_min 20.18 against 20.06 with 5 nights). One night at S/N ~4 can be off; with >= 4 nights,
    # range again without each night in turn and use the subset with the lowest chi2 that has bound orbits. The verdict names the night left out.
    tries = []
    if len(F_) >= 4:
        for j in range(len(F_)):
            sub = F_[:j] + F_[j + 1:]; n_, c_ = attempt(sub, {}, write=False)
            if n_:
                tries.append((c_, j, n_))
    if not tries:
        json.dump(dict(error="no bound orbit fits", chi2_min=c2, nights=len(F_)), open(D + "/range.json", "w")); return
    c_, j, n_ = min(tries)
    attempt(F_[:j] + F_[j + 1:], dict(dropped_night=float(F_[j]["t"]), dropped_peak=float(F_[j]["peak"]), chi2_min_all=c2, leave_one_out=[(float(F_[jj]["t"]), round(cc, 2), nn) for cc, jj, nn in sorted(tries)]))


# ---------------------------------------------------------------- list
def st_list(S, D):
    F, P, E, HB, d0 = spec_field(S); Z = np.load(D + "/draws.npz"); DM, DP, MED, S99 = Z["mjd"], Z["P"], Z["med"], Z["s99"]; used = {e["image"].split("_ooi_")[0] for e in E}
    mid = float(np.mean([e["mjd_mid"] for e in E])); lo, hi = mid - 60, mid + 60; MJD0 = datetime.date(1858, 11, 17).toordinal()
    j = (DM >= lo) & (DM <= hi); box = [float(DP[:, j, 0].min()) - 1.3, float(DP[:, j, 0].max()) + 1.3, float(DP[:, j, 1].min()) - 1.3, float(DP[:, j, 1].max()) + 1.3]
    day = lambda m: datetime.date.fromordinal(int(m) + MJD0).isoformat()
    body = {"outfields": ["md5sum", "archive_filename", "filesize", "caldat", "exposure", "ifilter", "ra_center", "dec_center", "release_date", "dateobs_center"],
            "search": [["instrument", "decam"], ["proc_type", "instcal"], ["caldat", day(lo), day(hi)], ["ra_center", box[0], box[1]], ["dec_center", box[2], box[3]]]}
    rows = None
    for k in range(5):
        try:
            rows = json.loads(urllib.request.urlopen(urllib.request.Request("https://astroarchive.noirlab.edu/api/adv_search/find/?limit=100000", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())[1:]; break
        except Exception as ex:
            print("retry", k, repr(ex)[:100], flush=True); time.sleep(10 * (k + 1))
    exps = {}
    for x in rows or []:
        band = (x["ifilter"] or "").split(" ")[0]
        if band not in ("g", "r", "i", "z", "Y", "VR") or (x["exposure"] or 0) < 30 or (x["release_date"] or "9999") > time.strftime("%Y-%m-%d") or x["ra_center"] is None:
            continue
        try:
            dt = datetime.datetime.fromisoformat(x["dateobs_center"].replace(" ", "T")[:19]); m = dt.toordinal() - MJD0 + (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400
        except Exception:
            m = datetime.date.fromisoformat(x["caldat"]).toordinal() - MJD0 + 1.2
        i = int(round(m - DM[0]))
        if not 0 <= i < len(DM):
            continue
        if S99[i] > 120:
            continue
        pa, pd = DP[:, i, 0], DP[:, i, 1]; Pin = float((np.hypot(((pa - x["ra_center"] + 180) % 360 - 180) * np.cos(np.radians(pd)), pd - x["dec_center"]) < 1.15).mean())
        if Pin < .3:
            continue
        fn = x["archive_filename"].split("/")[-1]; g = re.match(r"(c4d_\d{6}_\d{6})_oo([iwd])_[^_]+_(.+)\.fits\.fz$", fn)
        if g and g.group(1) not in used:
            ee = exps.setdefault(g.group(1), dict(cal=x["caldat"], band=band, t=x["exposure"], P=round(Pin, 3), m=m, vers={})); ee["vers"].setdefault(g.group(3), {})[g.group(2)] = (x["md5sum"], x["filesize"], fn)
    out = {}
    for base, ee in sorted(exps.items(), key=lambda kv: abs(kv[1]["m"] - mid)):
        full = {v: d for v, d in ee["vers"].items() if set(d) == {"i", "w", "d"}}
        if not full:
            continue
        cp = sorted((v for v in full if re.fullmatch(r"v\d+", v)), key=lambda v: int(v[1:])); v = cp[-1] if cp else sorted(full)[0]
        files = [dict(window="confirm", name=full[v][c][2], md5=full[v][c][0], size=full[v][c][1]) for c in "iwd"]
        if len(out) >= 30 or sum(f["size"] for e in out.values() for f in e["files"]) + sum(f["size"] for f in files) > 15e9:
            break
        out[base] = dict(f=F, cal=ee["cal"], band=ee["band"], t=ee["t"], P=ee["P"], cover=[S["id"]], files=files)
    json.dump(dict(windows=[dict(window="confirm", files=[f for e in out.values() for f in e["files"]])], exposures=out), open(D + "/files.json", "w"), indent=0)
    print("list:", len(rows or []), "archive rows;", len(out), "exposures,", round(sum(f["size"] for e in out.values() for f in e["files"]) / 1e9, 1), "GB:", sorted(collections.Counter((e["cal"], e["band"]) for e in out.values()).items()), flush=True)


# ---------------------------------------------------------------- fetch
def md5_of(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def st_fetch(S, D):
    # fetch_scenarios.py skips any file already on disk at the right size, damaged or not (<lead>: one exposure from an earlier
    # fetch had a garbage header). So every listed file already on disk is md5-checked first; a mismatch is moved to data/confirm/_bad/ and fetched again.
    J = json.load(open(D + "/files.json")); n = len(J["exposures"]); note = []
    for e in J["exposures"].values():
        for f in e["files"]:
            q = os.path.join(H, "data/confirm", f["name"])
            if os.path.exists(q) and os.path.getsize(q) == f["size"] and md5_of(q) != f["md5"]:
                os.makedirs(H + "/data/confirm/_bad", exist_ok=True); shutil.move(q, H + "/data/confirm/_bad/" + f["name"] + time.strftime(".%Y%m%d%H%M%S"))
                note.append("md5 mismatch on disk, moved to _bad and refetched: " + f["name"])
    if n:
        p = subprocess.run([PY, "-I", H + "/bin/fetch_scenarios.py", "get", D + "/files.json", H + "/data", "--workers", "3"], capture_output=True, text=True, cwd=C)
        open(D + "/fetch.log", "w").write("\n".join(note + [""]) + p.stdout[-5000:] + p.stderr[-2000:])
    else:
        open(D + "/fetch.log", "w").write("nothing to fetch\n")


def readable(paths):
    """Every HDU header of every file parses (a damaged download fails here, not inside confirm_rest)."""
    import warnings
    from astropy.io import fits
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for q in paths:
                with fits.open(q, memmap=False) as hl:
                    for h in hl:
                        [c.value for c in h.header.cards]
        return True
    except Exception as ex:
        return repr(ex)[:160]


# ---------------------------------------------------------------- check
def st_check(S, D):
    F, P, E, HB, d0 = spec_field(S); held = {e["image"] for e in HB}
    src = open(C + "/confirm_rest.py").read().split('\nif __name__ == "__main__":')[0]
    OLD = 'used_here = {i.split("_ooi_")[0] for i in E}'; assert src.count(OLD) == 1
    src = src.replace(OLD, 'used_here = {i.split("_ooi_")[0] for i in E if i not in HELD}')
    ns = {"__name__": "confirm_rest_lead", "HELD": held}; exec(compile(src, "confirm_rest.py", "exec"), ns); G = ns["G"]
    from astropy.time import Time

    SKIP = {}; RD = {}

    def targets(f):
        rows = []; seen = set()
        for b, e in json.load(open(D + "/files.json"))["exposures"].items():
            ps = [os.path.join(H, "data/confirm", next(x["name"] for x in e["files"] if f"_oo{c}_" in x["name"])) for c in "iwd"]
            if b in seen:
                continue
            seen.add(b)
            if not all(os.path.exists(q) for q in ps):
                SKIP[b] = "not fetched (" + ", ".join(os.path.basename(q) for q in ps if not os.path.exists(q)) + ")"; continue
            ok = RD[b] if b in RD else RD.setdefault(b, readable(ps))
            if ok is not True:
                SKIP[b] = "unreadable: " + ok; continue
            rows.append(dict(base=b, cal=e["cal"], band=e["band"], t=e["t"], paths=ps))
        for e in HB:
            b = e["image"].split("_ooi_")[0]; ps = [os.path.join(d0, e[k]) for k in ("image", "weight", "mask")]
            if b in seen:  # the same exposure also fetched from NOIRLab: count it once (fixed 2026-11-09 23:20 PST; F9 01-25 had two exposures twice)
                continue
            if all(os.path.exists(q) for q in ps):
                seen.add(b); rows.append(dict(base=b, cal=Time(math.floor(e["mjd_mid"] - .5), format="mjd").iso[:10], band=e["band"], t=e["exptime"], paths=ps))
        return rows
    ns["targets"] = targets
    Eall = sorted(P["all_exposures"], key=lambda e: e["mjd_mid"])

    def track(c):
        tr = []
        for e in Eall:
            if e["image"] in held:
                continue
            q = G.positions(dict(state=np.array(c["state"], float), epoch=c["epoch_mjd"]), [e["mjd_mid"]], [np.array([e["obs_x"], e["obs_y"], e["obs_z"]])]); tr.append([e["mjd_mid"], float(q[0][0]), float(q[1][0]), e["image"]])
        return tr
    os.makedirs(D + "/check", exist_ok=True); lead = pick(S); ctl, pdes = control(S); T = []
    for kind, name, c in (("candidate", "lead", lead), ("control", f"control_{pdes}".replace(" ", "_"), ctl)):
        if c is None:
            continue
        base = dict(f=F, sid=S["sid"], track=track(c), state=c["state"], epoch=float(c["epoch_mjd"]), name=name); T.append(dict(base, kind=kind, off=[0, 0], id=name))
        if kind == "candidate":
            T += [dict(base, kind="null", off=list(o), id=f"null{i}") for i, o in enumerate(ns["NULLS"])]
    print("check: target images", sorted({(x["cal"], x["band"]) for x in targets(F)}), flush=True)
    os.makedirs(D + "/check", exist_ok=True); json.dump(SKIP, open(D + "/check/skipped.json", "w"), indent=0)
    if SKIP:
        print("check: skipped", len(SKIP), "exposures:", SKIP, flush=True)
    for t in T:
        out = os.path.join(D, "check", t["id"] + ".json")
        if not (os.path.exists(out) and os.path.getsize(out)):
            ns["run"](t, out)


# ---------------------------------------------------------------- cfht
def ssh(*a, timeout=300):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-S", SOCK, NERSC] + list(a), capture_output=True, text=True, timeout=timeout)


def st_cfht(S, D):
    os.makedirs(D + "/cfht", exist_ok=True); F, P, E, HB, d0 = spec_field(S); mid = float(np.mean([e["mjd_mid"] for e in E]))
    lead_res = json.load(open(D + "/check/lead.json")) if os.path.exists(D + "/check/lead.json") else None
    if not lead_res or not lead_res.get("points"):
        json.dump(dict(skipped="no lead fit from the check stage"), open(D + "/cfht/result.json", "w")); return
    src = open(C + "/confirm_rest.py").read().split('\nif __name__ == "__main__":')[0]; ns = {"__name__": "cr"}; exec(compile(src, "confirm_rest.py", "exec"), ns)
    Cf, G, family = ns["C"], ns["G"], ns["family"]
    import astropy.units as u
    from astropy.coordinates import Angle, EarthLocation, get_body_barycentric, solar_system_ephemeris
    from astropy.time import Time
    MK = EarthLocation.from_geodetic(-155.4681 * u.deg, 19.8253 * u.deg, 4204 * u.m)

    def obs_mk(m):
        t = Time(m, format="mjd", scale="utc")
        with solar_system_ephemeris.set("builtin"):
            return (get_body_barycentric("earth", t) - get_body_barycentric("sun", t)).xyz.to(u.au).value + MK.get_gcrs_posvel(t)[0].xyz.to(u.au).value
    obs = [(m, ra, de, Cf.observer(m), s) for m, ra, de, s in lead_res["points"]]; fam, rms = family(obs, np.array(lead_res["state"]), lead_res["epoch"]); ep = lead_res["epoch"]
    box = H + f"/candidates/archive_cover/box_{F}.json"; M = []; seen = set()
    if os.path.exists(box):
        src_rows = [x for x in json.load(open(box)) if x["coll"] == "CFHT" and x["inst"] == "MegaPrime"]
    else:
        q0 = G.positions(dict(state=np.array(lead_res["state"]), epoch=ep), [mid], [Cf.observer(mid)]); ra0, de0 = float(q0[0][0]), float(q0[1][0])
        q = (f"SELECT obs_id, s_ra, s_dec, s_fov, t_min, t_exptime FROM ivoa.ObsCore WHERE obs_collection = 'CFHT' AND instrument_name = 'MegaPrime' AND dataproduct_type = 'image' AND calib_level = 1 "
             f"AND s_ra BETWEEN {ra0 - 2.5} AND {ra0 + 2.5} AND s_dec BETWEEN {de0 - 2} AND {de0 + 2} AND t_min BETWEEN {mid - 60} AND {mid + 60}")
        try:
            t = subprocess.run(["curl", "-sS", "-L", "-m", "1200", "-A", "archive-cover/1.0", "--data-urlencode", "REQUEST=doQuery", "--data-urlencode", "LANG=ADQL", "--data-urlencode", "FORMAT=csv", "--data-urlencode", f"QUERY={q}",
                                "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync"], capture_output=True, text=True, timeout=1300).stdout
            import csv, io
            src_rows = [dict(coll="CFHT", inst="MegaPrime", id=r["obs_id"], ra=float(r["s_ra"]), dec=float(r["s_dec"]), mjd=float(r["t_min"]), exp=float(r["t_exptime"])) for r in csv.DictReader(io.StringIO(t))]
        except Exception as ex:
            json.dump(dict(skipped=f"CADC query failed: {repr(ex)[:100]}"), open(D + "/cfht/result.json", "w")); return
    for x in src_rows:
        if x["exp"] >= 60 and abs(x["mjd"] - mid) <= 60 and x["id"] not in seen:
            seen.add(x["id"]); M.append(x)
    T = []
    for x in M:
        m = x["mjd"] + x["exp"] / 172800; o = obs_mk(m); Pp = []; P2 = []
        for st, _ in fam.values():
            q = G.positions(dict(state=st, epoch=ep), [m, m + 1 / 24.], [o, obs_mk(m + 1 / 24.)])
            if q is not None:
                Pp.append((float(q[0][0]), float(q[1][0]))); P2.append((float(q[0][1]), float(q[1][1])))
        if not Pp:
            continue
        a0 = float(np.median([p[0] for p in Pp])); dd = float(np.median([p[1] for p in Pp])); cd = math.cos(math.radians(dd)); spread = max(math.hypot((p[0] - a0) * cd, p[1] - dd) for p in Pp) * 3600
        if math.hypot((x["ra"] - a0) * cd, x["dec"] - dd) > .8 or spread >= 1.0:
            continue
        a1 = float(np.median([p[0] for p in P2])); d1 = float(np.median([p[1] for p in P2]))
        T.append(dict(name="Z", id=x["id"], mjd=float(m), exp=x["exp"], ra=a0, dec=dd, rra=float((a1 - a0) * 3600 * cd), rdec=float((d1 - dd) * 3600), V=None, spread=round(spread, 2)))
    if not T:
        json.dump(dict(skipped="no CFHT image holds the lead with spread < 1\"", images_listed=len(M)), open(D + "/cfht/result.json", "w")); return
    night = collections.defaultdict(list)
    for t in T:
        night[int(t["mjd"])].append(t)
    for n, v in sorted(night.items()):
        x = v[0]
        qq = {"-mime": "text", "-ra": x["ra"], "-dec": x["dec"], "-rd": .75, "-ep": x["mjd"] + 2400000.5, "-loc": "568", "-output": "all", "-objFilter": "110", "-filter": "0", "-refsys": "EQJ2000", "-from": "archive-cover"}
        try:
            txt = urllib.request.urlopen(urllib.request.Request("https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?" + urllib.parse.urlencode(qq), headers={"User-Agent": "archive-cover/1.0"}), timeout=120).read().decode("utf-8", "replace")
        except Exception:
            continue
        head = None
        for l in txt.splitlines():
            if l.startswith("#") and "|" in l:
                head = [c.strip(" #") for c in l.split("|")]; continue
            if not head or "|" not in l or l.startswith("#"):
                continue
            r = dict(zip(head, [c.strip() for c in l.split("|")]))
            try:
                dh = float(r.get("Dh(ua)", 0)); V = float(r["Mv"]); a0 = Angle(r["RA(h)"], unit=u.hourangle).deg; d0_ = Angle(r["DE(deg)"], unit=u.deg).deg; rr, rd = float(r["dRA(arcsec/h)"]), float(r["dDEC(arcsec/h)"])
            except (KeyError, ValueError):
                continue
            if dh < 25 or V > 24.8:
                continue
            for y in [t for t in T if t["name"] == "Z" and int(t["mjd"]) == n]:
                dt = (y["mjd"] - x["mjd"]) * 24; T.append(dict(name=r["Name"], id=y["id"], mjd=y["mjd"], exp=y["exp"], ra=a0 + rr * dt / 3600 / math.cos(math.radians(d0_)), dec=d0_ + rd * dt / 3600, rra=rr, rdec=rd, V=V))
        time.sleep(.5)
    tj = D + "/cfht/targets.json"; json.dump(T, open(tj, "w"))
    if os.path.exists(D + "/cfht/phot_result.json"):  # a finished NERSC result: only re-evaluate
        return cfht_eval(D, "earlier job")
    if ssh("true", timeout=60).returncode != 0 or subprocess.run(["ssh", "-O", "check", "-S", SOCK, NERSC], capture_output=True).returncode != 0:
        json.dump(dict(skipped="NERSC socket not alive (MFA login needed)", images=sum(t["name"] == "Z" for t in T)), open(D + "/cfht/result.json", "w")); return
    zr = [t for t in T]; box_ = (min(t["ra"] for t in zr) - .8, max(t["ra"] for t in zr) + .8, min(t["dec"] for t in zr) - .8, max(t["dec"] for t in zr) + .8)
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", S["id"]); rdir = f"{NC}/rsi/candidates/leads/{tag}"
    wrap = (f'import os, sys\nsrc = open("{NC}/rsi/candidates/cfht_fetch.py").read()\nOLD = "WHERE ra BETWEEN 213 AND 235 AND dec BETWEEN -20 AND -11 AND"\nassert src.count(OLD) == 1\n'
            f'exec(compile(src.replace(OLD, "WHERE ra BETWEEN {box_[0]:.3f} AND {box_[1]:.3f} AND dec BETWEEN {box_[2]:.3f} AND {box_[3]:.3f} AND"), "cfht_fetch.py", "exec"), {{"__name__": "__main__"}})\n')
    open(D + "/cfht/fetch_box.py", "w").write(wrap)
    sb = (f"#!/bin/bash\n#SBATCH -A $NERSC_CPU_ACCOUNT\n#SBATCH -C cpu\n#SBATCH -q debug\n#SBATCH -N 1\n#SBATCH -t 00:30:00\n#SBATCH -J lead-cfht\n#SBATCH -o {rdir}/job-%j.out\n"
          f"export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore\nulimit -v $((8*1024*1024))\ncd {NC}/rsi/candidates\n"
          f"{NC}/env/bin/python -I {rdir}/fetch_box.py {rdir}/targets.json {NC}/rsi/data/cfht_leads/{tag}\n{NC}/env/bin/python -I cfht_phot.py {rdir}/targets.json {NC}/rsi/data/cfht_leads/{tag} {rdir}/out\n")
    open(D + "/cfht/job.sbatch", "w").write(sb)
    ssh(f"mkdir -p {rdir}"); subprocess.run(["scp", "-q", "-o", f"ControlPath={SOCK}", tj, D + "/cfht/fetch_box.py", D + "/cfht/job.sbatch", f"{NERSC}:{rdir}/"], check=True)
    o = ssh(f"cd {rdir} && sbatch job.sbatch").stdout; jid = re.search(r"(\d+)", o).group(1); print("cfht: NERSC job", jid, flush=True)
    for _ in range(240):
        st = ssh(f"sacct -j {jid} -X -n -o State%20").stdout.strip().split("\n")[0].strip()
        if st and not st.startswith(("PENDING", "RUNNING", "REQUEUED", "CONFIGURING")):
            break
        time.sleep(60)
    subprocess.run(["scp", "-q", "-o", f"ControlPath={SOCK}", f"{NERSC}:{rdir}/out/result.json", D + "/cfht/phot_result.json"])
    log = ssh(f"cat {rdir}/job-{jid}.out").stdout; open(D + "/cfht/job.log", "w").write(log)
    if not os.path.exists(D + "/cfht/phot_result.json"):
        json.dump(dict(skipped=f"NERSC job {jid} {st}: no result"), open(D + "/cfht/result.json", "w")); return
    return cfht_eval(D, jid)


def cfht_eval(D, jid):
    R = json.load(open(D + "/cfht/phot_result.json")); z = R.get("Z")
    if not z:
        json.dump(dict(job=jid, verdict="UNDECIDED", why="the lead fell on no measured CCD"), open(D + "/cfht/result.json", "w")); return
    a = z["all"]; nights = z["nights"]; good_n = [n for n in nights if n["snr"] >= 3 and math.hypot(*n["off"]) <= .6]
    here = {str(n["night"]): n["flux"] for n in nights}  # a faint fixed source under the track does not count against a far brighter mover
    stat_ok = all(abs(v["flux"] / v["err"]) < 3 or v["flux"] < .25 * here.get(k, 0) for k, v in z.get("static", {}).items() if v["err"] > 0)
    ctl = [dict(name=k, V=v["V"], snr=v["all"]["snr"]) for k, v in R.items() if k != "Z"]; c24 = [c for c in ctl if c["V"] is not None and c["V"] <= 24]
    ctl_ok = (sum(c["snr"] >= 5 for c in c24) >= 2 / 3 * len(c24)) if c24 else any(c["snr"] >= 5 for c in ctl)
    if a["snr"] >= 5 and a["snr"] >= a["null_med"] + 5 * a["null_sig"] and len(good_n) >= 2 and stat_ok:
        v = "FOUND" if ctl_ok else "FOUND BUT CONTROLS FAILED (does not count)"
    elif a["snr"] < 2 and a["n"] >= 6:
        v = "NOT SEEN" if ctl_ok else "NOT SEEN BUT CONTROLS FAILED (does not count)"
    else:
        v = "UNDECIDED"
    json.dump(dict(job=jid, verdict=v, snr=a["snr"], n=a["n"], mag=a["mag"], null=[a["null_med"], a["null_sig"]], nights=[[n["night"], round(n["snr"], 1), n["off"]] for n in nights], static_ok=stat_ok, controls=ctl, controls_ok=ctl_ok), open(D + "/cfht/result.json", "w"), indent=0)
    print("cfht:", v, flush=True)


# ---------------------------------------------------------------- verdict
def st_verdict(S, D):
    pos = json.load(open(D + "/pos.json")); rng = json.load(open(D + "/range.json")) if os.path.exists(D + "/range.json") else {}
    ck = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(D + "/check/*.json")}; lead = ck.get("lead", {}); nulls = [v for k, v in ck.items() if k.startswith("null")]; ctl = [v for k, v in ck.items() if k.startswith("control")]
    cf = json.load(open(D + "/cfht/result.json")) if os.path.exists(D + "/cfht/result.json") else {}
    null_gain = sum(len(v.get("new", [])) for v in nulls); dec_new = [(n["cal"], n["band"], n["snr"]) for n in lead.get("new", [])]; dec_counts = null_gain == 0
    seen = (dec_counts and dec_new) or cf.get("verdict") == "FOUND"
    overall = "SEEN IN OTHER IMAGES" if seen else ("NOT SEEN" if cf.get("verdict") == "NOT SEEN" and not dec_new else "NO DECISIVE DATA")
    R = dict(id=S["id"], overall=overall, field=pos.get("field"), lead_nights=[(o["night"], round(o["peak"], 1)) for o in pos.get("lead", [])], range=rng.get("r_bary"), decam_new=dec_new, decam_missed=[(m.get("cal"), m.get("band"), m.get("snr"), m.get("rejected") or m.get("why")) for m in lead.get("missed", [])],
             nulls_gained=null_gain, control=[(c.get("name"), [(n["cal"], n["snr"]) for n in c.get("new", [])]) for c in ctl], cfht=cf, arc_days=lead.get("arc_days"), rms=lead.get("rms"))
    json.dump(R, open(D + "/summary.json", "w"), indent=0)
    line = f"{overall}; DECam new nights {dec_new or 'none'} (fakes gained {null_gain}); CFHT {cf.get('verdict') or cf.get('skipped') or 'not run'}" + (f" S/N {cf['snr']:.1f} in {cf['n']} images" if cf.get("snr") is not None else "")
    if rng.get("error"):
        line += f"; ranging: {rng['error']}" + (f" (chi2_min {rng['chi2_min']:.1f}, {rng.get('nights')} nights)" if rng.get("chi2_min") is not None else "")
    elif rng.get("dropped_night") is not None:
        line += f"; ranging left out night {int(rng['dropped_night'])} (peak {rng['dropped_peak']:.1f}): no bound orbit through all nights (chi2_min {rng['chi2_min_all']:.1f})"
    open(D + "/verdict.txt", "w").write(line + "\n"); print("VERDICT", S["id"], ":", line, flush=True)


STAGES = [("pos", "pos.json", st_pos), ("range", "range.json", st_range), ("list", "files.json", st_list), ("fetch", "fetch.log", st_fetch), ("check", "check/.done", st_check),
          ("cfht", "cfht/result.json", st_cfht), ("verdict", "verdict.txt", st_verdict)]

if __name__ == "__main__":
    spec, D = sys.argv[1], sys.argv[2]; os.makedirs(D, exist_ok=True); S = json.load(open(spec)); json.dump(S, open(D + "/spec.json", "w"))
    if len(sys.argv) > 3:  # one stage, in its own process
        name = sys.argv[3]; fn = {n: f for n, _, f in STAGES}[name]; fn(S, D)
        if name == "check":
            open(D + "/check/.done", "w").write(time.strftime("%F %T"))
        sys.exit(0)
    for name, out, fn in STAGES:
        if os.path.exists(os.path.join(D, out)):
            continue
        if name in ("list", "fetch", "check", "cfht") and "error" in (json.load(open(D + "/range.json")) if os.path.exists(D + "/range.json") else {}):
            break
        t0 = time.time(); p = subprocess.run([PY, "-I", __file__, spec, D, name], capture_output=True, text=True, cwd=C, env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"))
        open(os.path.join(D, f"stage_{name}.log"), "w").write(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-8000:])
        print(time.strftime("%T"), name, "exit", p.returncode, f"{time.time() - t0:.0f} s", flush=True)
        if p.returncode != 0 and name != "cfht":
            open(os.path.join(D, "verdict.txt"), "w").write(f"ERROR in stage {name} (see stage_{name}.log)\n"); break
    if not os.path.exists(os.path.join(D, "verdict.txt")):
        subprocess.run([PY, "-I", __file__, spec, D, "verdict"], cwd=C)
    print(open(os.path.join(D, "verdict.txt")).read().strip(), flush=True)
