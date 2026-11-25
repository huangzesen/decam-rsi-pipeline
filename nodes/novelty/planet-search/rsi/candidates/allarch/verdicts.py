"""Verdicts for the all-archive requests of 2026-11-17 and 2026-11-19 (rsi/decam-big-tno/runs/CONFIRM_REQUEST_2026111[79].md), one rule for every body.
A detection counts when: forced S/N >= 5 against the null; peak within max(1", 2 sigma) of the prediction (prediction 1-sigma <= 3");
no Legacy Survey DR10 (else DR9) source within 1.5" of the place; no DR10 source brighter than r 21 within 3"; for CFHT, the same sky on
other nights carries < half the flux. A night counts once. SEEN: >= 2 nights (any archives), or an ITF tracklet proven the same body
(itf_joint.json); SEEN ONCE: one night; else NO DECISIVE DATA. Rubin DP2 detections are data under our data rights (rubin_rights: true).
Earlier verdicts in the confirm file are kept (prefixed "earlier"). Writes rsi/decam-big-tno/runs/<run>/confirm/<id>.json, one line per
body in confirm_queue/RESULTS.md, and allarch/VERDICTS.md. Usage: ~/anaconda3/bin/python -I verdicts.py [ID_SUBSTRING ...]"""
import glob, json, math, os, sys, time, urllib.parse, urllib.request
H = os.path.dirname(os.path.abspath(__file__)); C = os.path.dirname(H); D = os.path.expanduser("$ALE_HOME/rsi/decam-big-tno")
want = sys.argv[1:]
L = lambda f: json.load(open(f"{H}/{f}")) if os.path.exists(f"{H}/{f}") else {}
def LM(*fs):  # later files add bodies or replace them
    out = {}
    for f in fs:
        out.update(L(f))
    return out


def LF(*fs):  # measurements: the first file's entry wins, later files only add (deterministic reruns)
    out = {}
    for f in fs:
        for k, v in L(f).items():
            if "_v2" in f:  # 2026-11-21 vetting pass: a re-measurement on the new orbit replaces the old one
                out[k] = v
            else:
                out.setdefault(k, v)
    return out


def counts(snr, frac, N, mag_ok):
    # 1. the null rule (2026-11-17 07:00): k = random places of N as high; a night counts if S/N >= 10 with frac <= 5 %, or (k+1)/N <= 2 %;
    # a single night (SEEN ONCE) needs S/N >= 10 with a brightness check passed, or (k+1)/N <= 0.5 %. Returns (counts, single-night grade)
    k = round((frac if frac is not None else 1) * N); upper = (k + 1) / N
    night = (snr >= 10 and (frac or 0) <= 0.05) or upper <= 0.02
    once = (snr >= 10 and mag_ok is True) or upper <= 0.005
    return night, once
B = L("bodies_dbt.json"); _ids = {b["id"] for b in B}
for _f in ("bodies_p4.json", "bodies_c.json", "bodies_a15.json", "bodies_u05.json", "bodies_re.json", "bodies_u06.json", "bodies_u16.json"):  # 2026-11-17 P4 (s10b), 2026-11-19 new candidates, 11-19b third 5 % (sa15)
    B += [b for b in (L(_f) or []) if b["id"] not in _ids]; _ids = {b["id"] for b in B}
for _b in (L("bodies_v2.json") or []):  # 2026-11-21 vetting pass: a re-run body replaces its older entry (new arc, joint orbit)
    B = [x for x in B if x["id"] != _b["id"]] + [_b]
_ids = {b["id"] for b in B}
CNUM = L("cnum.json"); REQ = D + "/runs/CONFIRM_REQUEST_20261120.md"
NOV = H + "/novelty_1119"; SKY = L("skybot_1119.json"); PS1 = L("ps1_path.json"); NSC = L("nsc_path.json"); NSCP = L("nsc_pairs.json"); WIDE = {a: L(f"wide_{a}.json") for a in ("decam", "eso")}
INST = {"c4d": "DECam", "k4m": "Mayall Mosaic3", "ksb": "Bok 90Prime"}
PRI = {}  # priorities as the <coordinator-session> session set them in the request (rebuilt 2026-11-17 05:03)
for l in open(REQ):
    c_ = [x.strip() for x in l.split("|")]
    if len(c_) > 5 and c_[1].isdigit():
        PRI[next((x for x in c_ if x.startswith("dbt-")), c_[4])] = int(c_[1])  # the id column, wherever the table puts it
for b in B:
    b["pri"] = PRI.get(b["id"], b["pri"])
cd, cc, ce, cs, cr = (LM(f"cover_{a}.json", f"cover_{a}_p4.json", f"cover_{a}_c.json", f"cover_{a}_a15.json", f"cover_{a}_u05.json", f"cover_{a}_re.json", f"cover_{a}_u06.json", f"cover_{a}_u16.json", f"cover_{a}_v2.json") for a in ("decam", "cfht", "eso", "smoka", "rubin"))
dp, ep = LF("decam_phot.json", "decam_phot_p4.json", "decam_phot_c.json", "decam_phot_a15.json", "decam_phot_u05.json", "decam_phot_re.json", "decam_phot_u06.json", "decam_phot_u16.json", "decam_phot_v2.json"), LF("eso_p3_phot.json", "eso_p3_phot_p4.json", "eso_p3_phot_c.json", "eso_p3_phot_a15.json", "eso_p3_phot_u05.json", "eso_p3_phot_re.json", "eso_p3_phot_u06.json", "eso_p3_phot_u16.json", "eso_p3_phot_v2.json")
hs, ij = LM("hsc_select.json", "hsc_select_p4.json", "hsc_select_c.json", "hsc_select_a15.json", "hsc_select_u05.json", "hsc_select_re.json", "hsc_select_u06.json", "hsc_select_u16.json", "hsc_select_v2.json"), LM("itf_joint.json", "itf_joint_p4_wide.json", "itf_joint_all.json", "itf_joint_a15.json", "itf_joint_u05.json", "itf_joint_re.json", "itf_joint_u06.json", "itf_joint_u16.json", "itf_joint_v2.json")  # ITF: today's scan replaces
hp = L("hsc_phot.json"); REQ1 = {l.split()[0] for l in open(H + "/hsc_request_frames.txt")} if os.path.exists(H + "/hsc_request_frames.txt") else set()
HSC_SETS = ("P03_1117", "P03_1117b", "P03_1119", "P03_1119b", "P03_a15", "P03_u05", "P03_re", "P03_u06", "P03_u16")  # one catalogue folder per SMOKA request
HSC_CAT = "$DATA_DRIVE/ale-astronomy/planet-search/rsi/data/smoka_hsc_P03_1117_cat"
GAIA_D = C + "/cand9_gaia"; _zp = {}


def hsc_zp(fr):  # aperture (r 5 px) zero point against Gaia G from the frame's own solved stars; None if too few
    if fr not in _zp:
        z = None
        try:
            cat_ = next((c_ for c_ in tuple(HSC_CAT.replace("P03_1117", s_) for s_ in HSC_SETS) if os.path.exists(f"{c_}/{fr}.npz")), HSC_CAT)
            S = np.load(f"{cat_}/{fr}.npz"); A = np.load(f"{cat_}/{fr}_ast.npz"); w = json.loads(str(S["meta"]))["wcs"]
            rf = f"{H}/hsc_ref/{fr}.npz"  # Legacy Survey reference of that CCD (r mags; stars = Tractor sources with a Gaia proper motion)
            if os.path.exists(rf):
                G = dict(np.load(rf)); st_ = (G["pmra"] != 0) | (G["pmdec"] != 0); G = {k_: v_[st_] for k_, v_ in G.items()}
            else:
                G = np.load(f"{GAIA_D}/{round(w['CRVAL1'], 3):.3f}_{round(w['CRVAL2'], 3):.3f}.npz")
            from scipy.spatial import cKDTree
            ok = (S["snr"] > 30) & (S["sat"] == 0) & (S["flux"] > 0); ca = np.cos(np.radians(np.median(A["dec"])))
            t = cKDTree(np.c_[G["ra"] * ca, G["dec"]]); d, k = t.query(np.c_[A["ra"][ok] * ca, A["dec"][ok]])
            m = (d * 3600 < 0.5) & (G["phot_g_mean_mag"][k] > 16) & (G["phot_g_mean_mag"][k] < 21.5)
            if m.sum() >= 10:
                z = float(np.median(G["phot_g_mean_mag"][k][m] + 2.5 * np.log10(S["flux"][ok][m])))
        except Exception:
            z = None
        _zp[fr] = z
    return _zp[fr]
tags = {v: k for k, v in L("cfht_tags.json").items()}
HH = {o["id"]: o.get("H") for o in json.load(open(D + "/progress/objects/objects.json"))["decam_big_tno"]}
sys.path.insert(0, os.path.expanduser("$ALE_HOME/planet-search/linker"))
import ast, numpy as np
import orbit_fit as OF
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
OFF = {"g": -0.45, "r": 0.3, "i": 0.6, "z": 0.8, "u": -1.6, "gri": 0.2}  # V - band for a TNO (known_yield.VMINUS; gri ~ r-i mean)


def vpred(arc, m):
    A = json.load(open(arc)); lit_ = lambda v: ast.literal_eval(v) if isinstance(v, str) else v
    p = np.array(lit_(A["p"]), float); t0 = float(A["epoch_tdb"]); R, tdb = OF.observer_bary(np.atleast_1d(m), "W84")
    X = OF.propagate(p[:3], p[3:], np.atleast_1d(tdb) - t0); S = get_body_barycentric("sun", Time(np.atleast_1d(m), format="mjd")).xyz.value.T
    return np.linalg.norm(X - S, axis=1), np.linalg.norm(X - np.atleast_2d(R), axis=1)


def vmag(b, m):
    rh, dl = vpred(b["arc"], m); return HH[b["id"]] + 5 * math.log10(rh[0] * dl[0]) - 0.3


def stack_nights(arch, rows, smax, det, notes, miss, tried):
    """Per-night stack at the predicted place (2026-11-19, "shift-and-stack along the motion"): each frame is already measured at the
    body's own predicted place at its own time, so the night's stack is the sum of the frames' S/N over sqrt(n) (equal weights; under
    the null each frame's S/N is about unit normal). Only frames whose prediction 1-sigma is <= smax (the aperture holds the body), on
    nights with >= 2 such frames. A night counts when the stack reaches S/N 7 with every frame positive and no Legacy Survey source at
    the place; NOT SEEN when the frames' expected S/N add (in quadrature) to >= 10 and the stack is < 2."""
    by = {}
    for n, snr, sg, ra, de, m, ex in rows:
        if sg is not None and sg <= smax and snr is not None:
            by.setdefault(n, []).append((snr, ra, de, m, ex if sg <= 0.3 else None))
    out = []
    for n, L_ in sorted(by.items()):
        if len(L_) < 2 or any(d[1] == n and d[0].split()[0] == arch.split()[0] for d in det):  # nights already seen in single frames are not stacked again
            continue
        st = sum(x[0] for x in L_) / math.sqrt(len(L_)); exs = [x[4] for x in L_ if x[4] is not None]; ex = math.sqrt(sum(e * e for e in exs)) if len(exs) == len(L_) else None
        out.append(f"{day(n + .5)} {len(L_)} frames: stack S/N {st:.1f}" + (f" (expected {ex:.0f})" if ex else ""))
        if st >= 7 and min(x[0] for x in L_) > 0:
            ok, why = static_ok(L_[0][1], L_[0][2])
            notes.append(f"{arch} night stack {day(n + .5)} ({len(L_)} frames at the predicted place): S/N {st:.1f}; {why}")
            if ok:
                det.append((arch + " (night stack)", n, "", st, False))
        elif ex and ex >= 10 and st < 2:
            miss.append(f"{arch} night stack {day(n + .5)} ({len(L_)} frames): expected S/N {ex:.0f}, measured {st:.1f}")
    if out:
        tried.append(f"{arch}, per-night stacks at the predicted place (frames with 1-sigma <= {smax}\"): " + "; ".join(out[:8]) + (" ..." if len(out) > 8 else ""))


def cfht_band(b):
    b = b.split(".")[0]
    return "gri" if b == "gri" else b[0]
day = lambda m: time.strftime("%Y-%m-%d", time.gmtime((m - 40587) * 86400))
TOK = {"User-Agent": "archive-cover/1.0", "X-DL-AuthToken": "anonymous.0.0.anon_access"}
DRC = H + "/dr10_cache.json"; drc = json.load(open(DRC)) if os.path.exists(DRC) else {}


sys.path.insert(0, H); import conecache as CC
CONES = H + "/dr10_cones.json"; cones = CC.load(CONES); CC.register(CONES, cones)


def dr10_cone(ra, de):
    # one 150" cone per sky place, reused for every check within 60" of its centre (Data Lab answers in ~45 s per query on 2026-11-17)
    cd_ = math.cos(math.radians(de))
    for c in cones:
        if math.hypot((c["ra"] - ra) * cd_, c["dec"] - de) * 3600 < 60:
            return c
    c = None
    for tab in ("ls_dr10.tractor", "ls_dr9.tractor"):
        q = f"SELECT ra, dec, mag_r FROM {tab} WHERE q3c_radial_query(ra, dec, {ra}, {de}, {150 / 3600:.6f})"
        try:
            L_ = urllib.request.urlopen(urllib.request.Request("https://datalab.noirlab.edu/query/query?" + urllib.parse.urlencode(dict(sql=q, ofmt="csv", **{"async": "false"})), headers=TOK), timeout=300).read().decode().strip().splitlines()
        except Exception as e:
            return dict(error=repr(e)[:100])
        rows = [[float(a), float(b_), float(m) if m not in ("", "Infinity", "NaN", "inf") else 99.] for a, b_, m in (r.split(",") for r in L_[1:])]
        if rows:
            c = dict(ra=ra, dec=de, table=tab, rows=rows); break
    c = c or dict(ra=ra, dec=de, table=None, rows=[])
    cones.append(c); CC.save(CONES, cones); return c


def dr10(ra, de):
    k = f"{ra:.5f},{de:.5f}"
    if k not in drc:
        c = dr10_cone(ra, de)
        if "error" not in c:
            cd_ = math.cos(math.radians(de)); sep = [(m, math.hypot((a - ra) * cd_, b_ - de) * 3600) for a, b_, m in c["rows"]]
            return dict(table=c["table"], n60=sum(1 for m, x in sep if x < 60), near=[(m, x) for m, x in sep if x < 3]) if c["table"] else dict(table=None)
        out = None
        for tab, col in (("ls_dr10.tractor", "mag_r"), ("ls_dr9.tractor", "mag_r")):
            sql = f"SELECT {col}, q3c_dist(ra, dec, {ra}, {de}) * 3600 AS sep FROM {tab} WHERE q3c_radial_query(ra, dec, {ra}, {de}, {3 / 3600:.6f})"
            n60 = f"SELECT count(*) FROM {tab} WHERE q3c_radial_query(ra, dec, {ra}, {de}, {60 / 3600:.6f})"
            try:
                g = lambda q: urllib.request.urlopen(urllib.request.Request("https://datalab.noirlab.edu/query/query?" + urllib.parse.urlencode(dict(sql=q, ofmt="csv", **{"async": "false"})), headers=TOK), timeout=300).read().decode().strip().splitlines()
                cov = int(g(n60)[-1])
                if cov == 0:
                    continue
                rows = [(float(a) if a not in ("", "Infinity", "NaN", "inf") else 99., float(s)) for a, s in (r.split(",") for r in g(sql)[1:])]
                out = dict(table=tab, n60=cov, near=rows); break
            except Exception as e:
                out = dict(error=repr(e)[:100])
        drc[k] = out or dict(table=None); json.dump(drc, open(DRC, "w"))
    return drc[k]


def static_ok(ra, de):
    r = dr10(ra, de)
    if not r.get("table"):
        return None, "no Legacy Survey coverage"
    close = [s for m, s in r["near"] if s < 1.5]; bright = [(m, s) for m, s in r["near"] if m < 21]
    if close:
        return False, f"{r['table'].split('.')[0].upper()} source {min(close):.1f}\" away"
    if bright:
        return False, f"r {bright[0][0]:.1f} source {bright[0][1]:.1f}\" away"
    return True, f"no {r['table'].split('.')[0].upper()} source within 1.5\" ({r['n60']} within 60\")"


now = time.strftime("%Y-%m-%d %H:%M"); lines = []; summary = []
for b in sorted(B, key=lambda b: (b["pri"], b["id"])):
    i = b["id"]
    if want and not any(w in i for w in want):
        continue
    if os.environ.get("VSKIP") and any(x in i for x in os.environ["VSKIP"].split(",")):
        continue  # bodies whose checks are still running (2026-11-20: VSKIP=su05); a full run without it follows
    if "LEAD13" in i:
        continue
    tried, det, notes, miss, soft, likely = [], [], [], [], [], []
    # DECam other programs (same InstCal pixels NSC DR2 uses for 2012-2019)
    D_ = [x for x in cd.get(i, []) if not x.get("ours")]
    ph = sorted([v for k, v in dp.items() if k.startswith(i + "|")], key=lambda v: v.get("mjd", 0))
    if D_:
        meas = [v for v in ph if "snr" in v]
        tried.append(f"DECam InstCal of other programs (not in any run's lists; NSC DR2 uses the same pixels for 2012-2019): {len(D_)} exposures on the path, "
                     f"{len(meas)} measured, {sum(1 for v in ph if 'gap' in v.get('error', ''))} in chip gaps")
        for v in meas:
            off = math.hypot(v["peak"]["dx"], v["peak"]["dy"]) if v.get("peak") else 9
            if v["snr"] >= 5 and off <= max(1.0, 2 * v["sigma_pred"]):
                fr = next(x for x in D_ if x["key"] == v["frame"]); ok, why = static_ok(fr["pred_ra"], fr["pred_dec"])
                notes.append(f"DECam {v['prop']} {day(v['mjd'])} {v['band']} {int(v['exp'])} s: S/N {v['snr']:.1f}, peak {off:.2f}\" from the prediction (1-sigma {v['sigma_pred']:.2f}\"); {why}")
                if ok:
                    mp_ = None
                    if HH.get(i) is not None and v.get("mag"):
                        rh, dl = vpred(b["arc"], v["mjd"]); mp_ = HH[i] + 5 * math.log10(rh[0] * dl[0]) - OFF.get(str(v["band"])[0], 0.3)
                    # brightness at the peak, not at the predicted place: a peak 0.5-1" off loses most of a matched filter's flux there
                    mpk = (v["zp"] - 2.5 * math.log10(v["peak"]["snr_w"] * v["err_weight"])) if (v.get("zp") and v.get("peak") and v.get("err_weight") and v["peak"]["snr_w"] > 0) else v["mag"]
                    mag_ok = None if mp_ is None else abs(mpk - mp_) <= 1.0
                    nc, once = counts(v["snr"], v.get("null_frac_above"), 300, mag_ok)
                    if nc and mag_ok is not False:
                        det.append(("DECam " + str(v["prop"]), int(v["mjd"] - .5), v["band"], v["snr"], once))
        stack_nights("DECam other programs", [(int(v["mjd"] - .5), v["snr"], v["sigma_pred"], next(x for x in D_ if x["key"] == v["frame"])["pred_ra"], next(x for x in D_ if x["key"] == v["frame"])["pred_dec"], v["mjd"], None)
                                              for v in meas], 0.7, det, notes, miss, tried)
    else:
        tried.append("DECam of other programs: every exposure on the path is already in our runs' lists")
    # CFHT
    C_ = cc.get(i, []); broad = [x for x in C_ if x["band"] not in ("CaHK", "u")]
    tag = tags.get(i); res = None
    if tag and os.path.exists(f"{C}/dbt_confirm/{tag}_cfht_out/result.json"):
        res = json.load(open(f"{C}/dbt_confirm/{tag}_cfht_out/result.json")).get("Z") or {}
    if not C_:
        tried.append("CFHT MegaCam (CADC): no frame on the path")
    elif not broad:
        tried.append(f"CFHT MegaCam (CADC): {len(C_)} frames on the path, all Ca H&K or u (too shallow), not measured")
    elif res is not None and not res.get("nights"):
        tried.append(f"CFHT MegaCam (CADC): {len(broad)} broad-band frames on the path, fetched (NERSC, tag {tag}), but the astrometric tie to Gaia failed on every CCD (too few stars measured); not measured")
    elif res:
        tried.append(f"CFHT MegaCam (CADC): {len(broad)} broad-band frames measured (NERSC, tag {tag})")
        for n in res.get("nights", []):
            near = sorted(broad, key=lambda x: abs(x["mjd"] - n["night"] - .5)); off = math.hypot(*n["off"]); sig = near[0]["sigma"] if near else 9
            st = res.get("static", {}).get(str(n["night"])); stat_txt = "no other night to test" if not st else ("same sky other nights clear" if st["flux"] < 0.5 * n["flux"] else "same sky other nights shows a source")
            stat_ok = not st or st["flux"] < 0.5 * n["flux"]
            mp = None
            if HH.get(i) is not None and near:
                rh, dl = vpred(b["arc"], near[0]["mjd"]); mp = HH[i] + 5 * math.log10(rh[0] * dl[0]) - OFF.get(cfht_band(n["band"]), 0.3)
            if n["snr"] >= 5 and off <= max(1.0, 2 * sig):
                ok, why = static_ok(near[0]["pred_ra"], near[0]["pred_dec"])
                bright = mp is None or (n.get("mag") is not None and abs(n["mag"] - mp) <= 1.0)
                notes.append(f"CFHT {day(n['night'] + .5)} {n['band']} ({n['n']} frames): S/N {n['snr']:.1f}, {100 * n['frac']:.1f}% of random places as high, peak {off:.2f}\" from the prediction; "
                             f"mag {n.get('mag') or 0:.2f}" + (f" (expected {mp:.1f} from H)" if mp else "") + f"; {stat_txt}; {why}")
                nc, once = counts(n["snr"], n["frac"], n.get("nnull") or 820, None if (mp is None or n.get("mag") is None) else bright)
                if ok and stat_ok and nc and bright:
                    det.append(("CFHT MegaCam", n["night"], n["band"], n["snr"], once))
            elif mp is not None and sig <= 0.75 and n["snr"] < 2 and n.get("peak", 9) < 4 and n.get("err"):
                exp_snr = 10 ** (-0.4 * (mp - 25)) / n["err"]
                # 2026-11-22: the formal S/N only counts where the image behaves; scale by the robust spread of S/N at the 820 null places
                # around the target (C###'s 2021-05-10 gri night: null spread 11.6, median -3.0, so an expected S/N 12 had no power)
                ns_ = max(1.0, float(n.get("null_sig") or 1.0)); exp_snr = exp_snr / ns_
                if exp_snr < 10 or (n["snr"] - float(n.get("null_med") or 0)) / ns_ >= 2:
                    tried.append(f"CFHT {day(n['night'] + .5)} {n['band']}: no power (null places' S/N spread {ns_:.1f}, median {float(n.get('null_med') or 0):.1f})" if ns_ > 1.5 else "")
                    tried[:] = [x for x in tried if x]
                elif exp_snr >= 10 and sig > 0.5:  # the 1.5" search holds the body ~90 % of the time only
                    soft.append(f"CFHT {day(n['night'] + .5)} {n['band']} ({n['n']} x {int(n['exp'])} s): expected S/N {exp_snr:.0f} at mag {mp:.1f} (H {HH[i]}), measured {n['snr']:.1f}; "
                                f"prediction 1-sigma {sig:.2f}\", so the 1.5\" search holds the body only about {100 * (1 - math.exp(-1.125 / sig ** 2)):.0f} % of the time")
                elif exp_snr >= 10:
                    miss.append(f"CFHT {day(n['night'] + .5)} {n['band']} ({n['n']} x {int(n['exp'])} s): expected S/N {exp_snr:.0f} at mag {mp:.1f} (H {HH[i]}), measured {n['snr']:.1f}, nothing within the search; prediction 1-sigma {sig:.2f}\"")
    else:
        tried.append(f"CFHT MegaCam (CADC): {len(broad)} broad-band frames on the path, measurement pending (tag {tag})")
    # ESO
    E_ = ce.get(i, []); P3 = [(k, v) for k, v in ep.items() if k.startswith(i + "|")]
    if not E_:
        tried.append("ESO (OmegaCAM, VIRCAM, WFI, FORS, VIMOS, HAWK-I, MUSE): no frame on the path")
    else:
        sets = sorted({(x["inst"], x["band"], day(x["mjd"])[:7]) for x in E_})
        npr = sum(len(v.get("products", [])) for _, v in P3)
        tried.append(f"ESO: {len(E_)} raw frames on the path ({'; '.join(f'{a} {b} {c}' for a, b, c in sets[:8])}{' ...' if len(sets) > 8 else ''}); "
                     f"reduced (Phase 3) images at the place and time: {npr}; raw frames without a reduced product not measured")
        for k, v in P3:
            for p in v.get("products", []):
                if p.get("snr") is not None and p["snr"] >= 5 and p.get("peak") and math.hypot(p["peak"][1], p["peak"][2]) <= max(1.0, 2 * v["sigma"]) and p["peak"][0] < 3 * p["snr"] and counts(p["snr"], p.get("null_frac_above"), 300, None)[0]:
                    ok, why = static_ok(*p["pred"])
                    notes.append(f"ESO {p['collection']} {p['filter']} {day(v['night'] + .5)} (coadd over {p['span_h']} h): S/N {p['snr']:.1f}, peak {math.hypot(p['peak'][1], p['peak'][2]):.2f}\"; {why}")
                    if ok:
                        det.append(("ESO " + p["collection"], v["night"], p["filter"], p["snr"], counts(p["snr"], p.get("null_frac_above"), 300, None)[1]))
    # Subaru
    S_ = cs.get(i, []); hsc = [x for x in S_ if x["inst"] == "HSC"]; sup = [x for x in S_ if x["inst"] == "SUP"]; ccd = [x for x in hs.get(i, []) if x.get("frame")]
    hm = sorted([v for k, v in hp.items() if k.startswith(i + "|") and "snr_blank" in v], key=lambda v: v["mjd"])
    unm = [x for x in ccd if f"{i}|{x['frame']}" not in hp]
    pend = [x for x in unm if not any(os.path.exists(f"{HSC_CAT.replace('P03_1117', s_)}/{x['frame']}.npz") for s_ in HSC_SETS)]
    hsc_fail = [x for x in unm if x not in pend]
    if hsc:
        tried.append(f"Subaru HSC (SMOKA): {len(hsc)} shots on the path ({', '.join(sorted({x['band'] + ' ' + day(x['mjd'])[:4] for x in hsc}))}); {len(ccd)} CCD frames hold the place; "
                     f"{len(hm)} measured (raw frames of SMOKA requests (ids in SMOKA_REQUESTS.md and in coverage_all.json), cand9 reduction, Gaia DR3 astrometry, forced aperture photometry)"
                     + (f"; {len(hsc_fail)} not measurable (no astrometric solution, or the place off the CCD's edge)" if hsc_fail else "")
                     + (f"; {len(pend)} not yet measured (requested from SMOKA)" if pend else ""))
        for v in hm:
            off = math.hypot(*v["recentre"]) if v.get("recentre") else 9; sg = v["sigma"]; zp = hsc_zp(v["frame"]); mp = None
            if HH.get(i) is not None:
                rh, dl = vpred(b["arc"], v["mjd"]); mp = HH[i] + 5 * math.log10(rh[0] * dl[0]) - 0.3  # V -> r (zero point from r or G mags of the CCD's stars; rough, +-0.5 in other bands)
            rband = str(v.get("band", "")).startswith("HSC-r")  # zero points come from r magnitudes: other bands uncalibrated (no brightness gate there)
            mag = zp - 2.5 * math.log10(v["flux"]) if (zp and v.get("flux", 0) > 0 and rband) else None
            if sg <= 3 and v["snr_blank"] >= 5 and v.get("recentre_ok") and off <= max(1.0, 2 * sg):
                ok, why = static_ok(v["pred_ra"], v["pred_dec"]); bright = mp is None or mag is None or abs(mag - mp) <= 1.0
                notes.append(f"HSC {v['frame']} {day(v['mjd'])} {v['band']} {v['exp']:.0f} s: S/N {v['snr_blank']:.1f} against 500 blank apertures ({100 * v['frac_blank_above']:.1f}% as high), "
                             f"recentred {off:.2f}\" from the prediction (1-sigma {sg:.2f}\")" + (f", r {mag:.1f} (expected {mp:.1f})" if mag and mp else ", magnitude not calibrated in this band" if not rband else "") + f"; {why}")
                nc, once = counts(v["snr_blank"], v["frac_blank_above"], 500, None if (mp is None or mag is None) else bright)
                if ok and bright and nc:
                    det.append(("Subaru HSC", int(v["mjd"]), v["band"], v["snr_blank"], once))  # Mauna Kea nights do not cross 0 UT (as cfht_phot.night)
            elif rband and sg <= 0.3 and v["snr_blank"] < 2 and zp and mp is not None and v.get("sigma_blank"):  # the aperture (0.84") must hold the body: 1-sigma <= 0.3"
                exp_snr = 10 ** (-0.4 * (mp - zp)) / v["sigma_blank"]
                if exp_snr >= 10:
                    miss.append(f"HSC {v['frame']} {day(v['mjd'])} {v['band']} {v['exp']:.0f} s: expected S/N {exp_snr:.0f} at r {mp:.1f} (H {HH[i]}), measured {v['snr_blank']:.1f}; prediction 1-sigma {sg:.2f}\"")
        stack_nights("Subaru HSC", [(int(v["mjd"]), v["snr_blank"], v["sigma"], v.get("pred_ra"), v.get("pred_dec"), v["mjd"],
                                     (10 ** (-0.4 * (vmag(b, v["mjd"]) - hsc_zp(v["frame"]))) / v["sigma_blank"]) if (str(v.get("band", "")).startswith("HSC-r") and HH.get(i) is not None and hsc_zp(v["frame"]) and v.get("sigma_blank")) else None)
                                    for v in hm if v.get("snr_blank") is not None], 0.5, det, notes, miss, tried)
    else:
        tried.append("Subaru HSC (SMOKA): no shot on the path")
    tried.append("Subaru Suprime-Cam (SMOKA): " + (f"{len(sup)} shots on the path" if sup else "no shot on the path"))
    # Rubin
    R_ = cr.get(i, []); rr = f"{H}/rubin_out/{i}.json"; rub = json.load(open(rr)) if os.path.exists(rr) else None; rubin_rights = False
    if not R_:
        tried.append("Rubin (<rubin-science-platform>, DP1 and DP2 Visit tables, 2024-11 to 2026-01): no visit on the path")
    else:
        tried.append(f"Rubin DP2 (<rubin-science-platform> catalogues): {len(R_)} visits on the path")
        if rub:
            for n in rub["nights"]:
                if n.get("dia_hits", 0) >= 2 and n["dia_hits"] > 3 * max(n.get("null_per_offset") or [0]):
                    det.append(("Rubin DP2", n["night"], "g/r/i", None, True)); rubin_rights = True
                elif n.get("dia_hits", 0) == 0 and n.get("src_hits", 0) >= 1 and max(n.get("src_null_per_offset") or [0]) == 0 and \
                        any(p["src"] and p["sigma"] <= 3 and p["band"] != "u" and p["src"][0]["sep"] <= 1.0 for p in n["per"]):
                    # no difference imaging for these visits: a direct-image Source at the prediction (within max(1", 2 sigma)), if nothing fixed is there
                    hit = min((p for p in n["per"] if p["src"] and p["sigma"] <= 3 and p["band"] != "u" and p["src"][0]["sep"] <= 1.0), key=lambda p: p["src"][0]["sep"])
                    fr = next(x for x in R_ if x["key"].endswith(str(hit["visit"])))
                    ok, why = static_ok(fr["pred_ra"], fr["pred_dec"])
                    notes.append(f"Rubin DP2 visit {hit['visit']} ({hit['band']}): dp2.Source {hit['src'][0]['sep']}\" from the prediction (1-sigma {hit['sigma']}\"), no difference imaging for it; {why}; data under our data rights")
                    if ok:
                        det.append(("Rubin DP2", n["night"], hit["band"], None, False)); rubin_rights = True
            nerr = sum(1 for n in rub["nights"] if n.get("error"))
            tried[-1] += (f"; {nerr} of {len(rub['nights'])} nights failed to query" if nerr else "") + \
                (f"; no DiaSource at the prediction on any of {len(rub['nights']) - nerr} nights (offset places hold {rub.get('null_mean', 0):.2f} each)" if not rub.get("dia_hits") and len(rub["nights"]) > nerr else "")
            if rub.get("dia_hits"):
                notes.append(f"Rubin DP2: {rub['dia_hits']} DiaSource detections at the prediction over {len(rub['nights'])} nights (null {rub['null_mean']:.1f} per offset place); data under our data rights")
    # ITF
    same = [t for t in ij.get(i, []) if t["verdict"] == "SAME BODY"]; cons = [t for t in ij.get(i, []) if t["verdict"].startswith("CONSISTENT")]
    tried.append("MPC ITF along the whole path: " + ("; ".join(f"{t['station']} {t['trksub']} ({t['n']} obs, {day(t['first'])}): {t['verdict'].lower()}, rms {t['rms']}\"" for t in same + cons) or "no tracklet that fits"))
    ns = NSC.get(i)  # NSC DR2 along the whole path (2012-2019), any 1-sigma up to 600"
    if ns:
        ex = ns["expected_snr"]; nx = [x for x in ex if not x[4]]; nx5 = [x for x in nx if x[3] >= 5]
        tried.append(f"NSC DR2 (Astro Data Lab; DECam, Mayall Mosaic3, Bok 90Prime, 2012-09 to 2019-11; single-epoch objects along the whole path within max(2\", 3 sigma), "
                     f"then their measurements; identity by joint orbit fit): {sum(ns['exposures'].values())} exposures on the path ({', '.join(f'{INST[k]} {v}' for k, v in ns['exposures'].items() if v)}), "
                     f"{len(nx)} not in our runs' lists, {len(nx5)} of them with expected S/N >= 5 (best {max([x[3] for x in nx] or [0]):.0f}); {ns['n_candidates']} candidate detections"
                     + (f"; nights tested: " + ", ".join(f"{INST.get(t['inst'], t['inst'])} {day(t['night'] + .5)} ({t['n']}): {t['verdict'].lower()}" for t in ns["tests"][:6]) if ns["tests"] else ""))
        jn = ns.get("joint")
        for t in ns["tests"]:
            if t["verdict"] == "SAME BODY" and (t["n"] >= 2 or (jn and jn["verdict"] == "SAME BODY")):
                det.append((f"NSC DR2 {INST.get(t['inst'], t['inst'])}", t["night"], "/".join(t["filters"]), None, t["n"] >= 2))
                notes.append(f"NSC DR2 {INST.get(t['inst'], t['inst'])} {day(t['night'] + .5)}: {t['n']} measurement(s) in {', '.join(t['exposures'][:3])}, one orbit with the arc (rms {t['rms']}\", prior chi2 {t['prior_chi2']}, shifted copies fail)")
        if jn:
            notes.append(f"NSC DR2 joint fit of the {jn['nights']} passing nights: {jn['verdict'].lower()} (rms {jn['rms']}\", prior chi2 {jn['prior_chi2']})"
                         + (f"; shifted copies that also fit: " + ", ".join(f"{x['shift']}\" at {x['angle']} deg (rms {x['rms']}\", prior chi2 {x['prior_chi2']})" for x in jn["shifts_that_fit"]) if jn.get("shifts_that_fit") else ""))
            sf = jn.get("shifts_that_fit") or []
            if jn["verdict"].startswith("CONSISTENT") and jn["nights"] >= 3 and sf and all(x["shift"] == 5 and x["prior_chi2"] >= jn["prior_chi2"] + 10 for x in sf):
                likely.append(f"LIKELY SEEN in NSC DR2 on {jn['nights']} nights ({', '.join(day(t['night'] + .5) for t in ns['tests'] if t['verdict'] != 'NOT THE SAME BODY')}): one orbit with the arc "
                              f"(rms {jn['rms']}\", prior chi2 {jn['prior_chi2']}); the power test is borderline: {len(sf)} of 12 shifted copies (5\") also fit, but worse (prior chi2 "
                              f"{', '.join(str(x['prior_chi2']) for x in sf)}); check by hand")
    npj = NSCP.get(i)  # NSC DR2 rate-matched pairs where the place is uncertain by > 30"
    if npj and (npj["exposures_on_path"] or npj["n_pairs"]):
        jn = npj.get("joint")
        tried.append(f"NSC DR2 where the prediction is uncertain by more than 30\" (same-night detections that move at the predicted rate, fitted with the arc as a prior, "
                     f"joined across nights): {npj['exposures_on_path']} exposures not in our runs' lists, {npj['nights_two_plus']} nights with two or more, {npj['n_pairs']} rate-matched pairs, "
                     f"{len(npj['passing'])} fit the arc; joint over nights: " + (f"{jn['verdict'].lower()} ({len(jn['nights'])} nights, rms {jn['rms']}\", prior chi2 {jn['prior_chi2']})" if jn else "none"))
        if jn and jn["verdict"] == "SAME BODY":
            for inst_, n_ in jn["nights"]:
                det.append(("NSC DR2 (rate-matched)", n_, "", None, True))
            notes.append(f"NSC DR2 rate-matched pairs on {len(jn['nights'])} nights ({', '.join(day(n_ + .5) for _, n_ in jn['nights'])}) fit one orbit with the arc; check by hand before using these places")
    for a_, W_ in WIDE.items():  # wide windows (1-sigma 1-7"), pixel level
        wv = W_.get(i)
        if not wv:
            continue
        fr_ = [r for r in wv["frames"] if "error" not in r]; es = sorted([r["expected_snr"] for r in fr_ if r.get("expected_snr") is not None])
        tried.append(f"{'DECam of other programs' if a_ == 'decam' else 'ESO Phase 3 (KiDS, VIKING, ...)'}, wide windows (prediction 1-sigma 1-7\", search within max(2\", 3 sigma), matched filter, "
                     f"every DR10-clean peak with S/N >= 5, identity by joint orbit fit): {len(wv['frames'])} frames, {len(fr_)} measured, expected S/N {es[0] if es else '?'}-{es[-1] if es else '?'}; "
                     f"{sum(len(r.get('candidates', [])) for r in fr_)} candidate peaks; joint {wv['joint']['verdict'].lower() if wv.get('joint') else 'not possible (fewer than 2 nights pass)'}")
        if wv.get("joint") and wv["joint"]["verdict"] == "SAME BODY":
            for t in wv["tests"]:
                if t and t["verdict"] in ("SAME BODY", "CONSISTENT, UNTESTABLE"):
                    det.append((("DECam wide window" if a_ == "decam" else "ESO wide window"), t["night"], "", t["snr"], False))
                    notes.append(f"{a_} wide window {day(t['night'] + .5)}: peak S/N {t['snr']} at {t['ra']:.6f} {t['dec']:+.6f}, on the joint orbit")
    nov = json.load(open(f"{NOV}/{i}.json")) if os.path.exists(f"{NOV}/{i}.json") else None; known = None; known_src = None
    if nov:
        tried.append(f"Novelty check against the MPC files of {nov.get('mpc_files')} (MPCORB at our nights and, for arcs over 300 d, along the carried orbit from 2024; ITF at our nights; "
                     f"identity by a joint orbit fit, shifted copies must fail): {nov.get('verdict')}")
        for tk in [x.split()[-1] for x in str(nov.get("verdict", "")).split(":", 1)[-1].split(",") if "ITF" in x]:  # CHECK BY HAND: ITF <trkSub>
            jt = [t for t in ij.get(i, []) if t["trksub"] == tk]
            tried[-1] += (f"; ITF {tk} resolved by the joint fit: {jt[0]['verdict'].lower()} (rms {jt[0]['rms']}\", prior chi2 {jt[0]['prior_chi2']})" if jt else f"; ITF {tk}: no joint fit")
        syn = f"{C}/novelty_queue/done/{i}.result.json"  # sync_novelty.py's verdict (joint fit passes, borderline only for an outlier among the body's own MPC observations -> KNOWN)
        syn = json.load(open(syn)).get("verdict", "") if os.path.exists(syn) else ""
        if str(nov.get("verdict", "")).startswith(("KNOWN", "OURS")) and not str(nov.get("verdict", "")).startswith("KNOWN ITF"):
            known = nov["verdict"]; known_src = "novelty"
        elif str(syn).startswith("KNOWN") and not str(syn).startswith("KNOWN ITF"):  # "KNOWN ITF <trk>" (resolve_hand) is an undesignated tracklet: other observers' category
            known = syn; known_src = "novelty"
        elif str(syn).startswith(("UNSURE", "KNOWN ITF")):
            notes.append("novelty (resolve_hand): " + str(syn)[:200])
        elif str(nov.get("verdict", "")).startswith("PROBABLY"):
            notes.append("novelty: " + nov["verdict"])
    else:
        tried.append("Novelty check of 2026-11-19: not run (no spec)")
    sk = SKY.get(i)
    if sk:
        w30 = sk.get("within_30") or []; nn = {}
        for o in sk["nights"]:
            for h_ in o.get("hits", []):
                if h_["sep"] < 30:
                    nn.setdefault(h_["name"], [h_, 0])[1] += 1
        def why(h_, k):
            if k >= 2 or "TNO" in h_["cls"] or "KBO" in h_["cls"] or "Centaur" in h_["cls"]:
                return "CHECK: on several nights or a distant class"
            return "a passer-by on one night, not this body"
        tried.append(f"SkyBoT (IMCCE, {sk['checked'][:10]}) at {sk['n_answered']} of {sk['n_nights']} detection nights, 120\" cones: "
                     + ("known objects within 30\": " + "; ".join(f"{n_} ({v[0]['cls']}, V {v[0]['V']}) {v[0]['sep']}\" on {v[1]} of {sk['n_nights']} nights: {why(v[0], v[1])}" for n_, v in list(nn.items())[:4])
                        if w30 else "no known object within 30\""))
        for n_, v in nn.items():  # one named object at our place on most nights: a known body (SkyBoT is the MPC's orbits, propagated)
            seps = [h_["sep"] for o in sk["nights"] for h_ in o.get("hits", []) if h_["name"] == n_]
            if known:
                break
            if v[1] >= 2 and float(np.median(seps)) <= 2.0:
                known_src = "skybot"; known = (f"{n_} ({v[0]['cls']}, V {v[0]['V']}): SkyBoT puts it {min(seps)}-{max(seps)}\" from our detections on {v[1]} of {sk['n_answered']} nights"
                         + ("; the other nights do not match, so the arc may join this known body to other detections" if v[1] < sk["n_answered"] else ""))
            elif v[1] >= 3 and v[1] >= 0.8 * sk["n_answered"] and max(seps) - min(seps) <= 1.0 and float(np.median(seps)) <= 10:
                known_src = "skybot"; known = (f"probably {n_} ({v[0]['cls']}, V {v[0]['V']}): SkyBoT puts it at a steady {min(seps)}-{max(seps)}\" from our detections on {v[1]} of {sk['n_answered']} nights, "
                         "the error of its catalogue orbit; one orbit through both sets of positions would settle it")
        if any(why(v[0], v[1]).startswith("CHECK") for v in nn.values()):
            notes.append("SkyBoT: a known object near our place on several nights or of a distant class; see archives_tried")
    vmin = None
    if HH.get(i) is not None:
        gm = np.arange(55200., 61400., 60.); rh_, dl_ = vpred(b["arc"], gm); vv = HH[i] + 5 * np.log10(rh_ * dl_); vmin = float(vv.min())
    if vmin is None:
        tried.append("Pan-STARRS1 (single epochs r ~21.8) and SDSS Stripe 82 (r ~22.2): not queried, no H for this body")
    elif vmin > 21.8:
        tried.append(f"Pan-STARRS1 (single epochs, r ~21.8) and SDSS Stripe 82 (r ~22.2): too shallow, the body is V >= {vmin:.1f} at every epoch 2010-2026 (from H {HH[i]}); "
                     "DES DR2 is a coadd catalogue (moving bodies drop out); its single epochs are in NSC DR2 above; Legacy Survey DR10/DR9: used as the fixed-source check")
    else:
        pp = PS1.get(i)
        if pp:
            ok_ = [t for t in pp["tests"] if t["verdict"] in ("SAME BODY", "CONSISTENT, UNTESTABLE")]
            tried.append(f"Pan-STARRS1 DR2 single-epoch detections (MAST; 2009-06 to 2015-12, at {pp['epochs_V_le_22_3']} four-day steps with V <= 22.3 and 1-sigma <= 300\"; "
                         f"within max(2\", 3 sigma), within 1.5 mag; identity by joint orbit fit): {pp['queries']} cone queries ({pp['errors']} failed), {len(pp['hits'])} detections near the path on "
                         f"{len(pp['tests'])} nights; " + (", ".join(f"{day(t['night'] + .5)} ({t['n']}): {t['verdict'].lower()}" for t in ok_) if ok_ else "none on the orbit (every night's detections fail the joint fit)")
                         + f"; SDSS Stripe 82: not queried (V >= {vmin:.1f})")
            for t in ok_:
                if t["verdict"] == "SAME BODY" and t["n"] >= 2:
                    det.append(("Pan-STARRS1", t["night"], "/".join(t["bands"]), None, True))
        else:
            tried.append(f"Pan-STARRS1 / SDSS: the body reaches V {vmin:.1f} (from H {HH[i]}); not yet queried")
    best = {}
    for a, n, bd, sn, fr in det:
        best[(a, n)] = bool(best.get((a, n))) or bool(fr)
    trk_days = {(t["station"], int(t["first"] - .5)) for t in same}
    # the same exposures as an ITF tracklet are one data set, not two
    nights = sorted(k for k in best if not (k[0].startswith("DECam") and ("W84", k[1]) in trk_days) and not (k[0].startswith("Rubin") and any(s_ == "X05" for s_, _ in trk_days)))
    nn = len({n for a, n in nights}); arch = sorted({a for a, n in nights}); nonrubin = [a for a in arch if not a.startswith("Rubin")]
    head = (("KNOWN BODY by SkyBoT at our detection nights: " if known_src == "skybot" else f"KNOWN BODY by the novelty check of {(nov or {}).get('mpc_files') or '2026-11-19'}: ") + known + "; ") if known else ""
    if same:
        rub = [t for t in same if t["station"] == "X05"]
        head += ("REPORTED BY RUBIN to the MPC (ITF " + ", ".join(f"X05 {t['trksub']}, {t['n']} obs {day(t['first'])}..{day(t['last'])}" for t in rub) + "), NOT NEW; ") if rub else ""
        oth = [t for t in same if t["station"] != "X05"]
        if oth:
            head += "OTHER OBSERVERS' UNLINKED TRACKLETS AT THE MPC, same body by a joint orbit fit (" + ", ".join(f"{t['station']} {t['trksub']} {day(t['first'])}" for t in oth) + "): not designated, but not ours alone; "
    if known and not (same or nn >= 2):
        verdict = head.rstrip("; ")
    elif same or nn >= 2:
        parts = []
        if same:
            parts.append("in those tracklets")
        if nonrubin and (nn >= 2 or same):
            parts.append("in " + ", ".join(nonrubin) + f" ({len({n for a, n in nights if not a.startswith('Rubin')})} nights)")
        if rubin_rights and any(a.startswith("Rubin") for a, n in nights):
            parts.append(f"in Rubin DP2 ({len({n for a, n in nights if a.startswith('Rubin')})} nights; data under our data rights, not for OBSERVER_NAME or shared pages without approval)")
        verdict = head + "SEEN " + "; ".join(parts)
    elif nn == 1 and best[nights[0]]:
        verdict = f"SEEN ONCE in {arch[0]} ({day(nights[0][1] + .5)}); one night, a second epoch would settle it" + (" (Rubin: data under our data rights)" if rubin_rights else "")
    elif likely:
        verdict = likely[0]
    elif miss:
        verdict = "NOT SEEN: " + "; ".join(miss[:2])
    elif soft:
        verdict = "PROBABLY NOT SEEN: " + "; ".join(soft[:2])
    else:
        verdict = "NO DECISIVE DATA in any archive checked" + (" (one night at the prediction, but random places reach it too often)" if nn == 1 else "") + (" (HSC pending)" if pend else "")
    if cons and not same:
        verdict += "; consistent ITF tracklet " + ", ".join(f"{t['station']} {t['trksub']}" for t in cons) + " (untestable)"
    if b.get("refit"):
        notes.append("orbit: the refit arc without the detections on fixed sources (" + os.path.basename(b["arc"]) + ")")
    f = f"{D}/runs/{b['run']}/confirm/{i}.json"; os.makedirs(os.path.dirname(f), exist_ok=True)
    old = json.load(open(f)) if os.path.exists(f) else None; e1117 = None
    if old and str(old.get("by", "")).startswith("all-archive"):
        if str(old.get("checked", "")).startswith("2026-11-17"):
            e1117 = {k: old.get(k) for k in ("verdict", "checked", "notes", "detections", "archives")}
        else:
            e1117 = old.get("earlier_20261117")
        old = old.get("earlier")
    if old:
        notes.append(f"earlier ({old.get('checked', '?')}): {old.get('verdict')}")
        if str(old.get("verdict", "")).startswith("SEEN"):
            ov = old["verdict"].split(":")[0].split(";")[0]; rest = verdict[len(head):]
            verdict = head + ov + " (earlier check); this check: " + (rest if "SEEN" in rest else "nothing decisive beyond it" + (" (HSC pending)" if pend else ""))
    if miss and not verdict.startswith("NOT SEEN"):
        notes += ["not seen where expected: " + m for m in miss]
    rec = dict(id=i, C=CNUM.get(i), verdict=verdict, notes=" | ".join(notes), archives=tried, archives_tried=tried, detections=[dict(archive=a, night=day(n + .5), band=bd, snr=s_, single_night_grade=bool(fr)) for a, n, bd, s_, fr in det],
               mpc_itf_same=[dict(station=t["station"], trksub=t["trksub"], n=t["n"], first=day(t["first"]), rms=t["rms"]) for t in same],
               itf=same + cons, rubin_rights=rubin_rights, checked=now + " PST", by="all-archive check, TRKSUB01 session (planet-search/rsi/candidates/allarch/verdicts.py)",
               files="planet-search/rsi/candidates/allarch/", label=b.get("label"), priority=b["pri"])
    if old:
        rec["earlier"] = old
    if e1117:
        rec["earlier_20261117"] = e1117
    json.dump(rec, open(f, "w"), indent=1)
    lines.append(f"| {now} | {CNUM.get(i, '')} | {i} | {verdict} | ../allarch/ |")
    summary.append(f"| P{b['pri']} | {CNUM.get(i, '')} | {b.get('label') or ''} | {i} | {verdict} |")
    print(f"P{b['pri']} {(b.get('label') or ''):7s} {i[4:52]:48s} {verdict[:150]}", flush=True)
RP = f"{C}/confirm_queue/RESULTS.md"; done_ids = {l.split(" | ")[2] for l in lines}; today = "| " + now[:10]
keep = [l for l in open(RP).read().splitlines() if not (l.endswith("| ../allarch/ |") and l.startswith(today) and any(x in done_ids for x in l.split(" | ")[1:3]))]
open(RP, "w").write("\n".join(keep + lines) + "\n")
open(H + "/VERDICTS.md", "w").write(f"# All-archive verdicts ({now} PST)\n\n| pri | C | label | id | verdict |\n|---|---|---|---|---|\n" + "\n".join(summary) + "\n")
