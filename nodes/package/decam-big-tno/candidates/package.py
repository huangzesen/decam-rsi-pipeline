#!/usr/bin/env python3
"""Private Google Drive copy of the candidate database, one folder per object, holding the files we would send.

2026-11-19.

Builds candidates/_drive/ (gitignored: it holds Rubin positions) and copies it with rsync --checksum into
  My Drive/DECam TNO candidates/   (Drive for Desktop folder; the user's private Drive, never shared from here)
Top level: catalogue.html (mirror of the Promising New Bodies artifact, progress/objects/index.html), index.csv,
README.md. Per object "<CID>_<label or run-region-rank>/":
  README.md        the database README of the body plus what this folder holds
  positions.psv    ADES PSV of every position: our DECam detections of the arc, and every other-archive row in
                   planet-search/rsi/candidates/allarch/export_*.json, Rubin DP2 included (notes column names the source)
  positions.json   the same rows as JSON
  orbit.json       one two-body fit through all rows (rows over max(10", 5 sigma) from the arc's prediction left out and
                   listed), elements, and places on 2026-11-20 / 12-15 / 2027-01-15 with 1-sigma, V and elongation
  card.json, evidence/   copies from the database folder
  sent/            each email to OBSERVER_NAME that named the body (body.txt and the PSV attached)
Planet-search bodies (C###-C###) get their database files and the MPC report PSVs of their label.
Dropped bodies go under _dropped/. Our detection positions are cached in candidates/_drive_cache/<lead id>.json.
Usage: DBT_SITE=nersc ~/anaconda3/bin/python -I candidates/package.py [--no-sync]"""
import csv, glob, json, os, re, shutil, subprocess, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
HERE = "$ALE_HOME/rsi/decam-big-tno"
PS = os.path.join(HERE, "../../planet-search/rsi/candidates")
sys.path.insert(0, os.path.join(HERE, "harness"))
import lead_check as LC, arc_extend as AX, observer_batch as QB   # noqa: E402
from lead_check import O, radec_to_unit, unit_to_radec            # noqa: E402
CD = os.path.join(HERE, "candidates"); OUT = os.path.join(CD, "_drive"); CACHE = os.path.join(CD, "_drive_cache")
DRIVE = os.path.expanduser("$DRIVE_FOLDER/DECam TNO candidates")
DATES = ["2026-11-20", "2026-12-15", "2027-01-15"]
S = O.SITES
S.setdefault("T09", EarthLocation.from_geodetic(-155.47610 * u.deg, 19.82552 * u.deg, 4163 * u.m))  # Subaru
S.setdefault("T14", EarthLocation.from_geodetic(-155.46833 * u.deg, 19.82528 * u.deg, 4204 * u.m))  # CFHT
S.setdefault("X05", EarthLocation.from_geodetic(-70.74942 * u.deg, -30.24463 * u.deg, 2647 * u.m))  # Rubin
S.setdefault("309", EarthLocation.from_geodetic(-70.40300 * u.deg, -24.62700 * u.deg, 2635 * u.m))  # VST (KiDS)
S.setdefault("V00", EarthLocation.from_geodetic(-111.60000 * u.deg, 31.96300 * u.deg, 2120 * u.m))  # Bok 2.3 m, Kitt Peak
S.setdefault("F51", EarthLocation.from_geodetic(-156.25750 * u.deg, 20.70750 * u.deg, 3067 * u.m))  # Pan-STARRS 1
os.makedirs(CACHE, exist_ok=True)


def exports():
    E = {}
    for f in sorted(glob.glob(os.path.join(PS, "allarch", "export_*.json"))):
        try:
            B = json.load(open(f))["bodies"]
        except Exception:
            continue
        for k, v in B.items():
            e = E.setdefault(k, {"rows": [], "joint_check": None, "from": []}); seen = {(r.get("frame"), round(r["mjd_utc"], 5)) for r in e["rows"]}
            new = [r for r in v.get("rows", []) if (r.get("frame"), round(r["mjd_utc"], 5)) not in seen]
            e["rows"] += new; e["joint_check"] = v.get("joint_check") or e["joint_check"]
            if new:
                e["from"].append(os.path.basename(f))
    return E


def our_rows(cards):
    """[mjd, ra, dec, sigma, mag, band, snr, exptime] of the arc's DECam detections, cached per lead id."""
    need = {}
    for c in cards:
        i = c["identities"]["lead_id"]; run = c["identities"]["run"]
        if os.path.exists(os.path.join(CACHE, i + ".json")):
            continue
        A = arc_of(c); need.setdefault(c["identities"]["scenario"], set()).update(A["rows"])
    if need:
        print("fetching", sum(len(v) for v in need.values()), "detections in", len(need), "scenarios from NERSC", flush=True)
        R = AX.remote({sid: {"rows": sorted(int(r) for r in rows)} for sid, rows in need.items()})
        for c in cards:
            i = c["identities"]["lead_id"]; run = c["identities"]["run"]; sid = c["identities"]["scenario"]
            if os.path.exists(os.path.join(CACHE, i + ".json")):
                continue
            E = LC.exposures(sid, os.path.join(HERE, "runs", run, "lead_check")); out = []
            for r in arc_of(c)["rows"]:
                ra, dec, ex, fl, fe, sn = R[sid]["rows"][str(r)]; e = E[int(ex)]; mz = e.get("magzero")
                mag = None if mz is None or not np.isfinite(mz) else float(mz - 2.5 * np.log10(max(fl, 1e-9)))
                out.append([float(e["mjd_mid"]), float(ra), float(dec), float(0.08 + 0.6 / max(sn, 3.0)), mag, e["band"], float(sn), float(e["exptime"])])
            json.dump(out, open(os.path.join(CACHE, i + ".json"), "w"))


REMOTE2 = r'''
import json, sys, numpy as np
S = "%SCEN%"; q = json.load(sys.stdin); out = {}
for sid, rows in q.items():
    C = {k: np.load(f"{S}/{sid}/det_{k}.npy", mmap_mode="r") for k in ("ra", "dec", "exp", "flux", "ferr", "snr", "ccd", "x", "y", "flag", "sharp")}
    o = {}
    for r in rows:
        o[str(r)] = {k: (int(v[r]) if v.dtype.kind in "iub" else float(v[r])) for k, v in C.items()}
    out[sid] = {"rows": o, "region": json.load(open(f"{S}/{sid}/region.json"))}
json.dump(out, sys.stdout)
'''.replace("%SCEN%", LC.SCEN)


def det_detail(cards):
    """Every column of each arc detection (row id, CCD, pixel x/y, flux, flag, sharpness) plus its exposure record and the
    scenario's region, cached per lead id as _drive_cache/<id>.det.json, in the arc's row order."""
    need = {}
    for c in cards:
        if not os.path.exists(os.path.join(CACHE, c["identities"]["lead_id"] + ".det.json")):
            need.setdefault(c["identities"]["scenario"], set()).update(int(r) for r in arc_of(c)["rows"])
    if not need:
        return
    print("fetching detection details:", sum(len(v) for v in need.values()), "rows in", len(need), "scenarios", flush=True)
    assert "'" not in REMOTE2
    pr = subprocess.run(LC.VSSH + [f"{LC.VPY} '{REMOTE2}'"], input=json.dumps({k: sorted(v) for k, v in need.items()}), capture_output=True, text=True, timeout=1800)
    if pr.returncode != 0:
        sys.exit(f"detail read failed: {pr.stderr[-400:]}")
    R = json.loads(pr.stdout)
    for c in cards:
        i = c["identities"]["lead_id"]; sid = c["identities"]["scenario"]; f = os.path.join(CACHE, i + ".det.json")
        if os.path.exists(f) or sid not in R:
            continue
        E = LC.exposures(sid, os.path.join(HERE, "runs", c["identities"]["run"], "lead_check")); out = []
        for r in arc_of(c)["rows"]:
            d = dict(R[sid]["rows"][str(r)]); e = E[d["exp"]]
            out.append(dict(det_row=int(r), exposure_index=d.pop("exp"), exposure=e["stem"], **{k: v for k, v in e.items() if k not in ("stem", "obs_xyz")},
                            ccd=d.pop("ccd"), x_pix=d.pop("x"), y_pix=d.pop("y"), **d))
        json.dump(dict(scenario=sid, region=R[sid]["region"], detections=out), open(f, "w"))


EPH = [Time("2026-11-20T00:00:00", scale="utc").mjd + k for k in range(0, 193)]   # nightly, 2026-11-20 to 2027-05-31


def coverage(c, orbit, footprint=1.1):
    """Every exposure of the body's search scenario (all public DECam 2013-2026 there) whose pointing lies within ~1.1 deg of
    the predicted place, with the predicted place, 1-sigma, V, expected S/N from the exposure's 5-sigma depth, and whether
    our arc holds a detection on it. CCD gaps are not modelled."""
    I = c["identities"]; sid = I["scenario"]; E = LC.exposures(sid, os.path.join(HERE, "runs", I["run"], "lead_check"))
    if sid not in _BARY:
        m = np.array([e["mjd_mid"] for e in E]); _BARY[sid] = O.observer_bary(m, "W84")
    Rb, tt = _BARY[sid]; p = np.array(orbit["p"]); cov = np.array(orbit["cov"]); t0 = orbit["epoch_tdb"]
    uu, s1 = O.predict_sigma(p, cov, t0, tt, Rb); ra, dec = unit_to_radec(uu); ra = np.degrees(ra) % 360; dec = np.degrees(dec)
    r0 = np.radians([e["ra0"] for e in E]); d0 = np.radians([e["dec0"] for e in E])
    sep = np.degrees(np.arccos(np.clip(np.sin(d0) * np.sin(np.radians(dec)) + np.cos(d0) * np.cos(np.radians(dec)) * np.cos(np.radians(ra) - r0), -1, 1)))
    k = np.where(sep < footprint)[0]
    det = {}
    f = os.path.join(CACHE, I["lead_id"] + ".det.json")
    if os.path.exists(f):
        for d in json.load(open(f))["detections"]:
            det.setdefault(d["exposure_index"], []).append(str(d["det_row"]))
    rh, de, eg = QB.geometry(p, t0, [E[j]["mjd_mid"] for j in k]) if len(k) else ([], [], [])
    out = []
    for n, j in enumerate(k):
        e = E[j]; V = None if c.get("H") is None else float(c["H"] + 5 * np.log10(rh[n] * de[n]))
        m5 = e.get("m5"); snr = None if V is None or m5 is None or not np.isfinite(m5) else 5 * 10 ** (-0.4 * (V - m5))
        out.append(dict(exposure_index=int(j), exposure=e["stem"], utc=Time(e["mjd_mid"], format="mjd").isot[:19], mjd_mid=round(e["mjd_mid"], 6), band=e["band"],
                        exptime=e["exptime"], airmass=e.get("airmass"), fwhm=e.get("fwhm"), m5=m5, magzero=e.get("magzero"),
                        pointing_ra=e["ra0"], pointing_dec=e["dec0"], pred_ra=round(float(ra[j]), 6), pred_dec=round(float(dec[j]), 6),
                        sigma_arcsec=round(float(s1[j]), 2), off_pointing_deg=round(float(sep[j]), 3), V_pred=None if V is None else round(V, 2),
                        expected_snr=None if snr is None else round(snr, 1), detected_rows=" ".join(det.get(int(j), []))))
    return out


def ephemeris(c, orbit):
    p = np.array(orbit["p"]); cov = np.array(orbit["cov"]); t0 = orbit["epoch_tdb"]
    ra, dec, s1 = AX.predict(p, cov, t0, np.array(EPH)); rh, de, eg = QB.geometry(p, t0, EPH); H = c.get("H")
    return [dict(utc=Time(m, format="mjd").isot[:10] + " 00:00", ra=round(float(a), 6), dec=round(float(b), 6), ra_hms=QB.ra_hms(a), dec_dms=QB.dec_dms(b),
                 sigma_arcsec=round(float(s), 2), V=None if H is None else round(float(H + 5 * np.log10(r * d)), 2), r_au=round(float(r), 3),
                 delta_au=round(float(d), 3), elongation_deg=round(float(g), 1)) for m, a, b, s, r, d, g in zip(EPH, ra, dec, s1, rh, de, eg)]


def write_csv(path, rows):
    if rows:
        keys = list(rows[0]); [keys.append(k) for r in rows for k in r if k not in keys]
        clean = [{k: ("" if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]   # NaN -> empty
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(clean)


_BARY = {}


def arc_of(c):
    i = c["identities"]["lead_id"]; run = c["identities"]["run"]
    rf = os.path.join(HERE, "runs", run, "arc", i + ".refit.json")
    return json.load(open(rf if os.path.exists(rf) else os.path.join(HERE, "runs", run, "arc", i + ".json")))


def fit_all(c, ours, other, extra=None):
    A = arc_of(c); p0 = np.array(A["p"], float); c0 = np.array(A["cov"], float); t0 = float(A["epoch_tdb"])
    extra = extra or [{}] * len(ours)
    rows = [("W84", m, ra, dec, s, "DECam (our search)", mag, band, sn, ex, X) for (m, ra, dec, s, mag, band, sn, ex), X in zip(ours, extra)]
    left = []
    for y in other:
        site = y.get("site") or ("X05" if "rubin" in y["archive"].lower() else "W84")
        if site not in S:
            left.append(f"{y['archive']} {y['mjd_utc']:.4f} (site {site} unknown)"); continue
        R1, t1 = O.observer_bary(np.array([y["mjd_utc"]]), site); u1, s1 = O.predict_sigma(p0, c0, t0, t1, R1)
        sep = np.degrees(np.arccos(np.clip(u1[0] @ radec_to_unit(np.radians(y["ra"]), np.radians(y["dec"])), -1, 1))) * 3600
        if sep > max(10.0, 5 * s1[0]):
            left.append(f"{y['archive']} {y['mjd_utc']:.4f} ({sep:.0f}\" from the arc's prediction)"); continue
        s = max(float(y.get("sigma") or 0.2), 0.10 if site == "T14" else 0.03)
        rows.append((site, y["mjd_utc"], y["ra"], y["dec"], s, y["archive"], y.get("mag"), y.get("band") or "", y.get("snr") or 0, None, y))
    rows.sort(key=lambda q: q[1])
    Rb = np.zeros((len(rows), 3)); tt = np.zeros(len(rows))
    for site in {r[0] for r in rows}:
        k = [j for j, r in enumerate(rows) if r[0] == site]
        r_, t_ = O.observer_bary(np.array([rows[j][1] for j in k]), site); Rb[k] = r_; tt[k] = t_
    uo = radec_to_unit(np.radians([r[2] for r in rows]), np.radians([r[3] for r in rows]))
    p, cov, res, chi2, dof = O.fit(p0, t0, tt, Rb, uo, np.array([r[4] for r in rows]))
    res = np.hypot(np.asarray(res)[:, 0], np.asarray(res)[:, 1]); el = O.elements(p, t0, helio=True)
    now = []
    for dt in DATES:
        ra_, de_, s_ = AX.predict(p, cov, t0, Time(dt + "T00:00:00", scale="utc").mjd)
        now.append(dict(date=dt, ra=round(float(ra_[0]), 6), dec=round(float(de_[0]), 6), sigma_arcsec=round(float(s_[0]), 2)))
    rh, de, eg = QB.geometry(p, t0, [Time(w["date"]).mjd for w in now])
    H = c.get("H")
    for w, a, b, g in zip(now, rh, de, eg):
        w.update(V=None if H is None else round(float(H + 5 * np.log10(a * b)), 1), elong=round(float(g), 0), r_au=round(float(a), 2))
    ms = np.array([r[1] for r in rows]); src = {}
    for r in rows:
        src[r[5]] = src.get(r[5], 0) + 1
    orbit = dict(fit="two-body, all rows (Rubin included)", epoch_tdb=t0, p=[float(x) for x in p], cov=np.asarray(cov).tolist(),
                 a=round(float(el["a"]), 3), e=round(float(el["e"]), 4), i=round(float(el["i"]), 3), rms_arcsec=round(float(np.sqrt((res ** 2).mean())), 3),
                 max_resid_arcsec=round(float(res.max()), 3), n=len(rows), by_source=src, nights=len({int(m - 0.3) for m in ms}),
                 first=Time(ms.min(), format="mjd").isot[:10], last=Time(ms.max(), format="mjd").isot[:10], left_out=left, places=now)
    return rows, orbit, res


def psv(lab, rows, res):
    COL = ["trkSub", "mode", "stn", "obsTime", "ra", "dec", "rmsRA", "rmsDec", "astCat", "mag", "rmsMag", "band", "photCat", "logSNR", "exp", "notes"]
    R = []
    for row, rr in zip(rows, res):
        site, m, ra, dec, s, src, mag, band, sn, ex = row[:10]
        ours = src.startswith("DECam (our")
        R.append(dict(trkSub=lab[:8], mode="CCD", stn=site, obsTime=Time(m, format="mjd", scale="utc").isot[:22] + "Z", ra=f"{ra:.6f}", dec=f"{dec:.6f}",
                      rmsRA=f"{s:.3f}", rmsDec=f"{s:.3f}", astCat="Gaia2", mag="" if mag is None or not np.isfinite(mag) else f"{mag:.2f}",
                      rmsMag="" if mag is None or not sn else f"{1.0857 / max(sn, 1):.2f}", band=QB.BAND.get(band, band) if band else "",
                      photCat="PS1_DR1" if ours else "", logSNR=f"{np.log10(max(sn, 1)):.2f}" if sn else "", exp="" if ex is None else f"{ex:.1f}",
                      notes=("" if ours else src[:40]) + f" resid {rr:.2f}\""))
    w = {c: max(len(c), *[len(r[c]) for r in R]) for c in COL}
    head = ["# version=2022", "# observatory", "! mpcCode W84", "# submitter", "! name PROJECT_LEAD_NAME", "# measurers", "! name PROJECT_LEAD_NAME",
            "# comment", "! line Not a filing; candidates for follow-up. Rows with notes come from other archives. Rubin DP2 rows (stn X05) are proprietary Rubin data: not for redistribution or MPC submission.",
            "|".join(c.ljust(w[c]) for c in COL)]
    return "\n".join(head + ["|".join(r[c].ljust(w[c]) for c in COL) for r in R]) + "\n"


def sent_mail(lab):
    out = []
    if not lab:
        return out
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "r[12]", "observer", "*"))) + sorted(glob.glob(os.path.join(HERE, "runs", "r[12]", "observer", "*", "*"))):
        b = os.path.join(d, "body.txt")
        if os.path.isdir(d) and os.path.exists(b) and re.search(rf"\b{lab}\b", open(b, errors="replace").read()):
            out.append(d)
    return out


def folder_name(c):
    I = c["identities"]; lab = I.get("label")
    if lab:
        return f"{c['cid']}_{lab}"
    if I.get("run"):
        return f"{c['cid']}_{I['run']}-{I.get('region')}-rank{I.get('rank')}"
    return f"{c['cid']}_{re.sub(r'[^A-Za-z0-9.-]+', '-', c['name'])}"


def main():
    idx = json.load(open(os.path.join(CD, "index.json"))); cards = []
    for c0 in idx["cards"]:
        cards.append(json.load(open(os.path.join(CD, c0["cid"], "card.json"))))
    def has_arc(c):
        i = c["identities"]["lead_id"]; run = c["identities"]["run"]
        return os.path.exists(os.path.join(HERE, "runs", run, "arc", i + ".json")) or os.path.exists(os.path.join(HERE, "runs", run, "arc", i + ".refit.json"))
    dbt = [c for c in cards if c["identities"].get("run") and has_arc(c)]     # bodies still waiting for their arc get card + evidence only
    our_rows(dbt); det_detail(dbt)
    EX = exports()
    CV = {}
    try:                                                   # every other-archive frame checked per body (TRKSUB01 session), when exported
        CV = json.load(open(os.path.join(PS, "allarch", "coverage_all.json")))["bodies"]
    except Exception:
        pass
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    table = []
    for c in cards:
        I = c["identities"]; lab = I.get("label") or ""; name = folder_name(c)
        F = os.path.join(OUT, name) if c.get("active", True) else os.path.join(OUT, "_dropped", name)
        os.makedirs(F)
        shutil.copy(os.path.join(CD, c["cid"], "card.json"), F)
        if os.path.isdir(os.path.join(CD, c["cid"], "evidence")):
            shutil.copytree(os.path.join(CD, c["cid"], "evidence"), os.path.join(F, "evidence"))
            for f in glob.glob(os.path.join(F, "evidence", "*.json")):     # shared folder: no local paths
                t = open(f).read(); t2 = t.replace("$ALE_HOME/", "").replace("$HOME/", "~/")
                if t2 != t:
                    open(f, "w").write(t2)
        note = []; orbit = None; nsrc = {}
        if I.get("run") and not has_arc(c):
            note.append("Arc not extended yet (follow-up running); positions and orbit come with the next refresh.")
        elif I.get("run"):
            i = I["lead_id"]; ours = json.load(open(os.path.join(CACHE, i + ".json")))
            DD = json.load(open(os.path.join(CACHE, i + ".det.json"))) if os.path.exists(os.path.join(CACHE, i + ".det.json")) else {}
            extra = DD.get("detections") or None
            other = EX.get(i, {}).get("rows", [])
            for a in I.get("also_found_as") or []:
                other += EX.get(a, {}).get("rows", [])
            try:
                rows, orbit, res = fit_all(c, ours, other, extra)
                open(os.path.join(F, "positions.psv"), "w").write(psv(lab or c["cid"], rows, res))
                PJ = [dict(site=r[0], utc=Time(r[1], format="mjd").isot[:22], mjd_utc=r[1], ra=r[2], dec=r[3], sigma_arcsec=r[4], source=r[5], mag=r[6], band=r[7],
                           snr=r[8], exptime=r[9], resid_arcsec=round(float(q), 3),
                           **{k: v for k, v in (r[10] or {}).items() if k not in ("site", "mjd_utc", "mjd_mid", "ra", "dec", "sigma", "mag", "band", "snr", "exptime", "archive")})
                      for r, q in zip(rows, res)]
                json.dump(PJ, open(os.path.join(F, "positions.json"), "w"), indent=1); write_csv(os.path.join(F, "positions.csv"), PJ)
                if DD:
                    json.dump(dict(scenario=DD["scenario"], region=DD["region"]), open(os.path.join(F, "search_region.json"), "w"), indent=1)
                cov_ = coverage(c, orbit); write_csv(os.path.join(F, "decam_coverage.csv"), cov_)
                orbit["decam_coverage"] = dict(exposures_on_path=len(cov_), with_detection=sum(1 for x in cov_ if x["detected_rows"]),
                                               expected_snr_over_5=sum(1 for x in cov_ if (x["expected_snr"] or 0) >= 5))
                write_csv(os.path.join(F, "ephemeris.csv"), ephemeris(c, orbit))
                json.dump(orbit, open(os.path.join(F, "orbit.json"), "w"), indent=1); nsrc = orbit["by_source"]
            except Exception as e:
                note.append(f"Joint fit failed ({e!r}); positions not written.")
        else:
            for f in sorted(glob.glob(os.path.join(PS, "mpc_report", f"{lab or 'none'}*.psv"))):
                shutil.copy(f, F)
            for k, v in EX.items():
                if c["name"] in k or (lab and lab in k):
                    json.dump(v, open(os.path.join(F, "other_archive_rows.json"), "w"), indent=1); nsrc = {"other archives": len(v["rows"])}
        cvk = [k for k in ([I.get("lead_id")] + list(I.get("also_found_as") or []) + [lab, c["name"]]) if k and k in CV]
        cv_notes = [n for k in cvk for n in (CV[k].get("notes") or [])]
        if cvk:
            write_csv(os.path.join(F, "other_archive_coverage.csv"), [dict(r, key=k) for k in cvk for r in CV[k].get("frames", [])])
        for d in sent_mail(lab):
            dd = os.path.join(F, "sent", os.path.relpath(d, os.path.join(HERE, "runs")).replace("/", "_"))
            os.makedirs(dd, exist_ok=True); shutil.copy(os.path.join(d, "body.txt"), dd)
            for f in glob.glob(os.path.join(d, "*.psv")):
                shutil.copy(f, dd)
        rub = sum(v for k, v in nsrc.items() if "rubin" in k.lower())
        L = [open(os.path.join(CD, c["cid"], "README.md")).read().rstrip(), "", "## This folder", "",
             f"Built {Time.now().iso[:16]} UTC by `rsi/decam-big-tno/candidates/package.py` from the database folder `candidates/{c['cid']}/`.", ""]
        if orbit:
            L += [f"- `positions.psv` / `positions.json`: {orbit['n']} positions on {orbit['nights']} nights, {orbit['first']} to {orbit['last']}: "
                  + ", ".join(f"{k} {v}" for k, v in orbit["by_source"].items()) + ".",
                  f"- `orbit.json`: one two-body fit through all of them, a {orbit['a']} au, e {orbit['e']}, i {orbit['i']} deg, rms {orbit['rms_arcsec']}\"; "
                  f"places for {', '.join(DATES)}." + (f" Left out: {'; '.join(orbit['left_out'])}." if orbit["left_out"] else "")]
        if orbit and orbit.get("decam_coverage"):
            dc = orbit["decam_coverage"]
            L += ["- `positions.csv`: the same as a table, with every field we have: for our detections the DECam exposure (file stem, e.g. "
                  "`c4d_YYMMDD_hhmmss` in the NOIRLab archive), CCD, pixel x/y, flux and error, S/N, flag, sharpness, pointing centre, airmass, "
                  "seeing, zero point and 5-sigma depth; for other archives the frame id and measurement fields.",
                  f"- `decam_coverage.csv`: all {dc['exposures_on_path']} public DECam exposures (2013-2026) of the search region whose pointing lies "
                  "within 1.1 deg of the body's predicted place (CCD gaps not modelled): predicted place and 1-sigma, predicted V, expected S/N from "
                  f"the exposure's depth, and our detection on it if any ({dc['with_detection']} with a detection; {dc['expected_snr_over_5']} with "
                  "expected S/N >= 5).",
                  "- `ephemeris.csv`: nightly place (0h UTC) from 2026-11-20 to 2027-05-31 with 1-sigma, V, distances and elongation.",
                  "- `search_region.json`: the sky region our search covered."]
        if cvk:
            nf = sum(len(CV[k].get("frames", [])) for k in cvk)
            L += [f"- `other_archive_coverage.csv`: {nf} frames of other telescopes and DECam programs checked for this body, seen or not "
                  "(archive, frame, CCD, time, band, exposure, frame centre, predicted place and 1-sigma, predicted mag, expected and measured S/N, "
                  "offset, outcome, SMOKA request; `counted` marks the nights the archive verdict counts)."]
            L += [f"  - {n}" for n in cv_notes]
        L += ["- `card.json`, `evidence/`: the database card and its evidence files."]
        if os.path.isdir(os.path.join(F, "sent")):
            L += ["- `sent/`: the emails to OBSERVER_NAME that named this body, with the PSV each carried (Rubin rows were never in them)."]
        if rub:
            L += [f"- Holds {rub} Rubin DP2 positions (stn X05): proprietary Rubin data (DP2), for this follow-up only; not for redistribution or MPC submission."]
        L += [f"- {x}" for x in note]
        if c.get("image_tests"):              # TRKSUB01's tests in public frames (2026-11-23)
            L += ["", "## Measured in public images", ""] + [f"- {t['pass']}: {t['verdict']}{' (provisional)' if t.get('provisional') else ''}. {t.get('evidence') or ''}" for t in c["image_tests"]]
        open(os.path.join(F, "README.md"), "w").write("\n".join(L) + "\n")
        table.append(dict(folder=os.path.relpath(F, OUT), catalogue=c["cid"], label=lab, name=c["name"], active=c.get("active", True), stage=c.get("stage"),
                          known_as=((c.get("identities") or {}).get("known_as") or "")[:80],
                          images=(lambda t: "" if not t else ("SEEN AGAIN" if any(x["verdict"].startswith("SEEN AGAIN") for x in t)
                                                             and not any(x["verdict"] == "NOT SEEN" for x in t) else
                                                             "NOT SEEN" if any(x["verdict"] == "NOT SEEN" for x in t) else t[-1]["verdict"])
                                  + (" (provisional)" if any(x.get("provisional") for x in t) else ""))(c.get("image_tests") or []),
                          H=c.get("H"), D_km=json.dumps(c.get("D_km")), a=(orbit or {}).get("a", (c.get("orbit") or {}).get("a")),
                          e=(orbit or {}).get("e", (c.get("orbit") or {}).get("e")), i=(orbit or {}).get("i", (c.get("orbit") or {}).get("i")),
                          positions=(orbit or {}).get("n", ""), rubin_positions=rub or "", nights=(orbit or {}).get("nights", c.get("nights")),
                          first=(orbit or {}).get("first", ""), last=(orbit or {}).get("last", ""), rms_arcsec=(orbit or {}).get("rms_arcsec", ""),
                          sigma_2026_11_20=((orbit or {}).get("places") or [{}])[0].get("sigma_arcsec", ""), V_2026_11_20=((orbit or {}).get("places") or [{}])[0].get("V", ""),
                          sent=";".join(os.path.basename(d) for d in sent_mail(lab)), search=I.get("search", ""), lead_id=I.get("lead_id", "")))
    with open(os.path.join(OUT, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    shutil.copy(os.path.join(HERE, "progress", "objects", "index.html"), os.path.join(OUT, "catalogue.html"))
    na = sum(1 for t in table if t["active"])
    open(os.path.join(OUT, "README.md"), "w").write("\n".join([
        "# DECam TNO candidates", "",
        f"Every candidate distant body from the archive-wide DECam searches, one folder per object: {na} active, {len(table) - na} dropped (under `_dropped/`).",
        f"Built {Time.now().iso[:16]} UTC from the candidate database (`ale-astronomy/rsi/decam-big-tno/candidates/`, script `package.py`); "
        "rebuilt on every database refresh, so edit the database, not this folder.", "",
        "- `catalogue.html`: a copy of the Promising New Bodies page (<private-page-url>); open it in a browser.",
        "- `index.csv`: one row per object: label, stage, size, orbit, number of positions (Rubin separately), the 1-sigma and V on 2026-11-20, emails sent.",
        "- Each folder: `README.md` (identities, evidence ladder, history), `positions.psv` (ADES; our detections plus every other archive, Rubin DP2 included, "
        "notes column naming the source), `positions.csv` (the same with every field: DECam exposure file stem, CCD, pixel x/y, flux, S/N, flags, pointing, "
        "airmass, seeing, depth; other archives' frame ids), `orbit.json` (one fit through all positions), `ephemeris.csv` (nightly place to 2027-05-31 "
        "with 1-sigma and V), `decam_coverage.csv` (every public DECam exposure 2013-2026 of the search region that covered the predicted path, with "
        "expected S/N and our detection if any), `other_archive_coverage.csv` (frames of other telescopes checked, when available), "
        "`search_region.json`, `card.json`, `evidence/`, `sent/`.",
        "", "Shared for follow-up; please don't pass it on. Unfiled bodies' positions are not public, Rubin DP2 rows (station X05) are proprietary Rubin data, "
        "and nothing here is a filing. Labels: C### is our catalogue number; DBTnnn and TRKSUBnn are the labels used in emails."]) + "\n")
    print(f"built {OUT}: {len(table)} folders ({na} active)")
    if "--no-sync" not in sys.argv:
        os.makedirs(DRIVE, exist_ok=True)
        r = subprocess.run(["rsync", "-a", "--checksum", "--delete", OUT + "/", DRIVE + "/"], capture_output=True, text=True)
        print("rsync to Drive", "ok" if r.returncode == 0 else r.stderr[-400:], DRIVE)


if __name__ == "__main__":
    main()
