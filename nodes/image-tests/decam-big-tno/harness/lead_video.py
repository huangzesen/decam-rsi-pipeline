#!/usr/bin/env python3
"""Video of a lead in the DECam images: every exposure of its region where the fitted orbit puts it on a chip, those the
program used and the others ("extended": exposures the program did not use, in the same and in other seasons).

2026-11-16.

Each frame has two panels, north up, east left, 0.263"/pixel:
  left   the exposure, centred on the place the orbit predicts for that moment (the body stays in the centre if the orbit is right)
  right  the same piece of sky in an exposure of another night (same band when there is one, the deepest): a body that
         moves is not there; a star or galaxy is
and a caption: date, band, exposure time, depth, predicted magnitude and 1-sigma, and one of
  USED BY THE PROGRAM / ADDED BY THE ARC EXTENSION / IN THE CATALOGUE, NOT USED / NOT IN THE CATALOGUE.

Steps (all on the Mac; the images come one chip at a time from the NOIRLab archive, ~5 MB each, held in memory only):
  plan RUN ID...        orbit (runs/RUN/arc/<id>.json when there is one, else a fit of the lead's detections), the place in
                        every exposure of the region, the nearest catalogue detections and their chips (read on NERSC)
                        -> runs/RUN/video/<id>/plan.json
  fetch RUN ID...       one chip per frame from https://astroarchive.noirlab.edu/api/retrieve/<md5>/?hdus=<n>, resampled
                        stamps -> runs/RUN/video/<id>/stamps.npz
  render RUN ID...      -> runs/RUN/video/<id>/<name>.mp4 and <name>_sheet.png
  all RUN ID...         the three steps
ID is a substring of the lead id; "seen" = every lead of the run whose repeat test says SEEN AGAIN.
Usage: DBT_SITE=nersc ~/anaconda3/bin/python -I harness/lead_video.py all s38 seen
Private: frames carry dates and places. Nothing here goes on a shared page.
"""
import glob, io, json, os, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "harness"))
import lead_check as LC                                      # noqa: E402
from lead_check import O, radec_to_unit, unit_to_radec      # noqa: E402

LINKER = os.path.expanduser("$ALE_HOME/planet-search/linker")
N, PIX = 161, 0.263                 # stamp: 161 px = 42"
SIGMAX, MAXF, KNN = 2.0, 60, 6
TAGS = {"used": ("USED BY THE PROGRAM", (120, 220, 140)), "added": ("ADDED BY THE ARC EXTENSION", (120, 200, 240)),
        "detected": ("IN THE CATALOGUE, NOT USED BY THE PROGRAM", (250, 200, 90)), "none": ("NOT IN THE CATALOGUE", (240, 120, 120))}

REMOTE = r'''
import json, sys, numpy as np
from scipy.spatial import cKDTree
S = "%SCEN%"; q = json.load(sys.stdin); out = {}
for sid, job in q.items():
    L = lambda k: np.load(f"{S}/{sid}/det_{k}.npy", mmap_mode="r")
    ra, dec, ex, sn, cc, dx, dy = L("ra"), L("dec"), L("exp"), L("snr"), L("ccd"), L("x"), L("y")
    st = np.searchsorted(ex, np.arange(int(ex[-1]) + 2)); o = {}
    for e, pts in job["pts"].items():
        e = int(e); a, b = st[e], st[e + 1]
        if b - a < 1:
            o[str(e)] = [[] for _ in pts]; continue
        r_, d_ = np.radians(np.asarray(ra[a:b])), np.radians(np.asarray(dec[a:b]))
        T = cKDTree(np.stack([np.cos(d_) * np.cos(r_), np.cos(d_) * np.sin(r_), np.sin(d_)], 1))
        P = np.radians(np.asarray(pts, float))
        U = np.stack([np.cos(P[:, 1]) * np.cos(P[:, 0]), np.cos(P[:, 1]) * np.sin(P[:, 0]), np.sin(P[:, 1])], 1)
        k = min(%K%, b - a); dd, ii = T.query(U, k=k); dd = dd.reshape(len(pts), k); ii = ii.reshape(len(pts), k)
        o[str(e)] = [[[float(np.degrees(2 * np.arcsin(min(x / 2, 1.0))) * 3600), int(a + i), float(sn[a + i]), int(cc[a + i]), float(dx[a + i]), float(dy[a + i])]
                      for x, i in zip(d1, i1)] for d1, i1 in zip(dd, ii)]
    out[sid] = o
json.dump(out, sys.stdout)
'''.replace("%SCEN%", LC.SCEN).replace("%K%", str(KNN))


def remote(q):
    assert "'" not in REMOTE
    p = subprocess.run(LC.VSSH + [f"{LC.VPY} '{REMOTE}'"], input=json.dumps(q), capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        sys.exit(f"remote read failed: {p.stderr[-500:]}")
    return json.loads(p.stdout)


def leads(run, ids):
    RUN = os.path.join(HERE, "runs", run)
    led = {x["id"]: x for x in json.load(open(os.path.join(RUN, "leads_sent.json")))}
    res = {}
    for f in sorted(glob.glob(os.path.join(RUN, "lead_check", "*.json"))):
        if f.endswith("_plan.json") or f.endswith("_exposures.json"):
            continue
        for d in json.load(open(f)):
            if isinstance(d, dict) and "id" in d:
                res[d["id"]] = d
    if ids == ["seen"]:
        pick = [i for i, d in res.items() if str(d.get("verdict", "")).startswith("SEEN AGAIN") and i in led]
    else:
        pick = [i for i in led if any(s in i for s in ids)]
    return RUN, [led[i] for i in pick], res


def names():
    lab = {}
    for f in glob.glob(os.path.join(HERE, "runs", "*", "observer_labels.json")):
        lab.update(json.load(open(f)))
    return lab


def short(i, lab):
    return lab.get(i) or re.sub(r"^dbt-", "", i)


def sep_arcsec(ra1, dec1, ra2, dec2):
    u1 = radec_to_unit(np.radians(ra1), np.radians(dec1)); u2 = radec_to_unit(np.radians(ra2), np.radians(dec2))
    return np.degrees(np.arccos(np.clip(np.sum(u1 * u2, axis=-1), -1, 1))) * 3600


# ---------------------------------------------------------------- plan
def plan(run, ids):
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    RUN, led, res = leads(run, ids); cache = os.path.join(RUN, "lead_check")
    for x in led:
        outd = os.path.join(RUN, "video", x["id"]); os.makedirs(outd, exist_ok=True)
        if os.path.exists(os.path.join(outd, "plan.json")) and "--redo" not in sys.argv:
            print(x["id"], "plan exists"); continue
        sid = x["sid"]; E = LC.exposures(sid, cache); allt = np.array([e["mjd_mid"] for e in E])
        arcf = os.path.join(RUN, "arc", x["id"] + ".json"); arc = json.load(open(arcf)) if os.path.exists(arcf) else None
        rows = list(x["det"]); more = [r for r in (arc["rows"] if arc else []) if r not in set(rows)]
        R1 = LC.remote({sid: {"rows": rows + more}})[sid]["rows"]
        D = np.array([R1[str(r)] for r in rows + more]); ex = D[:, 2].astype(int); t_utc = allt[ex]
        used = set(ex[:len(rows)].tolist()); added = set(ex[len(rows):].tolist()) - used
        if arc:
            p, cov, t0 = np.array(arc["p"]), np.array(arc["cov"]), arc["epoch_tdb"]
        else:
            b = LC.fit_lead(t_utc, D[:, 0], D[:, 1], 0.08 + 0.6 / np.maximum(D[:, 5], 3.0))
            if b is None:
                print(x["id"], "no bound orbit"); continue
            _, p, cov, t0, _ = b
        el = O.elements(p, t0, helio=True)
        ok = np.array([k for k in range(len(E)) if E[k].get("ra0") is not None])
        Rc, tc = O.observer_bary(allt[ok], "W84"); uc, s1 = O.predict_sigma(p, cov, t0, tc, Rc)
        ra, dec = unit_to_radec(uc); ra, dec = np.degrees(ra) % 360, np.degrees(dec)
        pos = O.propagate(p[:3], p[3:], tc - t0); dl = np.linalg.norm(pos - Rc, axis=1)
        sun = get_body_barycentric("sun", Time(tc, format="mjd", scale="tdb")).xyz.to("au").value.T
        rh = np.linalg.norm(pos - sun, axis=1)
        H = (res.get(x["id"]) or {}).get("H")
        if H is None:                                    # as lead_check: from the lead's own fluxes
            j = {int(k): n for n, k in enumerate(ok)}
            hh = [E[e]["magzero"] - 2.5 * np.log10(max(f, 1e-9)) - LC.COL.get(E[e]["band"], 0.0) - 5 * np.log10(rh[j[e]] * dl[j[e]])
                  for e, f in zip(ex, D[:, 3]) if E[e].get("magzero") is not None and e in j]
            H = float(np.median(hh)) if hh else None
        fr = []
        for j, k in enumerate(ok):
            m = E[k]; k = int(k); mine = k in used or k in added
            pm = None if H is None else float(H + 5 * np.log10(rh[j] * dl[j]) + LC.COL.get(m["band"], 0.0))
            far = sep_arcsec(m["ra0"], m["dec0"], ra[j], dec[j]) / 3600
            if far > 1.2 or (not mine and (s1[j] > SIGMAX or m.get("m5") is None or pm is None or pm > m["m5"] + 0.2)):
                continue
            fr.append(dict(exp=k, stem=m["stem"], mjd=float(allt[k]), band=m["band"], exptime=m["exptime"], m5=m.get("m5"),
                           ra=float(ra[j]), dec=float(dec[j]), sigma=float(s1[j]), mag=pm, r=float(rh[j]),
                           tag="used" if k in used else ("added" if k in added else None)))
        if not fr:
            print(x["id"], "no frame"); continue
        NR = remote({sid: {"pts": {str(f["exp"]): [[f["ra"], f["dec"]]] for f in fr}}})[sid]
        keep, off = [], 0
        for f in fr:
            nn = NR[str(f["exp"])][0]
            if not nn or nn[0][0] > 90:                  # nothing within 90": between chips or off the camera
                off += 1; continue
            f["near"] = [round(nn[0][0], 2), nn[0][1], round(nn[0][2], 1)]
            f["ccds"] = list(dict.fromkeys(c for s_, _, _, c, _, _ in nn if s_ < 120))
            if f["tag"] is None:
                f["tag"] = "detected" if nn[0][0] <= max(1.0, 3 * f["sigma"]) else "none"
            keep.append(f)
        rank = {"used": 0, "added": 0, "detected": 1, "none": 2}
        keep.sort(key=lambda f: (rank[f["tag"]], (f["mag"] or 99) - (f["m5"] or 0)))
        dropped = max(0, len(keep) - MAXF); keep = sorted(keep[:MAXF], key=lambda f: f["mjd"])
        # the same sky on another night: up to 2 exposures per frame whose chips cover the frame's place while the body is elsewhere
        pr = np.array([E[k]["ra0"] for k in ok]); pd_ = np.array([E[k]["dec0"] for k in ok]); q, pos = {}, {}
        for n, f in enumerate(keep):
            d = sep_arcsec(f["ra"], f["dec"], pr, pd_) / 3600; moved = sep_arcsec(f["ra"], f["dec"], ra, dec)
            c = [j for j in range(len(ok)) if d[j] < 1.0 and abs(allt[ok[j]] - f["mjd"]) > 0.5 and moved[j] > 8 and E[ok[j]].get("m5") is not None]
            c.sort(key=lambda j: (E[ok[j]]["band"] != f["band"], -E[ok[j]]["m5"]))
            for j in c[:6]:
                e = str(int(ok[j])); q.setdefault(e, []).append([f["ra"], f["dec"]]); pos[(n, e)] = len(q[e]) - 1
        NR2 = remote({sid: {"pts": q}})[sid] if q else {}
        for (n, e), k_ in pos.items():
            nn = NR2[e][k_]; f = keep[n]; m = E[int(e)]
            if nn and nn[0][0] < 40 and len(f.setdefault("refs", [])) < 2:
                f["refs"].append(dict(exp=int(e), stem=m["stem"], mjd=float(allt[int(e)]), band=m["band"], exptime=m["exptime"], m5=m["m5"],
                                      ccds=list(dict.fromkeys(c_ for s_, _, _, c_, _, _ in nn if s_ < 120))))
        lc = res.get(x["id"]) or {}
        out = dict(id=x["id"], sid=sid, run=run, orbit_from="arc extension" if arc else "the lead's detections", H=H,
                   a=float(el["a"]), e=float(el["e"]), i=float(el["i"]), verdict=lc.get("verdict"), D_km=lc.get("D_km"),
                   off_chip=off, dropped_over_cap=dropped, frames=keep)
        json.dump(out, open(os.path.join(outd, "plan.json"), "w"), indent=1)
        c = {t: sum(1 for f in keep if f["tag"] == t) for t in TAGS}
        print(x["id"], len(keep), "frames", c, "off chip", off, "over cap", dropped, flush=True)


# ---------------------------------------------------------------- fetch
_idx, _pat = {}, re.compile(r"(c4d_\d{6}_\d{6})_ooi_([^_]+)_([^.]+)\.fits\.fz$")


def md5_of(stem):
    y = 2000 + int(stem[4:6])
    for yy in (y, y - 1, y + 1):
        if yy not in _idx:
            f = os.path.join(LINKER, "data_real", "noirlab_decam_instcal", f"instcal_{yy}.json"); d = {}
            if os.path.exists(f):
                for name, md5, size, rel, expnum in json.load(open(f))["rows"]:
                    m = _pat.search(name)
                    if m and (m.group(1) not in d or name > d[m.group(1)][0]):
                        d[m.group(1)] = (name, md5)
            _idx[yy] = d
        if stem in _idx[yy]:
            return _idx[yy][stem]
    return None


def get_hdu(md5, ccd):
    """The image HDU of one chip. HDU n is chip n unless chips are missing from the file, so step by the difference."""
    from astropy.io import fits
    n = ccd
    for _ in range(5):
        for a in range(4):
            try:
                b = urllib.request.urlopen(urllib.request.Request(f"https://astroarchive.noirlab.edu/api/retrieve/{md5}/?hdus={n}",
                                                                  headers={"User-Agent": "lead-video/1.0"}), timeout=300).read()
                break
            except Exception as err:
                if a == 3:
                    raise
                time.sleep(5 * (a + 1))
        hl = fits.open(io.BytesIO(b))
        if len(hl) < 2:
            return None
        c = hl[1].header.get("CCDNUM")
        if c == ccd:
            return hl[1].header, np.asarray(hl[1].data, np.float32), len(b)
        if c is None or n + ccd - c < 1 or n + ccd - c > 62:
            return None
        n += ccd - c
    return None


def stamp(data, w, ra0, dec0):
    """N x N stamp centred on (ra0, dec0), north up, east left, PIX arcsec per pixel; None when the centre is off the chip."""
    from scipy.ndimage import map_coordinates
    x0, y0 = w.all_world2pix([[ra0, dec0]], 0, quiet=True)[0]
    if not (8 < x0 < data.shape[1] - 8 and 8 < y0 < data.shape[0] - 8):
        return None
    g = (np.arange(N) - N // 2) * PIX / 3600
    eta, xi = np.meshgrid(-g, -g, indexing="ij")           # row 0 = north, column 0 = east
    de = dec0 + eta; ra = ra0 + xi / np.cos(np.radians(de))
    xy = w.all_world2pix(np.stack([ra.ravel(), de.ravel()], 1), 0, quiet=True)
    return map_coordinates(data, [xy[:, 1], xy[:, 0]], order=1, mode="constant", cval=np.nan).reshape(N, N).astype(np.float32)


def chip_stamp(stem, ccds, ra, dec):
    from astropy.wcs import WCS
    import warnings
    warnings.simplefilter("ignore")
    m = md5_of(stem)
    if m is None:
        return None, "no InstCal file in the archive list", 0
    nb = 0
    for ccd in ccds:
        g = get_hdu(m[1], ccd)
        if g is None:
            continue
        hdr, data, n1 = g; nb += n1
        s = stamp(data, WCS(hdr), ra, dec)
        if s is not None:
            return s, f"ccd {ccd} {m[0]}", nb
    return None, "place not on the chips tried", nb


def fetch_one(k, F):
    f = F[k]
    own, note, nb = chip_stamp(f["stem"], f["ccds"], f["ra"], f["dec"])
    ref, rk = None, -1
    if own is not None:
        for n, r in enumerate(f.get("refs", [])):
            s, _, n1 = chip_stamp(r["stem"], r["ccds"], f["ra"], f["dec"]); nb += n1
            if s is not None and np.isfinite(s).mean() > 0.9:
                ref, rk = s, n; break
    return k, own, ref, rk, note, nb


def fetch(run, ids):
    RUN, led, _ = leads(run, ids)
    for x in led:
        outd = os.path.join(RUN, "video", x["id"]); pf = os.path.join(outd, "plan.json")
        if not os.path.exists(pf) or (os.path.exists(os.path.join(outd, "stamps.npz")) and "--redo" not in sys.argv):
            continue
        P = json.load(open(pf)); F = P["frames"]; t0 = time.time()
        own = np.full((len(F), N, N), np.nan, np.float32); ref = np.full((len(F), N, N), np.nan, np.float32); nb = 0
        with ThreadPoolExecutor(4) as ex:
            for k, o, r, rk, note, n1 in ex.map(lambda k: fetch_one(k, F), range(len(F))):
                F[k]["image"] = note; F[k]["ok"] = o is not None; F[k]["ref"] = rk; nb += n1
                if o is not None:
                    own[k] = o
                if r is not None:
                    ref[k] = r
        np.savez_compressed(os.path.join(outd, "stamps.npz"), own=own, ref=ref)
        P["downloaded_mb"] = round(nb / 1e6, 1)
        json.dump(P, open(pf, "w"), indent=1)
        print(x["id"], sum(f["ok"] for f in F), "of", len(F), "frames with an image;", sum(f["ref"] >= 0 for f in F), "with an other-night view;",
              P["downloaded_mb"], "MB;", round(time.time() - t0), "s", flush=True)


# ---------------------------------------------------------------- render
def norm(s):
    from astropy.visualization import ZScaleInterval
    v = s[np.isfinite(s)]
    if v.size < 50:
        return np.zeros(s.shape, np.uint8)
    lo, hi = ZScaleInterval(contrast=0.15).get_limits(v); hi = max(hi, lo + 1e-6)
    return (np.clip((np.nan_to_num(s, nan=lo) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def render(run, ids):
    from PIL import Image, ImageDraw, ImageFont
    from astropy.time import Time
    RUN, led, _ = leads(run, ids); lab = names()
    ft = lambda n, b=False: ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if b else "/System/Library/Fonts/Supplemental/Arial.ttf", n)
    Z = 3; S = N * Z; G = 14; W = 2 * S + 3 * G; TOP, BOT = 96, 60; Hh = TOP + S + BOT; Hh += Hh % 2; W += W % 2
    for x in led:
        outd = os.path.join(RUN, "video", x["id"]); pf = os.path.join(outd, "plan.json"); sf = os.path.join(outd, "stamps.npz")
        if not (os.path.exists(pf) and os.path.exists(sf)):
            continue
        P = json.load(open(pf)); A = np.load(sf); name = short(x["id"], lab)
        F = [(k, f) for k, f in enumerate(P["frames"]) if f.get("ok")]
        if not F:
            print(x["id"], "no image"); continue
        tmp = os.path.join(outd, "frames"); os.makedirs(tmp, exist_ok=True)
        for g in glob.glob(os.path.join(tmp, "*.png")):
            os.remove(g)
        t_first = F[0][1]["mjd"]; D = P.get("D_km") or {}
        size = f"H {P['H']:.1f}" + (f", about {D.get('0.20')}-{D.get('0.05')} km" if D else "") if P.get("H") is not None else ""
        head = f"{name}   a {P['a']:.1f} au, e {P['e']:.2f}, i {P['i']:.1f} deg   {size}"
        ims = []
        for n, (k, f) in enumerate(F):
            im = Image.new("RGB", (W, Hh), (14, 16, 20)); d = ImageDraw.Draw(im)
            for col, arr in ((0, A["own"][k]), (1, A["ref"][k])):
                x0 = G + col * (S + G)
                if np.isfinite(arr).any():
                    im.paste(Image.fromarray(norm(arr)).resize((S, S), Image.NEAREST).convert("RGB"), (x0, TOP))
                else:
                    d.rectangle([x0, TOP, x0 + S, TOP + S], fill=(26, 29, 35))
                    d.text((x0 + S // 2, TOP + S // 2), "no other-night exposure\ncovers this place", font=ft(15), fill=(150, 155, 165), anchor="mm", align="center")
                cx, cy = x0 + S // 2, TOP + S // 2; c = TAGS[f["tag"]][1] if col == 0 else (170, 175, 185)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):            # four ticks, the centre left clear
                    d.line([cx + 12 * dx, cy + 12 * dy, cx + 30 * dx, cy + 30 * dy], fill=c, width=2)
            t = Time(f["mjd"], format="mjd")
            d.text((G, 10), head, font=ft(17, True), fill=(235, 238, 242))
            d.text((W - G, 10), f"{n + 1} / {len(F)}", font=ft(15), fill=(150, 155, 165), anchor="ra")
            m5 = f"{f['m5']:.1f}" if f.get("m5") is not None else "?"; pm = f"{f['mag']:.1f}" if f.get("mag") is not None else "?"
            d.text((G, 36), f"{t.strftime('%Y-%m-%d %H:%M')} UT   day {f['mjd'] - t_first:+.1f}   {f['band']} band, {f['exptime']:.0f} s   "
                            f"5-sigma depth {m5}, predicted {pm}   place known to {f['sigma']:.2f}\"", font=ft(15), fill=(205, 210, 218))
            tag, colr = TAGS[f["tag"]]
            nr = f.get("near") or [None, None, None]
            extra = f"   catalogue source {nr[0]:.2f}\" from the predicted place, S/N {nr[2]:.0f}" if f["tag"] != "none" else f"   nearest catalogue source {nr[0]:.1f}\" away"
            d.text((G, 64), tag, font=ft(16, True), fill=colr); d.text((G + d.textlength(tag, font=ft(16, True)), 65), extra, font=ft(14), fill=(170, 175, 185))
            d.text((G, TOP + S + 8), "this exposure, centred on the predicted place", font=ft(14), fill=(205, 210, 218))
            if f.get("ref", -1) >= 0:
                h = f["refs"][f["ref"]]
                d.text((2 * G + S, TOP + S + 8), f"same sky on {Time(h['mjd'], format='mjd').strftime('%Y-%m-%d')} ({h['band']}, {h['exptime']:.0f} s, depth {h['m5']:.1f}): the body is elsewhere", font=ft(14), fill=(205, 210, 218))
            d.text((G, TOP + S + 30), f"north up, east left, {N * PIX:.0f}\" across", font=ft(13), fill=(130, 135, 145))
            fn = os.path.join(tmp, f"f_{n:03d}.png"); im.save(fn); ims.append(im)
        mp4 = os.path.join(outd, name + ".mp4")
        p = subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-framerate", "1", "-i", os.path.join(tmp, "f_%03d.png"),
                            "-vf", "fps=10", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4], capture_output=True, text=True)
        if p.returncode != 0:
            print(x["id"], "ffmpeg failed:", p.stderr[-300:]); continue
        cols = 4; rows_ = (len(ims) + cols - 1) // cols; sh = Image.new("RGB", (cols * W // 2, rows_ * Hh // 2), (14, 16, 20))
        for n, im in enumerate(ims):
            sh.paste(im.resize((W // 2, Hh // 2), Image.LANCZOS), ((n % cols) * W // 2, (n // cols) * Hh // 2))
        sh.save(os.path.join(outd, name + "_sheet.png"))
        for g in glob.glob(os.path.join(tmp, "*.png")):
            os.remove(g)
        os.rmdir(tmp)
        print(x["id"], "->", mp4, len(F), "frames", flush=True)


if __name__ == "__main__":
    cmd, run, ids = sys.argv[1], sys.argv[2], [a for a in sys.argv[3:] if not a.startswith("--")]
    for step, fn in (("plan", plan), ("fetch", fetch), ("render", render)):
        if cmd in (step, "all"):
            fn(run, ids)
