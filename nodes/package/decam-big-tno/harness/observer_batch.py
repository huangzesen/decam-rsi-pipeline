#!/usr/bin/env python3
"""Next batch of plausible decam-big-tno candidates for OBSERVER_NAME, ten per email (2026-11-13).

Plausible = the novelty check says NOT IN MPC FILES and lead_check says SEEN AGAIN (detected where its orbit predicts in
exposures the program did not use). Largest first (lowest H). Each goes through harness/arc_extend.py (other seasons;
place and 1-sigma on 2026-11-20 / 12-15 / 2027-01-15). OBSERVER_NAME's reply of 2026-11-12 asked for bodies near opposition now.

Writes runs/<run>/observer/batchNN/: body.txt, DBT_batchNN_W84.psv (every detection, ADES 2022 PSV), batch.json; with
--append also an IMAP draft in reply to his message (gm id GMAIL_ID, thread "Four new TNOs ..."), To OBSERVER_NAME
only (no cc: user, 2026-11-13). Send it with the gmail-send skill; then --record GMAIL_ID logs it in runs/<run>/observer_sent.json.
Labels DBTnnn (trkSub-style, our own; nothing is filed). Never prints credentials.
Usage: ~/anaconda3/bin/python -I harness/observer_batch.py RUN [--n 10] [--append] | --record GMAIL_ID BATCH
"""
import argparse, imaplib, json, os, re, subprocess, sys, time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "harness"))
import lead_check as LC                                       # noqa: E402
import arc_extend as AX                                       # noqa: E402
from lead_check import O                                      # noqa: E402

NQ = os.path.expanduser("$ALE_HOME/planet-search/rsi/candidates/novelty_queue")
CFG = os.path.expanduser("$IMAP_CREDENTIALS_FILE")
ADDR = "sender@example.org"
TO, CC = "observer@example.org", None                                # no cc (2026-11-13)
REPLY_GM = "GMAIL_ID"                                 # OBSERVER_NAME's reply of 2026-11-12 (thread GMAIL_ID)
SUBJECT = "Re: Four new TNOs from our archival DECam search: details"
ALLOW_SHORT = True                                            # 2026-11-13
BAND = {"g": "g", "r": "r", "i": "i", "z": "z", "Y": "y", "VR": "W"}


def novelty():
    out = {}
    for line in open(os.path.join(NQ, "RESULTS.md"), errors="replace"):
        m = re.match(r"\|\s*([\d-]+ [\d:]+)\s*\|\s*(dbt-\S+)\s*\|\s*([^|]+)\|", line)
        if m:
            out[m.group(2)] = (m.group(3).strip(), m.group(1))
    return out


def plausible(RUN):
    nov = novelty(); lc = {}
    for f in sorted(os.listdir(os.path.join(RUN, "lead_check"))):
        if re.match(r"gen\d{3}\.json$", f):
            for d in json.load(open(os.path.join(RUN, "lead_check", f))):
                lc[d["id"]] = d
    sent = {i for b in (json.load(open(os.path.join(RUN, "observer_sent.json"))) if os.path.exists(os.path.join(RUN, "observer_sent.json")) else []) for i in b["ids"]}
    ok = [d for i, d in lc.items() if d.get("verdict", "").startswith("SEEN AGAIN") and nov.get(i, ("",))[0].startswith("NOT IN MPC")
          and i not in sent and d.get("H") is not None]
    return sorted(ok, key=lambda d: d["H"]), nov


def geometry(p, t0, mjd):
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    R, t = O.observer_bary(np.atleast_1d(mjd), "W84")
    pos = O.propagate(p[:3], p[3:], t - t0)
    sun = get_body_barycentric("sun", Time(t, format="mjd", scale="tdb")).xyz.to("au").value.T
    rh = np.linalg.norm(pos - sun, axis=1); de = np.linalg.norm(pos - R, axis=1); es = np.linalg.norm(R - sun, axis=1)
    elong = np.degrees(np.arccos(np.clip((de ** 2 + es ** 2 - rh ** 2) / (2 * de * es), -1, 1)))
    return rh, de, elong


def ra_hms(ra):                                   # rounded once to 0.1 s, so 59.96 s carries into the minute (fix 2026-11-19)
    t = round((ra % 360) / 15 * 3600, 1) % 86400; h, r = divmod(t, 3600); m, s = divmod(r, 60); return f"{int(h):02d}:{int(m):02d}:{s:04.1f}"


def dec_dms(d):                                  # rounded once to 0.1", so 59.96" carries into the minute (fix 2026-11-19)
    g = "-" if d < 0 else "+"; t = round(abs(d) * 3600, 1); dd, r = divmod(t, 3600); m, s = divmod(r, 60); return f"{g}{int(dd):02d}:{int(m):02d}:{s:04.1f}"


def build(RUN, n):
    ok, nov = plausible(RUN)
    if len(ok) < n and not ALLOW_SHORT:
        sys.exit(f"only {len(ok)} plausible candidates not yet sent; a batch needs {n} (--short sends what there is)")
    pick = ok[:n]
    if not pick:
        sys.exit("no plausible candidate left to send")
    labf = os.path.join(RUN, "observer_labels.json"); labels = json.load(open(labf)) if os.path.exists(labf) else {}
    bdir = os.path.join(RUN, "observer"); os.makedirs(bdir, exist_ok=True)
    k = 1 + max([int(x[5:]) for x in os.listdir(bdir) if re.match(r"batch\d+$", x)] or [0])
    B = os.path.join(bdir, f"batch{k:02d}"); os.makedirs(B, exist_ok=True)
    led = {x["id"]: x for x in json.load(open(os.path.join(RUN, "leads_sent.json")))}
    rows_psv, items = [], []
    for d in pick:
        i = d["id"]
        if i not in labels:
            labels[i] = f"DBT{len(labels) + 1:03d}"
        af = os.path.join(RUN, "arc", i + ".json")
        if not os.path.exists(af):
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "harness/arc_extend.py"), os.path.basename(RUN), i],
                           cwd=HERE, capture_output=True, text=True, timeout=1800)
        A = json.load(open(af))
        x = led[i]; sid = x["sid"]; E = LC.exposures(sid, os.path.join(RUN, "lead_check"))
        R = AX.remote({sid: {"rows": A["rows"]}})[sid]["rows"]
        dets = sorted((E[int(v[2])]["mjd_mid"], v, E[int(v[2])]) for v in (R[str(r)] for r in A["rows"]))
        for mjd, (ra, dec, ex, fl, fe, sn), e in dets:
            from astropy.time import Time
            t = Time(mjd, format="mjd", scale="utc")
            rows_psv.append(dict(trkSub=labels[i], mode="CCD", stn="W84", obsTime=t.isot[:22] + "Z", ra=f"{ra:.6f}", dec=f"{dec:.6f}",
                                 rmsRA=f"{0.08 + 0.6 / max(sn, 3):.3f}", rmsDec=f"{0.08 + 0.6 / max(sn, 3):.3f}", astCat="Gaia2",
                                 mag=f"{e['magzero'] - 2.5 * np.log10(max(fl, 1e-9)):.2f}", rmsMag=f"{1.0857 / max(sn, 1):.2f}",
                                 band=BAND.get(e["band"], e["band"]), photCat="PS1_DR1", logSNR=f"{np.log10(max(sn, 1)):.2f}",
                                 exp=f"{e['exptime']:.1f}"))
        p = np.array(A["p"]); t0 = A["epoch_tdb"]
        from astropy.time import Time
        now = [Time(w["date"]).mjd for w in A["now"]]
        rh, de, el = geometry(p, t0, now)
        V = d["H"] + 5 * np.log10(rh * de)
        years = sorted({int(Time(m, format="mjd").decimalyear) for m, _, _ in dets})
        items.append(dict(label=labels[i], id=i, H=d["H"], H_spread=d.get("H_spread"), D=d["D_km"], a=A["a"], e=A["e"], i=A["i"], r=A["r"],
                          n=A["n"], nights=A["nights"], arc_days=A["arc_days"], rms=A["rms"], years=years,
                          extra=d.get("extra"), extra_found=d.get("extra_found"), null=d.get("null_hits"),
                          novelty=nov[i][0], novelty_at=nov[i][1],
                          confirm=(json.load(open(os.path.join(RUN, "confirm", i + ".json"))) if os.path.exists(os.path.join(RUN, "confirm", i + ".json")) else {}),
                          now=[dict(w, V=round(float(v), 1), elong=round(float(g), 0)) for w, v, g in zip(A["now"], V, el)],
                          seasons_added=sum(1 for s in A["seasons"] if s.get("found"))))
    json.dump(labels, open(labf, "w"), indent=1)
    COL = ["trkSub", "mode", "stn", "obsTime", "ra", "dec", "rmsRA", "rmsDec", "astCat", "mag", "rmsMag", "band", "photCat", "logSNR", "exp"]
    w = {c: max(len(c), *[len(r[c]) for r in rows_psv]) for c in COL}
    psv = ["# version=2022", "# observatory", "! mpcCode W84", "# submitter", "! name PROJECT_LEAD_NAME", "# observers",
           "! name NOIRLab Astro Data Archive, public DECam images", "# measurers", "! name PROJECT_LEAD_NAME", "# telescope",
           "! name Blanco", "! design Reflector", "! aperture 4.0", "! detector CCD", "# software",
           "! astrometry Community Pipeline WCS (Gaia DR2); 3x3 centroid of the S/N map",
           "! objectDetection Gaussian matched filter, S/N >= 4.5, single exposures", "# comment",
           "! line Candidates for follow-up only, not a filing. Labels DBTnnn are ours. Linked by programs written by AI agents.",
           "|".join(c.ljust(w[c]) for c in COL)] + ["|".join(r[c].ljust(w[c]) for c in COL) for r in rows_psv]
    pf = os.path.join(B, f"DBT_batch{k:02d}_W84.psv"); open(pf, "w").write("\n".join(psv) + "\n")
    body = email_body(items, k)
    open(os.path.join(B, "body.txt"), "w").write(body)
    json.dump(dict(batch=k, ids=[it["id"] for it in items], labels=[it["label"] for it in items], items=items, psv=pf), open(os.path.join(B, "batch.json"), "w"), indent=1)
    print(body); print(pf, len(rows_psv), "positions")
    return B


# Confirmation still running in the TRKSUB01 session (no verdict file yet): what to say instead of the default line.
PENDING = {}   # id -> text while a confirmation is still running; batch 1's two entries cleared after their verdicts (2026-11-13 13:24)


def telescopes(it):
    c = it.get("confirm") or {}
    if not c:
        return PENDING.get(it["id"], "no archive image (CFHT, ESO, Subaru) covers its predicted path closely enough to test it.")
    v = (c.get("verdict") or "").strip().replace("NO DECISIVE DATA: ", "nothing decisive: ")
    n = re.sub(r" \((?:no other DECam, )?dbt_\w+\.py\)", "", c.get("notes") or "")
    for a, b in (("is already in our archive5 region lists", "is already in our search, so there is no other DECam image"),
                 (" (from the Mac or NERSC)", ""), ("; retry pending", "; I will retry"),
                 ("for a red body near V 24", "for this body in u band"), ("its 2018-only arc", "its one-season arc")):
        n = n.replace(a, b)
    v = v.replace("its 2018-only arc", "its one-season arc")
    return v + (f". {n}" if n else "")


def sig(s):
    return "under 1\"" if s < 1 else f"{s:.0f}\""


def email_body(items, k):
    L = ["Hi OBSERVER_NAME,", "",
         ("You asked for bodies that are well placed now. " if k == 1 else "") +
         f"Here {'is one more candidate' if len(items) == 1 else f'are {len(items)} more candidates'} (batch {k}) from a second archival DECam search, this one over every public DECam "
         "exposure of a few ecliptic regions (2013-2026), aimed at large bodies (H about 8 or brighter). The linking programs are again "
         "written by AI agents. Each candidate below: matches nothing in the MPC's files (MPCORB, the isolated tracklet file, SkyBoT), "
         "and was detected again where its orbit predicts in exposures the program did not use, with no detection at control places "
         "30\" away. I searched the other seasons of the same archive along each orbit; where that found nothing the arc is one "
         "season, and the places for this winter carry the linear 1-sigma shown.", "",
         "Positions are single-exposure detections (Community Pipeline WCS, not re-measured), all in the attached ADES PSV. "
         "Nothing here is filed.", ""]
    for it in items:
        n0 = it["now"][0]; n1 = it["now"][1]
        L += [f"{it['label']}: H {it['H']:.1f}, roughly {it['D']['0.20']}-{it['D']['0.05']} km (albedo 0.20-0.05)",
              f"- Orbit (two-body, {it['n']} detections on {it['nights']} nights, arc {it['arc_days']:.0f} days in "
              f"{', '.join(map(str, it['years']))}, rms {it['rms']:.2f}\"): a {it['a']:.1f} au, e {it['e']:.2f}, i {it['i']:.1f} deg, "
              f"seen at {it['r']:.1f} au.",
              f"- Check: detected in {it['extra_found']} of {it['extra']} predicted exposures the program did not use; "
              f"{it['null']} detections at the shifted control places. Novelty: {it['novelty'].lower().replace('mpc', 'the MPC')} (checked {it['novelty_at'][:10]}).",
              f"- Other telescopes: {telescopes(it)}",
              f"- Now: {n0['date']} RA {ra_hms(n0['ra'])} Dec {dec_dms(n0['dec'])}, 1-sigma {sig(n0['sigma_arcsec'])}, "
              f"V about {n0['V']:.1f}, elongation {n0['elong']:.0f} deg; {n1['date']} RA {ra_hms(n1['ra'])} Dec {dec_dms(n1['dec'])}, "
              f"1-sigma {sig(n1['sigma_arcsec'])}.", ""]
    L += ["Happy to send stamps, the orbit files or the scripts behind each check.", "", "Best,", "PROJECT_LEAD_NAME", "", "--",
          "SENDER_NAME", "SENDER_TITLE"]
    return "\n".join(L)


def imap():
    acct = next(a for a in json.load(open(CFG))["accounts"] if a["email_address"] == ADDR)
    m = imaplib.IMAP4_SSL(acct["imap_host"], acct["imap_port"]); m.login(ADDR, acct["email_password"]); return m


def append(B):
    meta = json.load(open(os.path.join(B, "batch.json")))
    m = imap()
    m.select('"[Gmail]/All Mail"', readonly=True)
    _, ids = m.search(None, "X-GM-MSGID", str(int(REPLY_GM, 16)))
    _, data = m.fetch(ids[0].split()[0], "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID REFERENCES)])")
    hdr = data[0][1].decode(errors="replace")
    mid = re.search(r"^Message-ID:\s*(<[^>]+>)", hdr, re.I | re.M).group(1)
    refs = re.search(r"^References:\s*((?:.+\n?(?:[ \t].+\n?)*))", hdr, re.I | re.M)
    msg = EmailMessage()
    msg["From"] = f"SENDER_NAME <{ADDR}>"; msg["To"] = TO; msg["Subject"] = SUBJECT
    if CC:
        msg["Cc"] = CC
    msg["Date"] = formatdate(localtime=True); msg["Message-ID"] = make_msgid(domain="example.org")
    msg["In-Reply-To"] = mid; msg["References"] = (" ".join(refs.group(1).split()) + " " if refs else "") + mid
    msg.set_content(open(os.path.join(B, "body.txt")).read())
    msg.add_attachment(open(meta["psv"], "rb").read(), maintype="text", subtype="plain", filename=os.path.basename(meta["psv"]))
    typ, _ = m.append('"[Gmail]/Drafts"', "(\\Draft)", imaplib.Time2Internaldate(time.time()), msg.as_bytes())
    m.logout()
    print("APPEND", typ, msg["Subject"], "to", TO, "cc", CC or "-", "In-Reply-To", mid)


def record(RUN, gm, B):
    meta = json.load(open(os.path.join(B, "batch.json")))
    f = os.path.join(RUN, "observer_sent.json"); L = json.load(open(f)) if os.path.exists(f) else []
    L.append(dict(batch=meta["batch"], ids=meta["ids"], labels=meta["labels"], gmail_id=gm, at=time.strftime("%Y-%m-%d %H:%M")))
    json.dump(L, open(f, "w"), indent=1); print("recorded", meta["batch"], gm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--append", action="store_true"); ap.add_argument("--record", nargs=2, metavar=("GMAIL_ID", "BATCH_DIR"))
    ap.add_argument("--dry", action="store_true", help="list the plausible candidates only")
    ap.add_argument("--short", action="store_true", help="send fewer than --n when that is all there is (2026-11-13)")
    a = ap.parse_args(); RUN = os.path.join(HERE, "runs", a.run)
    if a.record:
        record(RUN, *a.record)
    elif a.dry:
        ok, _ = plausible(RUN)
        print(len(ok), "plausible, not yet sent"); [print(d["id"], d["H"], d.get("extra_found"), d.get("extra")) for d in ok[:30]]
    else:
        B = build(RUN, a.n)
        if a.append:
            append(B)
