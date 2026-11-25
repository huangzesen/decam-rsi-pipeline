#!/usr/bin/env python3
"""Promising New Bodies page, <private-page-url> (2026-11-17, 2026-11-19).
Reads the candidate database candidates/index.json (candidates/build.py; active cards only) and observe.json
(observe.py --sites G37: the next 14 nights at LDT), writes progress/objects/index.html from page.html.
No sky positions or observation dates of unfiled bodies: verdict texts lose their dates, and detection ids, scenario
ids and positions stay in the database. Usage: python3 -I progress/objects/build_page.py (or candidates/refresh.sh)"""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(os.path.dirname(HERE))
DATE = re.compile(r"\b(19|20)\d\d([-/]\d\d){1,2}(/\d\d)?\b|\b(19|20)\d\d-(19|20)?\d\d\b")
GROUP = {"seen in another archive": "arch", "seen once in another archive": "arch", "sent to OBSERVER_NAME": "sent",
         "seen; others' unlinked MPC tracklets": "sent", "filed": "sent", "strong: seen again, clean, multi-year arc": "strong",
         "seen again, clean, short arc": "strong", "new": "strong", "seen again, needs a refit": "refit",
         "seen again, fixed-source test pending": "refit", "seen": "refit", "large, not yet tested in other images": "large",
         "held back after the hand check": "other", "seen again on one night only": "other", "not seen where expected": "other", "probably not seen where expected": "other", "already reported by another survey": "other",
         "known MPC body": "known", "probably a known MPC body": "known", "known to another survey, not in the MPC": "known", "probably known to another survey, not in the MPC": "known"}
STAGE = {"seen in another archive": "Seen in another archive", "seen once in another archive": "Seen once in another archive",
         "sent to OBSERVER_NAME": "Sent to OBSERVER_NAME", "seen; others' unlinked MPC tracklets": "Real; in others' unlinked tracklets",
         "filed": "Filed with the MPC", "strong: seen again, clean, multi-year arc": "Seen again, multi-year arc",
         "seen again, clean, short arc": "Seen again, short arc", "new": "Confirmed new, not yet filed",
         "seen again, needs a refit": "Seen again, needs a refit", "seen again, fixed-source test pending": "Fixed-source test pending",
         "seen": "Seen in other images, orbit open", "large, not yet tested in other images": "Large, not yet tested",
         "held back after the hand check": "Held back after the hand check", "seen again on one night only": "Seen again on one night only", "not seen where expected": "Not seen where expected", "probably not seen where expected": "Probably not seen where expected",
         "already reported by another survey": "Already reported by another survey",
         "known MPC body": "Known MPC body", "probably a known MPC body": "Probably a known MPC body",
         "known to another survey, not in the MPC": "Known to another survey, not in the MPC",
         "probably known to another survey, not in the MPC": "Probably known to another survey, not in the MPC"}


def nodate(s):
    return DATE.sub("(date)", s) if isinstance(s, str) else s


db = json.load(open(os.path.join(TOP, "candidates", "index.json")))
cards = []
for c in db["cards"]:
    if not c.get("active"):
        continue
    I, o = c["identities"], c.get("orbit") or {}
    cards.append(dict(cid=c["cid"], name=c["name"], kind=c["kind"], key=c.get("key"), label=I.get("label"), also=I.get("also_found_as") or [],
                      aliases=I.get("aliases") or [], search=I.get("search") if c["kind"] == "decam-big-tno" else "planet-search hill climbs",
                      program=I.get("program"), region=I.get("region"), rank=I.get("rank"), p=I.get("p"), field=I.get("field"),
                      batch=I.get("observer_batch"), rubin=I.get("rubin_rights"), group=GROUP.get(c["stage"], "other"),
                      stage=c["stage"], stageTxt=STAGE.get(c["stage"], c["stage"]), status=nodate(c.get("status")), summary=nodate(c.get("summary")),
                      facts=[[k, nodate(v)] for k, v in (c.get("facts") or [])], H=c.get("H"), D=c.get("D_km"),
                      a=o.get("a"), e=o.get("e"), i=o.get("i"), r=o.get("r"), rms=o.get("rms"), onote=nodate(o.get("note")), mpc=bool(o.get("mpc")), imgs=[dict(p=t.get("pass"), v=t.get("verdict"), ev=nodate(t.get("evidence")), prov=bool(t.get("provisional"))) for t in c.get("image_tests") or []], arc=c.get("arc_years"), nights=c.get("nights"),
                      ladder=[dict(step=s["step"], state=s["state"], text=nodate(s["text"])) for s in c["ladder"]],
                      files=c.get("evidence") or [], added=c.get("added"), history=c.get("history") or []))
DB = json.dumps(dict(asof=db["asof"], cards=cards))
OBSF = os.path.join(HERE, "observe.json")
OBS = open(OBSF).read() if os.path.exists(OBSF) else "null"
if OBS != "null":
    o = json.loads(OBS); o["sites"] = [s for s in o["sites"] if s["code"] == "G37"]
    for s in o["sites"]:
        for n in s["nights"]:
            for b in n["bodies"].values():
                for k in ("ra", "dec", "rate"):
                    b.pop(k, None)          # never places on this page
    OBS = json.dumps(o, separators=(",", ":"))
page = open(os.path.join(HERE, "page.html")).read().replace("/*DB*/null", DB.replace("</", "<\\/")).replace("/*OBS*/null", OBS.replace("</", "<\\/"))
open(os.path.join(HERE, "index.html"), "w").write(page)
open(os.path.join(HERE, "db_public.json"), "w").write(DB)   # the same friend-safe cards, read by pages/team-site/decam/build.py
print("wrote", os.path.join(HERE, "index.html"), len(page) // 1024, "kB;", len(cards), "cards")
