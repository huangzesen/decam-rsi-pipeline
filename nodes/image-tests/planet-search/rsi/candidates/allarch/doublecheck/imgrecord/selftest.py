"""Self-test of imagetests_build: the evidence rule and the automatic builder on every current entry (no file written)."""
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("b", os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagetests_build.py"))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
bad = ["seen on a late-March 2025 night", "MJD 58143.2", "night 59322", "on 2019-03-04", "in 2019-03", "at RA 123.4567",
       "Dec -12.345", "file c4d_190304_123456_ooi_r_v1.fits.fz", "openimg/plan_C000.json", "at 04:31 UT", "12h34m56s",
       "lead dbt-sXX-a15-gen000-R000a-rank1-p0.50000", "exposure 912345", "on May 3", "the 4th of June", "2019.25"]
good = ["Seen on 2 nights of 2014-15 at S/N 8.2 and 8.3, 0.14\" and 0.20\" from the track; rms 0.127\" over 2015-2025.",
        "Chance from shifted copies 0.00186; offset 1,671\" off."]
fail = 0
for t in bad:
    if not b.check_text(t):
        print("NOT CAUGHT:", t); fail += 1
for t in good:
    if b.check_text(t):
        print("WRONGLY CAUGHT:", t, b.check_text(t)); fail += 1
# automatic builder on every current standard entry
notes = []
for name, _ in b.PASSES:
    p = os.path.join(b.HERE, name + ".json")
    if not os.path.exists(p) or name in ("deepframes", "deepframes2"):
        continue
    for c, e in b.entries_of(name, json.load(open(p)), notes):
        v = b.label_of(e.get("verdict") if isinstance(e.get("verdict"), str) else "") or "UNSURE"
        ev, how = b.auto_evidence(v, e, b.num(e.get("nights_seen")) or 0, b.num(e.get("nights_expected_not_seen")) or 0)
        pr = b.check_text(ev)
        if pr:
            print("AUTO FAILS", name, c, pr); fail += 1
        if "-v" in sys.argv:
            print(name, c, v, "|", how, "|", ev)
# a combined entry of openimg4's draft form
e = json.load(open(os.path.join(b.HERE, "openimg4", "combstat_C030_s10.json"))) if os.path.exists(os.path.join(b.HERE, "openimg4", "combstat_C030_s10.json")) else None
if e:
    ev, how = b.auto_evidence("SEEN AGAIN (combined)", e, b.num(e.get("nights")) or 0, 0)
    print("combined draft ->", how, "|", ev, "|", b.check_text(ev))
print("selftest failures:", fail)
