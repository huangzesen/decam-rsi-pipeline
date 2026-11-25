"""imgrecord14: the ps1night2 joined entries under the new season rule, their fingerprints, and (C###) the fingerprint with the
old label put back, which must equal imgrecord13's a48328cb1397 (so only the label moved, not the pass's entry).
Usage (from A/doublecheck): ~/anaconda3/bin/python -I imgrecord14/fpcheck.py"""
import importlib.util, json, os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("b", os.path.join(H, "imgrecord", "imagetests_build.py"))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
for line in b.FP_TEXT.strip().splitlines():
    p, c, h = line.split(); b.FP[(p, c)] = h
ents = b.entries_of("ps1night2", json.load(open(os.path.join(H, "ps1night2_imagetests.json"))), [])
for c, e in ents:
    print(c, "fp now", b.fp(e), "| FP_TEXT", b.FP.get(("ps1night2", c)), "|", e["verdict"][:60])
    if e["verdict"].startswith("UNSURE"):
        old = dict(e, verdict="SEEN AGAIN (combined)" + e["verdict"][len("UNSURE"):])
        print("   with the old label put back:", b.fp(old))
for k in sorted(b.SEASON_LOG, key=str):
    print("LOG", b.SEASON_LOG[k])
