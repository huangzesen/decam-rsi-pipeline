"""imgrecord14 (copy of imgrecord13, snapshots of imgrecord14; was a copy of imgrecord10/check_record.py, snapshots of imgrecord13; was a copy of imgrecord9/check_record.py; was a copy of imgrecord7/check_record.py, snapshots of imgrecord8; originally a copy of imgrecord4/check_record.py against imgrecord7/ snapshots; "started HH:MM" allowed too; RECORD_DIR = check a copy; record times "restarted HH:MM" allowed like "at HH:MM"): the record still parses; only new `passes` lines were added; the new lines and md rows obey the years-only rule
(the date-free checks of imagetests_build.check_text: no MJD, 5-digit numbers, dates, month names, clock times, coordinates)."""
import importlib.util, json, os, re, sys
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.dirname(H)
spec = importlib.util.spec_from_file_location("b", os.path.join(H, "imgrecord", "imagetests_build.py"))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
def probs(t):  # rule checks only (not the sentence count / length limits, which are for imagetests.json)
    return [p for p in b.check_text(t) if not p.startswith(("more than two", "too long"))]
old = json.load(open(os.path.join(H, "imgrecord14", "doublecheck.json.before")))
A = os.environ.get("RECORD_DIR", A)
new = json.load(open(os.path.join(A, "doublecheck.json")))
assert set(old) == set(new), "top-level keys changed"
added, bad, changed = 0, 0, 0
for c in new:
    if old[c] == new[c]:
        continue
    o, n = old[c], new[c]
    for k in set(o) | set(n):
        if k != "passes" and o.get(k) != n.get(k):
            changed += 1; print("CHANGED field", c, k)
    op, np_ = o.get("passes") or {}, n.get("passes") or {}
    for p in op:
        if op[p] != np_.get(p):
            changed += 1; print("CHANGED line", c, p)
    for p in set(np_) - set(op):
        added += 1
        pr = probs(np_[p])
        if pr:
            bad += 1; print("RULE", c, p, pr, "|", np_[p])
print("json: parses; lines added", added, "; old fields or lines changed", changed, "; new lines breaking the rule", bad)
md_old = open(os.path.join(H, "imgrecord14", "doublecheck.md.before")).read().split("\n")
md_new = open(os.path.join(A, "doublecheck.md")).read().split("\n")
gone = [l for l in md_old if l not in md_new]
new_l = [l for l in md_new if l not in md_old]
print("md: lines removed", len(gone), [l[:60] for l in gone])
for l in new_l:
    if not l.strip():
        continue
    t = re.sub(r"\b20\d\d-\d\d-\d\d \d\d:\d\d [A-Z]{3}\b|\| \d\d-\d\d \d\d:\d\d \||\b(since|at|restarted|started) \d\d:\d\d\b", "", l)  # record times, not body dates
    t = re.sub(r"\b[\w./{},-]+\.(json|md)\b|`[^`]*`", "", t)  # record file names
    print("md new line:", l[:70], "...", "rule ok" if not probs(t) else probs(t))
