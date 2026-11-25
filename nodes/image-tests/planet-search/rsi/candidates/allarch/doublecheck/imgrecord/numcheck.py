"""Every number in imagetests.json evidence must appear in the pass's own files (json entry, .md, .handback.md); print the ones that do not."""
import json, os, re
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(H, "imagetests.json")))
src = {}
for p in d["passes"]:
    n = p["pass"]; t = ""
    for ext in (".json", ".md", ".handback.md"):
        f = os.path.join(H, n + ext)
        if os.path.exists(f):
            t += open(f).read()
    src[n] = t.replace("\\\"", "\"")
for c, rs in d["cards"].items():
    for r in rs:
        nums = re.findall(r"-?\d+(?:[.,]\d+)?", r["evidence"])
        miss = [x for x in nums if x not in src[r["pass"]] and not re.fullmatch(r"(19|20)\d\d|[0-9]|1[0-2]", x)]
        if miss:
            print(c, r["pass"], miss)
