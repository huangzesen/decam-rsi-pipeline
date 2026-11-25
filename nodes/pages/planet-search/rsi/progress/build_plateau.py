"""Build the sub-page explaining the gen071 jump (2026-11-04).

Reads out/plateau_hist.json (written by plateau_hist.py) and writes a snapshot to
$PAGES_OUT/plateau/index.html; $PAGE_PUBLISHER publishes it on its next
tick. Stops without writing if a sealed body name or field id reached the page: shared pages stay friend-safe.

Usage: python3 -I build_plateau.py
"""
import glob, json, os, re, time

HERE = os.path.dirname(os.path.abspath(__file__))
GENS = os.path.join(HERE, "..", "runs", "r1", "gens")
OUT = os.path.expanduser("$PAGES_OUT/plateau/index.html")
FIRST = 30


def num(g):
    return int(re.match(r"gen(\d+)", g).group(1))


def main():
    hist = json.load(open(os.path.join(HERE, "out", "plateau_hist.json")))
    rewards = sorted(((num(os.path.basename(os.path.dirname(p))), json.load(open(p)).get("reward") or 0.0)
                      for p in glob.glob(os.path.join(GENS, "gen*", "reward.json"))))
    rows = []
    for h in hist:
        if h["n"] < FIRST:
            continue
        best = max(r for n, r in rewards if n <= h["n"])
        # plateau_hist lists the bodies in truth-file order; on the page Body 1 is the one at 0.84 in gen071
        two, one = h["bodies"]
        rows.append(dict(n=h["n"], reward=h["reward"], best=best, cut=h["cut"], b1=one, b2=two))
    html = open(os.path.join(HERE, "plateau.html")).read()
    html = html.replace("/*PLATEAU_DATA*/", json.dumps({"rows": rows}, separators=(",", ":")))
    html = html.replace("/*SNAPSHOT*/", time.strftime("%Y-%m-%d %H:%M %Z"))
    sealed = set()
    for p in glob.glob(os.path.join(GENS, "gen*", "grade", "grade.json")):
        d = json.load(open(p))
        sealed |= {h["pdes"] for h in d["hits"]} | set(d["settings"]["scenarios"])
    leaked = [s for s in sealed if s in html]
    if leaked:
        raise SystemExit(f"stopped: {len(leaked)} sealed names or field ids would reach the page")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    open(tmp, "w").write(html)
    os.replace(tmp, OUT)
    print("wrote", OUT, len(html), "rows", len(rows), "last", rows[-1]["n"])


if __name__ == "__main__":
    main()
