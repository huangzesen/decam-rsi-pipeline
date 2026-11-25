"""imgrecord14 step 4: stages from R/candidates/C###/card.json (every card, active or not as the card says) against the table.
Lists: (1) every card at "seen in another archive" or "strong: seen again, clean, multi-year arc" with no SEEN AGAIN row of either
form (in the table with other rows, or not in the table at all); (2) rows that contradict the stage (imgrecord13's stages.py
rules: a firm stage with a NOT SEEN row and no SEEN AGAIN row; "seen in another archive" with no SEEN AGAIN row; a card with
both a SEEN AGAIN and a NOT SEEN row); (3) the ps1night2 cards' stages and rows."""
import json, os, re
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = "$ALE_HOME/rsi/decam-big-tno"
FIRM = ("filed", "sent to observer_name", "seen in another archive", "strong: seen again, clean, multi-year arc")
t = json.load(open(os.path.join(H, "imagetests.json")))
cards = {}
for c in sorted(os.listdir(os.path.join(R, "candidates"))):
    p = os.path.join(R, "candidates", c, "card.json")
    if re.fullmatch(r"C\d{3}", c) and os.path.exists(p):
        j = json.load(open(p)); cards[c] = ((j.get("stage") or "").strip(), j.get("active"))
for want in ("seen in another archive", "strong: seen again, clean, multi-year arc"):
    no_row, no_seen = [], []
    for c, (s, act) in cards.items():
        if s.lower() != want:
            continue
        rs = t["cards"].get(c)
        if rs is None:
            no_row.append(c + ("" if act else "(inactive)"))
        elif not any(r["verdict"].startswith("SEEN AGAIN") for r in rs):
            no_seen.append("%s[%s]" % (c, ", ".join("%s %s" % (r["pass"], r["verdict"]) for r in rs)))
    n = sum(1 for s, _ in cards.values() if s.lower() == want)
    print("STAGE %r: %d cards; in the table without a SEEN AGAIN row %d: %s" % (want, n, len(no_seen), "; ".join(no_seen)))
    print("   not in the table at all %d: %s" % (len(no_row), " ".join(no_row)))
for c, rs in sorted(t["cards"].items()):
    s = cards.get(c, ("(no card)", None))[0]; sl = s.lower()
    seen = [r["pass"] for r in rs if r["verdict"].startswith("SEEN AGAIN")]
    ns = [r["pass"] for r in rs if r["verdict"] == "NOT SEEN"]
    firm = any(sl.startswith(f) for f in FIRM)
    if seen and ns:
        print("CONTRA", c, s, "| rows disagree: SEEN AGAIN %s, NOT SEEN %s" % (seen, ns))
    if firm and ns and not seen:
        print("CONTRA", c, s, "| firm stage, NOT SEEN row(s) %s and no SEEN AGAIN row" % ns)
    if firm and ns:
        print("FIRM+NOTSEEN", c, s, ns)
    if sl.startswith("seen in another archive") and not seen:
        print("CONTRA", c, s, "| 'seen in another archive' but no SEEN AGAIN row")
for c in ("C###", "C###", "C###", "C###"):
    print("CARD", c, "|", cards.get(c), "|", [(r["pass"], r["verdict"]) for r in t["cards"].get(c, [])])
