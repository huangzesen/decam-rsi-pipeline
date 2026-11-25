"""Write allarch/novelty_hand.json's hand verdicts into novelty_queue/done/<id>.result.json (old verdict kept as verdict_before_hand),
for leads sync_novelty.py does not cover (leads not on a card). 2026-11-22, TRKSUB01 session. Usage: python3 -I apply_hand.py"""
import json, os
C = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = C + "/novelty_queue/done/"
H = json.load(open(C + "/allarch/novelty_hand.json")); n = 0
for i, v in H.items():
    f = D + i + ".result.json"
    if not os.path.exists(f): print("no result file:", i); continue
    R = json.load(open(f))
    if R.get("verdict") == v: continue
    R.setdefault("verdict_before_hand", R.get("verdict")); R["verdict"] = v; R["hand"] = "allarch/novelty_hand.json"
    json.dump(R, open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f); n += 1; print(i[:50], "->", v[:90])
print(n, "result files written")
