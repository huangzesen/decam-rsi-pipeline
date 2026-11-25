import sys, json
exec(open('replay.py').read().split("def run(")[0])
import math
for s,(rows,vd,lab) in data.items():
    srt = sorted(range(len(vd)), key=lambda i: -float(vd[i]["p"]))
    no = [(r, i) for r, i in enumerate(srt) if not vd[i]["f"] or not math.isfinite(float(vd[i]["f"].get("cnn", float("nan"))))]
    print(s, "rows", len(vd), "no-cnn", len(no), "no-f", sum(1 for i in range(len(vd)) if not vd[i]["f"]), "ranks", [r for r,i in no][:20], "hits among", [lab[i] for r,i in no if lab[i]])
    hc = sorted(round(vd[i]["f"]["cnn"],1) for i in range(len(vd)) if lab[i] and vd[i]["f"] and "cnn" in vd[i]["f"])
    print("  hit cnn", hc[:10])
    pr = [(r, round(vd[i]["f"].get("cnn",-99),1), round(vd[i]["f"]["zlbno"],1)) for r,i in enumerate(srt) if vd[i]["f"] and F.prom_target(vd[i]["f"], vd[i]["src"]) is not None]
    print("  promotable (rank,cnn,zlbno)", pr[:30])
