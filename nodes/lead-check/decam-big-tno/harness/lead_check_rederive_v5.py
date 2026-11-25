#!/usr/bin/env python3
"""Re-derive lead-check verdicts written by v4 under the v5 rule (an exposure whose predicted place holds another source is
not tested), from the per-hit details v4 stored; no new data is read (2026-11-22). Usage: python3 -I lead_check_rederive_v5.py"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lead_check import blocked_exposures, verdict_v5
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
n_files = n_changed = 0
HAND = (json.load(open(os.path.join(HERE, "harness", "lead_check_hand_blocked.json"))).get("leads") or {}
        if os.path.exists(os.path.join(HERE, "harness", "lead_check_hand_blocked.json")) else {})   # by-hand untestable exposures (2026-11-23)
_ec = {}


def EXP(run_dir, sid):
    k = (run_dir, sid)
    if k not in _ec:
        f_ = os.path.join(run_dir, "lead_check", f"{sid}_exposures.json")
        _ec[k] = json.load(open(f_)) if os.path.exists(f_) else None
    return _ec[k]
for f in glob.glob(os.path.join(HERE, "runs", "*", "lead_check", "*.json")):
    if f.endswith(("_plan.json", "_exposures.json")):
        continue
    try:
        L = json.load(open(f))
    except Exception:
        continue
    if not isinstance(L, list) or not any(str(d.get("check", "")).startswith(("v4", "v5", "v6")) for d in L):
        continue
    ch = 0
    run_dir = os.path.dirname(os.path.dirname(f))
    SID = {x["id"]: x["sid"] for x in (json.load(open(os.path.join(run_dir, "leads_sent.json"))) if os.path.exists(os.path.join(run_dir, "leads_sent.json")) else [])}
    for d in L:
        if not str(d.get("check", "")).startswith(("v4", "v5", "v6")):
            continue
        bs = blocked_exposures(d.get("extra_hits", []), d.get("rejected_why", [])); b = len(bs)
        if d["id"] in HAND:
            b = max(b, int(HAND[d["id"]]["n_blocked"])); d["blocked_by_hand"] = HAND[d["id"]]["why"]
        sid = SID.get(d["id"]); E = EXP(run_dir, sid) if sid else None
        kept = [h["exp"] for h in d.get("extra_hits", []) if h["exp"] not in bs]
        hn = len({int(E[e]["mjd_mid"] - 0.5) for e in kept}) if E else None
        v = verdict_v5(d["expected"], d["found"], d["extra"], d["extra_found"], d["null_hits"], b, hn)
        if v != d["verdict"]:
            print(os.path.relpath(f, HERE), d["id"][-30:], d["verdict"][:12], "->", v[:12]); ch += 1
        d.update(verdict=v, blocked=b, hit_nights=hn, check="v6b (re-derived from stored details, 2026-11-22)")
    json.dump(L, open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f)
    n_files += 1; n_changed += ch
print(n_files, "files,", n_changed, "verdicts changed")
