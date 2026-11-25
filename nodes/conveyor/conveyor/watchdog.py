#!/usr/bin/env python3
"""conveyor watchdog (2026-11-19). Called at the end of every tick; writes
conveyor/health.json and appends "ALERT ..." / "OK ..." lines to conveyor/alerts.log when a check changes state (an alert
repeats once an hour while it lasts). The main session watches alerts.log with a Monitor and tells the user.
Checks: NERSC and Vista reachable; NERSC scratch under the 15 TiB cap (alert above 15.5, critical above 18.5); downloads
moving (alert when below 50 MB/s for 30 min while scratch is under the cap); the Vista relay copying (no new relayed
file for 2 h while the fill downloads); unit stages not stuck (prep with no detection progress for 2 h); search sites
held back (site_bad); the same tick error 3+ times in the last 60 log lines; hill climb log quiet for 4 h; Globus copies of the conveyor failed."""
import json, os, subprocess, time
H = os.path.dirname(os.path.abspath(__file__)); PS = os.path.dirname(H); ALE = os.path.dirname(PS)
HEALTH = os.path.join(H, "health.json"); ALERTS = os.path.join(H, "alerts.log")
SSH = ["ssh", "-S", os.path.expanduser("$SSH_CM_DIR/nersc-$NERSC_USER.sock"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "$NERSC_USER@perlmutter.nersc.gov"]


def check(st, scans):
    now = time.time(); h = json.load(open(HEALTH)) if os.path.exists(HEALTH) else {}
    prev = h.get("alerts", {}); hist = h.get("rates", [])
    cur = {}
    for s in ("nersc", "vista"):
        if s not in scans:
            cur[f"{s}_down"] = f"{s} not reachable (SSH master gone? a new login may be needed)"
    # the same error on every tick (11-19 05:08-09:55: a15's build step raised each tick and starved downloads and detection)
    try:
        tail = subprocess.run(["tail", "-n", "60", os.path.join(H, "loop.out")], capture_output=True, text=True).stdout.splitlines()
        errs = [l for l in tail if " error" in l and ("units side" in l or "step error" in l or "release error" in l or l.split(" ")[2:3] in (["keep_downloads"], ["feed_detection"]))]
        if len(errs) >= 3:
            cur["tick_errors"] = f"{len(errs)} errors in the last ticks, e.g. {errs[-1][15:200]}"
    except Exception:
        pass
    rate = None
    if "nersc" in scans:
        try:
            p = subprocess.run(SSH + ["python3 -I $NERSC_SCRATCH/units/rates.py"], capture_output=True, text=True, timeout=300)
            rate = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as e:
            cur["rates"] = f"download rate unreadable ({type(e).__name__})"
    if rate:
        hist = [r for r in hist if now - r["t"] < 7200] + [dict(t=now, **rate)]
        sc = rate["scratch_tib"]
        # 15 TiB is a target kept by releasing (2026-11-19); downloads pause only at the limits file's HIGH (18.5 TiB, quota 20)
        if sc > rate["high_tib"]:
            cur["scratch"] = f"CRITICAL NERSC scratch {sc} TiB at the download pause ({rate['high_tib']} TiB, quota 20): releases far behind"
        elif sc > 15.5:
            cur["scratch"] = f"NERSC scratch {sc} TiB above the 15 TiB target: releases lagging (downloads still run)"
        last30 = [r for r in hist if now - r["t"] < 1800]
        import unit as U                                   # no slow-download alert while NERSC downloads are paused on purpose
        if len(last30) >= 5 and max(r["mbs"] for r in last30) < 50 and sc < rate["low_tib"] and not U.nersc_dl_paused():
            cur["downloads"] = f"downloads below 50 MB/s for 30 min with scratch {sc} TiB under the cap ({last30[-1]['logs']} active logs)"
    rl = os.path.join(PS, "fill", "relay", "relay.log")
    if os.path.exists(rl) and now - os.path.getmtime(rl) > 5400:
        cur["relay"] = "fill relay log silent for 90 min (relay.py loop stopped?)"
    for u in st.get("units", []):
        if u.get("stage") == "prep" and u.get("detect"):
            left = sum(v[0] for v in u["detect"].values())
            key = f"stuck_{u['id']}"; mem = h.get("progress", {}).get(u["id"])
            if mem and mem["left"] == left and left > 0 and now - mem["t"] > 7200:
                cur[key] = f"unit {u['id']} detection made no progress for 2 h ({left} exposures left)"
            if not mem or mem["left"] != left:
                h.setdefault("progress", {})[u["id"]] = dict(left=left, t=now)
    for s, b in (st.get("site_bad") or {}).items():
        cur[f"site_{s}"] = f"search site {s} held back after a failed region ({b.get('task')}); check its log, then clear site_bad"
    cl = os.path.join(ALE, "rsi", "decam-big-tno", "runs", "r2", "climb.log")
    if os.path.exists(os.path.join(ALE, "rsi", "decam-big-tno", "progress", "STOP_TRACKER")):   # climb ended 11-19 10:38 (tracker stopped): no climb alert
        cl = None
    if cl and os.path.exists(cl) and now - os.path.getmtime(cl) > 4 * 3600:
        cur["climb"] = "hill climb log quiet for 4 h"
    lines = []
    for k, msg in cur.items():
        if k not in prev or now - prev[k]["last"] > 3600:
            lines.append(f"ALERT {k}: {msg}"); prev[k] = dict(msg=msg, since=prev.get(k, {}).get("since", now), last=now)
    for k in list(prev):
        if k not in cur:
            lines.append(f"OK {k}: cleared ({prev[k]['msg'][:80]})"); prev.pop(k)
    if lines:
        with open(ALERTS, "a") as f:
            for l in lines:
                f.write(time.strftime("%m-%d %H:%M ") + l + "\n")
    h.update(at=time.strftime("%Y-%m-%d %H:%M:%S"), alerts=prev, rates=hist[-40:], rate=rate)
    json.dump(h, open(HEALTH + ".new", "w"), indent=1); os.replace(HEALTH + ".new", HEALTH)
