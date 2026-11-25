#!/usr/bin/env python3
"""Per-device download and transfer rows for "DECam Data Movement" (2026-11-20). Reads fill/dl_watch.json (written every 10 min by fill/dl_watch.py) and the
hand-kept fill/dash/dl_hand.json; build.py calls rows() and adds the four lists to out/status.json:
  dl_hist   one row per watch cycle: t, MB/s of NOIRLab downloads per device (nersc, ls6, vista, mac; null = unreachable),
            and MB/s per transfer route (m2n, n2v, l2v, l2n, v2n; null = no copy ran)
  dl_now    one row per device from the latest cycle: state, MB/s, files and bad-md5 share in the last 10 min, errors, note
  dl_xfer   one row per route: latest MB/s, and the last value seen while a copy ran
  dl_probes dated list of dl_watch's PROBE runs plus the hand-kept ones
  dl_streams one row per stream-budget event of dl_watch's controller (a device was limited: reason, streams old -> new, cool-down)
dl_hist also carries "sum", the MB/s of all devices together. State, streams and cool-down come from dl_watch.json (since 23:05).
dl_hand.json: {"probes": [...]} hand-kept probe rows.
Usage: python3 -I fill/dash/dl_rows.py   (prints the lists)"""
import json, os, time
H = os.path.dirname(os.path.abspath(__file__)); F = os.path.dirname(H)
DEV = [("nersc", "NERSC"), ("ls6", "Lonestar6"), ("vista", "Vista"), ("mac", "Mac")]
ROUTES = [("mac->nersc", "m2n", "Mac to NERSC", "rsync"), ("nersc->vista", "n2v", "NERSC to Vista", "Globus"),
          ("ls6->vista", "l2v", "Lonestar6 to Vista", "Globus"), ("ls6->nersc", "l2n", "Lonestar6 to NERSC", "Globus"),
          ("vista->nersc", "v2n", "Vista to NERSC", "Globus")]
STALE_MIN = 40


def rows():
    p = os.path.join(F, "dl_watch.json")
    hist = json.load(open(p)).get("history", []) if os.path.exists(p) else []
    hand = json.load(open(os.path.join(H, "dl_hand.json"))) if os.path.exists(os.path.join(H, "dl_hand.json")) else {}
    dl_hist = []
    for r in hist:
        row = {"t": r["time"]}
        for k, _ in DEV:
            s = (r.get("sites") or {}).get(k) or {}
            row[k] = s.get("mbs") if s.get("reach") else None
        row["sum"] = round(sum(row[k] or 0 for k, _ in DEV), 1)
        for key, short, _, _ in ROUTES:
            row[short] = (r.get("transfers") or {}).get(key)
        dl_hist.append(row)
    dl_now, dl_xfer, probes, dl_streams = [], [], [], []
    if hist:
        last = hist[-1]; age = (time.time() - last["t"]) / 60
        for k, name in DEV:
            s = (last.get("sites") or {}).get(k) or {}
            logs, mbs, files, bad, err = (s.get(x) or 0 for x in ("logs", "mbs", "files", "bad", "errors"))
            pu = s.get("pause_until") or 0
            if age > STALE_MIN: state = "no recent data"
            elif not s.get("reach"): state = "unreachable"
            else: state = s.get("state") or ("degraded" if s.get("degraded") else "downloading" if logs and mbs > 0 else "idle")
            dl_now.append(dict(device=name, state=state, mbs=mbs if s.get("reach") else None, used=s.get("used"), total=s.get("total"),
                               streams=(f"{s.get('used', 0)} of {s['total']}" if s.get("total") is not None else ""),
                               cool_until=(time.strftime("%Y-%m-%d %H:%M", time.localtime(pu)) if pu and pu > time.time() else ""),
                               files=files if s.get("reach") else None,
                               bad_pct=round(100 * bad / (files + bad), 1) if s.get("reach") and files + bad else (0 if s.get("reach") else None),
                               errors=err if s.get("reach") else None, note=s.get("limited", ""), asof=last["time"]))
        for key, short, name, how in ROUTES:
            seen = [(r["time"], r["transfers"][key]) for r in hist if (r.get("transfers") or {}).get(key) is not None]
            dl_xfer.append(dict(route=name, how=how, mbs=(last.get("transfers") or {}).get(key), last_mbs=seen[-1][1] if seen else None,
                                last_at=seen[-1][0] if seen else "", asof=last["time"]))
    names = dict(DEV)
    fmt = lambda v: (time.strftime("%Y-%m-%d %H:%M", time.localtime(v)) if isinstance(v, (int, float)) and v > 1e9 else (v or ""))
    for k, c in ((json.load(open(p)).get("ctl") or {}).items() if os.path.exists(p) else []):
        for e in c.get("events", []):
            o, n = (e.get("streams") or [None, None])[:2]
            dl_streams.append(dict(t=e.get("time", ""), device=names.get(k, k), why=e.get("why", ""), old=o, new=n, change=f"{o} to {n}",
                                   until=fmt(e.get("until"))))
    dl_streams.sort(key=lambda r: r["t"], reverse=True)
    for r in hist:
        for k, v in (r.get("probes") or {}).items():
            if isinstance(v, dict):
                probes.append(dict(t=r["time"], device=names.get(k, k), mbs=v.get("mb_s"), files=v.get("files_done"), bad=v.get("md5_bad"),
                                   errors=v.get("errors"), note="automatic probe, 16 streams for 60 s"))
            else:
                probes.append(dict(t=r["time"], device=names.get(k, k), mbs=None, files=None, bad=None, errors=None, note=str(v)[:160]))
    seen = {(r["device"], r["mbs"], r["files"]) for r in probes}                  # a hand row that repeats an automatic one is dropped
    probes += [r for r in hand.get("probes", []) if (r["device"], r["mbs"], r["files"]) not in seen]
    probes.sort(key=lambda r: r["t"], reverse=True)
    return dict(dl_hist=dl_hist, dl_now=dl_now, dl_xfer=dl_xfer, dl_probes=probes[:40], dl_streams=dl_streams[:60])


if __name__ == "__main__":
    for k, v in rows().items():
        print(k, len(v)); [print("  ", json.dumps(x)) for x in v[-6:]]
