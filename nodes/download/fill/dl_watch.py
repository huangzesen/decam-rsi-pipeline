#!/usr/bin/env python3
"""Download monitor and stream controller (2026-11-20).

Every cycle (default 10 min), for each device (NERSC, Lonestar6, Vista, Mac):
  downloads from NOIRLab: the last line of every fetch.py log written in the last 6 min: summed "MB/s now", streams in
    use, and the change since the previous cycle in files done, bad md5 and errors (refused/failed connections), counted
    only on logs seen in both cycles (a new job's first lines are its baseline);
  stream control: each device has a budget of streams for all its fetch.py copies together, in fetch_ctl.json beside
    its fetch.py (fetch.py reads it every 30 s and splits it among the copies running there). The device is LIMITED when,
    in a cycle judged after its last cool-down ended: errors >= 10 and more errors than files, or bad md5 on more than
    25 % of >= 8 files. Then its budget drops by one stream (not below 1) and it cools down for 30 min (no file starts;
    files in flight finish). Since 11-21 00:30 a device clean for 3 cycles gets +1 while all together are under 200 MB/s (cap 10; for 3 h after a
    limit not back to the limited budget).
    Start budgets: NERSC 9, Lonestar6 8, Vista 8, Mac 8. Events in dl_watch.json "ctl" and the log.
  transfers between devices: Mac -> NERSC rsync batches (relay_mac/relay.log), and the Globus tasks of the NERSC -> Vista,
    Lonestar6 -> Vista, Lonestar6 -> NERSC relays and the conveyor's Vista -> NERSC pulls (effective rate).
When a device is limited, the idle devices (no fetch running, not cooling) are probed at once with
rsi/candidates/dltest/dltest.py (4 streams, 45 s; files md5-checked and deleted). Nothing is started or stopped here
besides the budgets and cool-downs: the session reads the LIMITED and PROBE lines and decides where work goes.
Output: one line per cycle on stdout (fill/dl_watch.log), history in fill/dl_watch.json (for the DECam data page).
Usage: python3 -I fill/dl_watch.py loop [MINUTES=10] | once        stop: touch fill/dl_watch.STOP
Change a budget by hand: python3 -I fill/dl_watch.py set DEVICE TOTAL (also ends its cool-down)"""
import json, os, re, subprocess, sys, time
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "dl_watch.json")
SOCK = {"nersc": "$SSH_CM_DIR/nersc-$NERSC_USER.sock", "ls6": "$SSH_CM_DIR/ls6-$TACC_USER.sock", "vista": "$SSH_CM_DIR/vista-$TACC_USER.sock"}
HOST = {"nersc": "$NERSC_USER@perlmutter.nersc.gov", "ls6": "x", "vista": "x"}
LOGS = {"nersc": "$NERSC_SCRATCH/decam-fill/p*/log/fetch_*.log $NERSC_SCRATCH/units/u*/p*/log/fetch*.log",
        "ls6": "$TACC_SCRATCH/decam-fill-ls6/log/fetch_*.log $TACC_SCRATCH/units-ls6/log/fetch_*.log "
               "$TACC_SCRATCH/units-ls6/u*/log/fetch_*.log",
        "vista": "$TACC_SCRATCH/decam-fill/log/fetch_*.log"}
MAC_LOG = "$DATA_DRIVE/decam-fill-mac/log/fetch_mac.log"
CTL = {"nersc": ["$NERSC_SCRATCH/decam-fill/fetch_ctl.json"],
       "ls6": ["$TACC_SCRATCH/units-ls6/fetch_ctl.json", "$TACC_SCRATCH/decam-fill-ls6/fetch_ctl.json"],
       "vista": ["$TACC_SCRATCH/decam-fill/fetch_ctl.json"],
       "mac": [os.path.join(H, "fetch_ctl.json")]}
START = {"nersc": 9, "ls6": 8, "vista": 8, "mac": 8}
COOL_S = 600                                       # 1800 until 2026-11-22 18:3x
# raising (user 11-21 00:30: "why download speed is down...?? you should keep monitoring and keep that up!"): a device
# that ran clean for RAISE_AFTER judged cycles while all devices together are under GOAL_MBS gets +1 stream, never above
# MAX_STREAMS (the user's 5-10 per device) and, for HOLD_S after a limit, never back to the budget that was limited.
GOAL_MBS, MAX_STREAMS, RAISE_AFTER, HOLD_S = 400, 15, 3, 3 * 3600   # cap 10 and goal 200 until 2026-11-22 18:3x
PROBE_DIR = {"nersc": "$NERSC_SCRATCH/probe", "ls6": "$TACC_SCRATCH/probe"}
LINE = re.compile(r"(\d+) MB/s now.*?files (\d+), bad md5 (\d+), errors (\d+)")
ENV = dict(os.environ, LC_ALL="C", LANG="C")
SITES = ("nersc", "ls6", "vista", "mac")


def ssh(site, cmd, timeout=300, inp=None):
    p = subprocess.run(["ssh", "-S", os.path.expanduser(SOCK[site]), "-o", "ControlMaster=no", "-o", "BatchMode=yes", HOST[site],
                        "LC_ALL=C; " + cmd], input=inp, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout


def write_ctl(site, total, pause_until):
    body = json.dumps(dict(total=total, pause_until=round(pause_until), set=time.strftime("%Y-%m-%d %H:%M"),
                           note="written by fill/dl_watch.py: device total streams; on each limit -1 and a 10 min cool-down")) + "\n"
    for path in CTL[site]:
        if site == "mac":
            open(path + ".tmp", "w").write(body); os.replace(path + ".tmp", path)
        else:
            rc, _ = ssh(site, f"[ -d {os.path.dirname(path)} ] && cat > {path}.tmp && mv {path}.tmp {path}", timeout=120, inp=body)
            if rc != 0:
                return False
    return True


def remote_lines(site):
    """{log path: last line} of the device's fetch logs written in the last 6 min; None if the device is unreachable"""
    if not os.path.exists(os.path.expanduser(SOCK[site])):
        return None
    rc, o = ssh(site, f"for f in $(find {LOGS[site]} -mmin -6 2>/dev/null); do echo \"$f\t$(tail -1 $f)\"; done")
    if rc != 0:
        return None
    return dict(l.split("\t", 1) for l in o.splitlines() if "\t" in l)


def _mac_lines():
    if os.path.exists(MAC_LOG) and time.time() - os.path.getmtime(MAC_LOG) < 360:
        with open(MAC_LOG) as f:
            last = [l for l in f.read().splitlines()[-5:] if "MB/s now" in l]
        return {MAC_LOG: last[-1]} if last else {}
    return {}


def mac_lines():
    """the Mac's log lives on Crucial-8TB; a hung drive (2026-11-23 18:56) froze the whole controller in this read.
    Read it in a daemon thread with a 15 s limit: no answer or an error counts as no Mac log (Mac idle), and the
    other devices are managed as usual."""
    import threading
    box = {}
    def run():
        try:
            box["v"] = _mac_lines()
        except Exception as e:
            box["e"] = e
    th = threading.Thread(target=run, daemon=True); th.start(); th.join(15)
    if "v" in box:
        return box["v"]
    say(f"mac log unreadable ({'timeout 15 s' if th.is_alive() else type(box.get('e')).__name__}): Mac counted idle this cycle")
    return {}


def gl_rate(task):
    """MB/s of a Globus task (effective rate) and its status"""
    p = subprocess.run(["globus", "task", "show", task, "--format", "json"], capture_output=True, text=True, env=ENV, timeout=120)
    if p.returncode != 0:
        return None, None
    d = json.loads(p.stdout)
    return (d.get("effective_bytes_per_second") or 0) / 1e6, d.get("status")


def transfers(since):
    """route -> MB/s over tasks or batches active since `since` (epoch)"""
    out = {}
    lg = os.path.join(H, "relay_mac", "relay.log"); gb = s = 0.0
    if os.path.exists(lg):
        for l in open(lg).read().splitlines()[-60:]:
            m = re.match(r"(\d\d-\d\d \d\d:\d\d:\d\d) .* files at NERSC, ([\d.]+) GB in (\d+) s", l)
            if m and time.mktime(time.strptime(time.strftime("%Y-") + m.group(1), "%Y-%m-%d %H:%M:%S")) >= since:
                gb += float(m.group(2)); s += float(m.group(3))
    out["mac->nersc"] = round(gb * 1e3 / s, 1) if s else None
    for route, folder in (("nersc->vista", "relay"), ("ls6->vista", "relay_ls6"), ("ls6->nersc", "relay_units_ls6")):
        d = os.path.join(H, folder, "tasks"); rates = []
        if os.path.isdir(d):
            for f in sorted(os.listdir(d))[-6:]:
                p = os.path.join(d, f)
                if os.path.getmtime(p) < since and json.load(open(p)).get("closed"):
                    continue
                r, st = gl_rate(json.load(open(p))["task"])
                if r:
                    rates.append(r)
        out[route] = round(sum(rates) / len(rates), 1) if rates else None
    try:
        st = json.load(open(os.path.join(os.path.dirname(H), "conveyor", "state.json")))
        rates = [gl_rate(u["pull_task"])[0] for u in st["units"] if u.get("stage") == "prep" and u.get("pull_task")]
        rates = [r for r in rates if r]
        out["vista->nersc"] = round(sum(rates) / len(rates), 1) if rates else None
    except Exception:
        out["vista->nersc"] = None
    return out


def probe(site):
    """45 s at 4 streams from the device's login node (or the Mac); returns the dltest PHASE dict or an error text"""
    T = os.path.join(os.path.dirname(H), "rsi", "candidates", "dltest")
    try:
        if site == "mac":
            tmp = "$DATA_DRIVE/decam-fill-mac/probe_tmp"
            p = subprocess.run(["python3", "-I", os.path.join(T, "dltest.py"), os.path.join(T, "files.json"), tmp, "4", "45"],
                               capture_output=True, text=True, timeout=240)
            o = p.stdout
        else:
            rc, o = ssh(site, f"cd {PROBE_DIR[site]} && timeout 200 nice python3 dltest.py files.json {PROBE_DIR[site]}/tmp \"4\" 45 2>&1",
                        timeout=260)
        ph = [l for l in o.splitlines() if l.startswith("PHASE ")]
        return json.loads(ph[-1][6:]) if ph else "no result: " + o[-200:]
    except Exception as e:
        return f"probe failed: {type(e).__name__} {str(e)[:120]}"


def why_limited(v):
    f, b, e = v["files"], v["bad"], v["errors"]
    if e >= 10 and e > f:
        return f"errors {e} > files {f}"
    # 2026-11-21: files given up after all tries, not bad tries: a few files whose archive copy never matches its listed
    # md5 (2013 Community Pipeline v1) are tried 4 times each and made bad tries look like a refusing archive
    fl = v.get("failed", 0)
    if f + fl >= 8 and fl > 0.25 * (f + fl):
        return f"files given up {fl} of {f + fl} (bad md5 tries {b})"
    return None


def ctl_init(state):
    C = state.setdefault("ctl", {})
    for s in SITES:
        if s not in C:
            C[s] = dict(total=START[s], pause_until=0, events=[])
    return C


def say(*x):
    print(time.strftime("%m-%d %H:%M"), *x, flush=True)


def cycle(state):
    now = time.time(); prev = state.get("logs", {}); since = state.get("t", now - 600)
    try:                                   # budgets set by hand (dl_watch.py set) while the loop runs
        state["ctl"] = json.load(open(OUT)).get("ctl", state.get("ctl", {}))
    except (OSError, ValueError):
        pass
    C = ctl_init(state); sites = {}; newlog = {}
    for site in SITES:
        L = mac_lines() if site == "mac" else remote_lines(site)
        c = C[site]; cooling = now < c["pause_until"]
        if L is None:
            sites[site] = dict(reach=False, total=c["total"], state="unreachable"); continue
        mbs = files = bad = err = used = n = failed = 0
        for path, line in L.items():
            m = LINE.search(line)
            if not m:
                continue
            r, f, b, e = map(int, m.groups()); key = site + ":" + path
            mf = re.search(r"failed (\d+)", line); fl = int(mf.group(1)) if mf else 0
            ms = re.search(r"streams (\d+)", line); used += int(ms.group(1)) if ms else 0
            mbs += r; n += 1; newlog[key] = (f, b, e, fl)
            if key in prev:
                pf, pb, pe, pfl = (list(prev[key]) + [fl])[:4]     # no stored count: no change
                if f < pf or b < pb or e < pe:          # a new chunk or job wrote this log from the start
                    pf = pb = pe = pfl = 0
                files += f - pf; bad += b - pb; err += e - pe; failed += fl - pfl
        v = dict(reach=True, logs=n, mbs=mbs, files=files, bad=bad, failed=failed, errors=err, used=used, total=c["total"])
        v["state"] = "cooling" if cooling else ("downloading" if n else "idle")
        judged = n > 0 and not cooling and now >= c["pause_until"] + 540 and now - since >= 300   # not on a short window (restart)
        lim = why_limited(v) if judged else None
        if judged:
            c["clean"] = 0 if lim else c.get("clean", 0) + (1 if v["files"] > 0 else 0)
        if lim:
            old = c["total"]; c["total"] = max(1, old - 1); c["pause_until"] = now + COOL_S
            c["limited_at"], c["limited_t"] = old, now
            ok = write_ctl(site, c["total"], c["pause_until"])
            c["events"].append(dict(time=time.strftime("%Y-%m-%d %H:%M"), why=lim, streams=[old, c["total"]],
                                    until=time.strftime("%H:%M", time.localtime(c["pause_until"])), written=ok))
            v.update(state="cooling", limited=lim, total=c["total"])
        v["pause_until"] = c["pause_until"] if now < c["pause_until"] else 0
        sites[site] = v
    state["logs"] = newlog
    together = sum(v.get("mbs", 0) for v in sites.values() if v.get("state") == "downloading")
    raised = []
    for site, v in sites.items():
        c = C[site]
        if v.get("state") != "downloading" or together >= GOAL_MBS or c.get("clean", 0) < RAISE_AFTER:
            continue
        cap = MAX_STREAMS
        if now - c.get("limited_t", 0) < HOLD_S:
            cap = min(cap, c.get("limited_at", MAX_STREAMS + 1) - 1)
        if c["total"] >= cap or v.get("used", 0) < c["total"]:         # at the cap, or not using what it has
            continue
        old = c["total"]; c["total"] = old + 1; c["clean"] = 0
        ok = write_ctl(site, c["total"], 0)
        c["events"].append(dict(time=time.strftime("%Y-%m-%d %H:%M"), why=f"clean {RAISE_AFTER} cycles, together {together} MB/s",
                                streams=[old, c["total"]], until="", written=ok))
        v["total"] = c["total"]; raised.append((site, old, c["total"], ok))
    tr = transfers(since)
    rec = dict(t=now, time=time.strftime("%Y-%m-%d %H:%M"), sites=sites, transfers=tr)
    hit = [s for s, v in sites.items() if v.get("limited")]
    if hit:
        idle = [s for s in ("nersc", "ls6", "mac") if sites.get(s, {}).get("state") == "idle"]
        rec["probes"] = {s: probe(s) for s in idle}
    state["last"] = sites; state["t"] = now
    hist = state.setdefault("history", []); hist.append(rec); del hist[:-432]          # 3 days at 10 min
    json.dump(state, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)

    def fmt(s, v):
        if not v.get("reach"):
            return f"{s} unreachable"
        if v["state"] == "cooling":
            return f"{s} cooling to {time.strftime('%H:%M', time.localtime(v['pause_until']))}, budget {v['total']}"
        if v["state"] == "idle":
            return f"{s} idle, budget {v['total']}"
        return (f"{s} {v['mbs']} MB/s, {v['used']}/{v['total']} streams ({v['logs']} logs; +{v['files']} files, "
                f"bad {v['bad']}, errors {v['errors']})")
    say(" | ".join(fmt(s, v) for s, v in sites.items()), "|| transfers MB/s:",
        ", ".join(f"{k} {v}" for k, v in tr.items() if v is not None) or "none")
    for s in hit:
        e = C[s]["events"][-1]
        say(f"LIMITED {s}: {e['why']} -> cool down {COOL_S // 60} min to {e['until']}, streams {e['streams'][0]} -> {e['streams'][1]}"
            + ("" if e["written"] else " (CONTROL FILE NOT WRITTEN)"))
    for s, old, new, ok in raised:
        say(f"RAISED {s}: clean {RAISE_AFTER} cycles, together {together} MB/s < {GOAL_MBS} -> streams {old} -> {new}"
            + ("" if ok else " (CONTROL FILE NOT WRITTEN)"))
    for s, r in (rec.get("probes") or {}).items():
        if isinstance(r, dict):
            say(f"PROBE {s}: {r['mb_s']} MB/s, {r['files_done']} files, bad md5 {r['md5_bad']}, errors {r['errors']}")
        else:
            say(f"PROBE {s}: {r}")


if __name__ == "__main__":
    state = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if sys.argv[1:2] == ["set"]:
        site, tot = sys.argv[2], int(sys.argv[3]); C = ctl_init(state)
        C[site].update(total=tot, pause_until=0, clean=0); C[site].pop("limited_at", None); C[site].pop("limited_t", None)
        C[site]["events"].append(dict(time=time.strftime("%Y-%m-%d %H:%M"), why="set by hand", streams=[None, tot], until="", written=write_ctl(site, tot, 0)))
        json.dump(state, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT); print(site, C[site]["events"][-1])
    elif sys.argv[1:2] == ["once"]:
        cycle(state)
    else:
        m = float(sys.argv[2]) if len(sys.argv) > 2 else 10
        while not os.path.exists(os.path.join(H, "dl_watch.STOP")):
            try:
                cycle(state)
            except Exception as e:
                say("cycle error:", type(e).__name__, str(e)[:200])
            time.sleep(m * 60)
