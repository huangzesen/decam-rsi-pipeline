#!/usr/bin/env python3
"""Bulk download of DECam InstCal files from the NOIRLab Astro Data Archive, from a manifest written by
make_manifest.py (md5, size, name, tag per line, in the order the images are needed).

Resumable: a file is skipped when a done log names it or it is on disk with the right size. Each file streams to
NAME.part with its md5 computed on the way; it is renamed only when size and md5 match, else retried (the archive
returns about 1 % damaged copies). Files land in DEST/img/<YYMMDD>/, logs in DEST/log/.

Several copies can share one manifest: --part k/N gives this copy every N-th block of 32 exposures, so all copies
move down the manifest together. Copies that run at the same time must use the same N.

It stops starting files when: --hours is used up, DEST/log/STOP exists, --max-tb is reached, the manifest is done,
or the archive keeps refusing (more than half of the last 200 files failed, three times in a row: exit code 3).

Device-wide control (2026-11-20): if the JSON file CTL (env FETCH_CTL, default
fetch_ctl.json beside this script) exists, every fetch.py of the device registers in CTL.d/ and runs only its share of
CTL's "total" streams (at most --streams), and starts no file before CTL's "pause_until" (epoch). Read every 30 s, so
changes apply to running jobs; fill/dl_watch.py writes it. Without the file: --streams, as before.

Usage: python3 -I fetch.py MANIFEST.tsv DEST [--part 0/4] [--streams 32] [--hours 47] [--max-tb 0]
Python 3.9, standard library only.
"""
import argparse, collections, glob, hashlib, json, os, signal, socket, sys, threading, time, urllib.error, urllib.request

API = "https://astroarchive.noirlab.edu/api/retrieve/"
ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("dest")
ap.add_argument("--part", default="0/1")
ap.add_argument("--streams", type=int, default=32)
ap.add_argument("--hours", type=float, default=47.0)
ap.add_argument("--max-tb", type=float, default=0.0)
ap.add_argument("--tries", type=int, default=4)
a = ap.parse_args()
K, N = (int(x) for x in a.part.split("/"))
tagp = f"{K}of{N}"
IMG, LOG = os.path.join(a.dest, "img"), os.path.join(a.dest, "log")
os.makedirs(IMG, exist_ok=True); os.makedirs(LOG, exist_ok=True)
t0 = time.time()


def say(*x):
    print(time.strftime("%m-%d %H:%M:%S"), *x, flush=True)


# this copy's share of the manifest: blocks of 32 exposures
rows = []; nexp = -1; last = None
for line in open(a.manifest):
    md5, size, name, tag = line.rstrip("\n").split("\t")
    if name[:17] != last:
        last = name[:17]; nexp += 1
    if (nexp // 32) % N == K:
        rows.append((md5, int(size), name))
done = set()
for f in glob.glob(os.path.join(LOG, "done_*.txt")):
    done.update(x.strip() for x in open(f))
skipf = os.path.splitext(a.manifest)[0] + ".skip.txt"     # names another job fetches first (2026-11-18: ldt5), left out here
skip = set(x.strip() for x in open(skipf)) if os.path.exists(skipf) else set()
nfail = collections.Counter()           # 2026-11-21: files that failed all tries in 2 earlier runs are left out; on the Mac 16
for f in glob.glob(os.path.join(LOG, "failed_*.txt")):   # such files were retried at the start of every hourly chunk (64 bad
    nfail.update(x.split("\t")[0].strip() for x in open(f) if x.strip())   # md5, 16.7 GB) and tripped dl_watch's limit
gaveup = {n for n, k in nfail.items() if k >= 2}
todo = [r for r in rows if r[2] not in done and r[2] not in skip and r[2] not in gaveup]
todo.sort(key=lambda r: nfail.get(r[2], 0) > 0)     # files that failed once go last (stable), so a chunk does not open
                                                     # on retries (2026-11-21: they made dl_watch see a bad-md5 burst)
if gaveup:
    say(f"{len([r for r in rows if r[2] in gaveup and r[2] not in done])} files left out: failed every try in >= 2 earlier runs (log/failed_*.txt)")
say(f"part {a.part} on {socket.gethostname()}: {len(rows)} files, {sum(r[1] for r in rows) / 1e12:.2f} TB in this share; "
    f"{len(todo)} files, {sum(r[1] for r in todo) / 1e12:.2f} TB still to fetch; {a.streams} streams, "
    f"{len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'} cpus allowed")

lock = threading.Lock()
it = iter(todo)
stop = threading.Event(); why = [""]
S = dict(bytes=0, files=0, skipped=0, bad_md5=0, errors=0, failed=0, pauses=0, short=0, wasted=0, md5diff=0, damaged=0)
recent = collections.deque(maxlen=200)          # True = a file attempt failed
pause_until = [0.0]; strikes = [0]
errs = collections.Counter()
done_f = open(os.path.join(LOG, f"done_{tagp}.txt"), "a")
fail_f = open(os.path.join(LOG, f"failed_{tagp}.txt"), "a")
diff_f = open(os.path.join(LOG, f"md5diff_{tagp}.txt"), "a")
dmg_f = open(os.path.join(LOG, f"damaged_{tagp}.txt"), "a")   # damaged in the archive itself: kept once, bad HDUs listed   # kept although the listed md5 differs: data checksums all pass


CTL = os.environ.get("FETCH_CTL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_ctl.json")
HB = CTL + ".d"; ME = f"{socket.gethostname()}-{os.getpid()}"
ctl = dict(allow=a.streams, pause_until=0.0, total=None)
exhausted = threading.Event()


def ctl_read():
    try:
        c = json.load(open(CTL))
    except (OSError, ValueError):
        ctl.update(allow=a.streams, pause_until=0.0, total=None); return
    # weights (2026-11-21): fetch_ctl.json.weights = {"substring of the manifest path": weight}; the device total is split
    # in proportion (largest remainder, a process may get 0 and wait), so units close to done are not starved by a new
    # unit's many shares (u17 had 1 of 10 streams beside u20's 7 processes). No file or no match: weight 1 = equal split.
    try:
        W = json.load(open(CTL + ".weights"))
    except (OSError, ValueError):
        W = {}
    mw = max([float(v) for k, v in W.items() if k in os.path.abspath(a.manifest)] or [1.0])
    live = {ME: mw}
    try:
        os.makedirs(HB, exist_ok=True); now = time.time()
        open(os.path.join(HB, ME), "w").write(f"{now:.0f} {mw}\n")
        live = {}
        for f in os.listdir(HB):
            p = os.path.join(HB, f); age = now - os.path.getmtime(p)
            if age < 150:
                try:
                    parts = open(p).read().split(); live[f] = float(parts[1]) if len(parts) > 1 else 1.0
                except (OSError, ValueError):
                    live[f] = 1.0
            elif age > 86400:
                os.remove(p)
        live.setdefault(ME, mw)
    except OSError:
        pass
    tot = max(0, int(c.get("total", a.streams))); sw = sum(live.values()) or 1.0
    q = {k: tot * w / sw for k, w in live.items()}; share = {k: int(v) for k, v in q.items()}
    for k in sorted(q, key=lambda k: (-(q[k] - share[k]), k))[:tot - sum(share.values())]:
        share[k] += 1
    ctl.update(allow=min(a.streams, share.get(ME, 0)), pause_until=float(c.get("pause_until", 0) or 0), total=tot)


def ctl_loop():
    while not stop.is_set():
        try:
            ctl_read()
        except Exception as e:
            say("control file unreadable:", type(e).__name__, str(e)[:80])
        stop.wait(30)


def held(i=None):
    """True while this thread must not start a file: over the device's share, or in a cool-down"""
    return time.time() < ctl["pause_until"] or (i is not None and i >= ctl["allow"])


def halt(reason):
    if not stop.is_set():
        why[0] = reason; stop.set(); say("stopping:", reason)


signal.signal(signal.SIGTERM, lambda *_: halt("SIGTERM"))


def bad_hdus(path):
    """Full-size file whose md5 differs from the archive's listing. Returns the HDUs whose data do not verify
    (DATASUM missing or wrong; "only N HDUs" when the file parses into fewer than 59), [] when all verify, None when
    the file cannot be opened. 2026-11-17: the archive edits FITS headers after listing some files (stale md5, data
    fine: keep), and its own copies of some files are damaged (the same wrong bytes on every download: keep once,
    with the bad HDUs listed in log/damaged_*.txt)."""
    try:
        import re, warnings
        from astropy.io import fits
        bad = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with fits.open(path, checksum=True, disable_image_compression=True, memmap=False) as h:
                n = len(h)
                for i, x in enumerate(h):
                    if x.header.get("NAXIS", 0):
                        if "DATASUM" not in x.header:
                            bad.append(str(x.header.get("EXTNAME", i)))
                        _ = x.data
        for m in w:
            r = re.search(r"Datasum verification failed for HDU \('([^']*)'", str(m.message))
            if r:
                bad.append(r.group(1))
        if n < 59:
            bad.append(f"only {n} HDUs")
        return sorted(set(bad))
    except Exception:
        return None


def mark(ok, name=None):
    with lock:
        recent.append(not ok)
        if ok:
            strikes[0] = 0
        elif len(recent) >= 100 and sum(recent) > len(recent) / 2 and time.time() >= pause_until[0]:
            strikes[0] += 1; S["pauses"] += 1; recent.clear()
            if strikes[0] >= 3:
                halt("the archive keeps refusing or failing")
            else:
                pause_until[0] = time.time() + 600
                say(f"more than half of the last files failed: pausing 10 min (strike {strikes[0]})")


def fetch(md5, size, name):
    d = os.path.join(IMG, name[4:10]); dst = os.path.join(d, name); tmp = dst + ".part"
    try:
        if os.path.getsize(dst) == size:
            with lock:
                S["skipped"] += 1; done_f.write(name + "\n")
            return
    except OSError:
        pass
    os.makedirs(d, exist_ok=True)
    prev = [None]                       # served md5 of the last full-size attempt that failed
    for k in range(a.tries):
        if stop.is_set():
            return
        while (time.time() < pause_until[0] or held()) and not stop.is_set():
            time.sleep(5)
        h = hashlib.md5(); got = 0
        try:
            with urllib.request.urlopen(API + md5 + "/", timeout=180) as r, open(tmp, "wb") as o:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    h.update(b); o.write(b); got += len(b)
                    with lock:
                        S["bytes"] += len(b)
                    if stop.is_set() and why[0] == "SIGTERM":
                        return
            if got == size and h.hexdigest() == md5:
                os.replace(tmp, dst)
                with lock:
                    S["files"] += 1; done_f.write(name + "\n"); done_f.flush()
                mark(True)
                return
            dig = h.hexdigest()
            bad = bad_hdus(tmp) if got == size else None
            if bad == []:
                os.replace(tmp, dst)
                with lock:
                    S["files"] += 1; S["md5diff"] += 1; done_f.write(name + "\n"); done_f.flush()
                    diff_f.write(f"{name}\t{md5}\t{dig}\n"); diff_f.flush()
                mark(True)
                return
            if bad and dig == prev[0]:          # the same damaged bytes twice: the archive's copy is damaged
                os.replace(tmp, dst)
                with lock:
                    S["files"] += 1; S["damaged"] += 1; done_f.write(name + "\n"); done_f.flush()
                    dmg_f.write(f"{name}\t{md5}\t{dig}\t{','.join(map(str, bad))}\n"); dmg_f.flush()
                return
            prev[0] = dig if got == size else None
            with lock:
                S["bad_md5"] += 1; S["wasted"] += got
                if got < size: S["short"] += 1      # cut off before the end (counted in bad md5 too)
            mark(False)
        except Exception as e:
            code = getattr(e, "code", None)
            with lock:
                S["errors"] += 1; errs[f"{type(e).__name__} {code or ''} {str(e)[:60]}"] += 1
            mark(False)
            if code in (403, 404):      # not public or gone: trying again will not help
                break
            time.sleep(min(60, 5 * 2 ** k))
    try:
        os.remove(tmp)
    except OSError:
        pass
    with lock:
        S["failed"] += 1; fail_f.write(f"{name}\t{md5}\n"); fail_f.flush()


def worker(i):
    while not stop.is_set():
        while held(i) and not stop.is_set() and not exhausted.is_set():
            time.sleep(5)
        if stop.is_set() or exhausted.is_set():
            return
        with lock:
            r = next(it, None)
        if r is None:
            exhausted.set(); return
        fetch(*r)


ctl_read()
if ctl["total"] is not None:
    say(f"control {CTL}: device total {ctl['total']} streams, this copy {ctl['allow']} of at most {a.streams}")
threading.Thread(target=ctl_loop, daemon=True).start()
th = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.streams)]
for t in th:
    t.start()
prev = (time.time(), 0)
while any(t.is_alive() for t in th):
    time.sleep(60)
    now = time.time()
    with lock:
        s = dict(S); top = errs.most_common(3)
    rate = (s["bytes"] - prev[1]) / 1e6 / (now - prev[0]); prev = (now, s["bytes"])
    st = dict(part=a.part, host=socket.gethostname(), hours=round((now - t0) / 3600, 2), mb_s_last_min=round(rate, 1),
              mb_s_run=round(s["bytes"] / 1e6 / (now - t0), 1), tb_run=round(s["bytes"] / 1e12, 3), **s,
              files_left=len(todo) - s["files"] - s["skipped"] - s["failed"], top_errors=top, time=time.strftime("%Y-%m-%d %H:%M:%S"),
              stopping=why[0])
    json.dump(st, open(os.path.join(LOG, f"status_{tagp}.json.tmp"), "w")); os.replace(os.path.join(LOG, f"status_{tagp}.json.tmp"),
                                                                                    os.path.join(LOG, f"status_{tagp}.json"))
    say(f"{rate:.0f} MB/s now, {st['mb_s_run']:.0f} MB/s run, {st['tb_run']:.2f} TB, files {s['files']}, bad md5 {s['bad_md5']}, "
        f"errors {s['errors']}, failed {s['failed']}, left {st['files_left']}, short {s['short']}, wasted {s['wasted'] / 1e9:.1f} GB, kept with stale md5 {s['md5diff']}, damaged at archive {s['damaged']}, "
        f"streams {ctl['allow']}" + (f", cool-down until {time.strftime('%H:%M', time.localtime(ctl['pause_until']))}" if now < ctl['pause_until'] else ""))
    if os.path.exists(os.path.join(LOG, "STOP")):
        halt("STOP file")
    elif now - t0 > a.hours * 3600:
        halt("time budget used")
    elif a.max_tb and s["bytes"] / 1e12 >= a.max_tb:
        halt("--max-tb reached")
    if stop.is_set() and why[0] == "SIGTERM":
        break
try:
    os.remove(os.path.join(HB, ME))
except OSError:
    pass
say(f"end: {S}; {why[0] or 'share finished'}; top errors {errs.most_common(5)}")
sys.exit(3 if why[0].startswith("the archive") else 0)
