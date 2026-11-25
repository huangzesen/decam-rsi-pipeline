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
todo = [r for r in rows if r[2] not in done]
say(f"part {a.part} on {socket.gethostname()}: {len(rows)} files, {sum(r[1] for r in rows) / 1e12:.2f} TB in this share; "
    f"{len(todo)} files, {sum(r[1] for r in todo) / 1e12:.2f} TB still to fetch; {a.streams} streams, "
    f"{len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'} cpus allowed")

lock = threading.Lock()
it = iter(todo)
stop = threading.Event(); why = [""]
S = dict(bytes=0, files=0, skipped=0, bad_md5=0, errors=0, failed=0, pauses=0)
recent = collections.deque(maxlen=200)          # True = a file attempt failed
pause_until = [0.0]; strikes = [0]
errs = collections.Counter()
done_f = open(os.path.join(LOG, f"done_{tagp}.txt"), "a")
fail_f = open(os.path.join(LOG, f"failed_{tagp}.txt"), "a")


def halt(reason):
    if not stop.is_set():
        why[0] = reason; stop.set(); say("stopping:", reason)


signal.signal(signal.SIGTERM, lambda *_: halt("SIGTERM"))


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
    for k in range(a.tries):
        if stop.is_set():
            return
        while time.time() < pause_until[0] and not stop.is_set():
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
            with lock:
                S["bad_md5"] += 1
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


def worker():
    while not stop.is_set():
        with lock:
            r = next(it, None)
        if r is None:
            return
        fetch(*r)


th = [threading.Thread(target=worker, daemon=True) for _ in range(a.streams)]
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
        f"errors {s['errors']}, failed {s['failed']}, left {st['files_left']}")
    if os.path.exists(os.path.join(LOG, "STOP")):
        halt("STOP file")
    elif now - t0 > a.hours * 3600:
        halt("time budget used")
    elif a.max_tb and s["bytes"] / 1e12 >= a.max_tb:
        halt("--max-tb reached")
    if stop.is_set() and why[0] == "SIGTERM":
        break
say(f"end: {S}; {why[0] or 'share finished'}; top errors {errs.most_common(5)}")
sys.exit(3 if why[0].startswith("the archive") else 0)
