"""Steps 1 and 2 of a slice: download its images and run its program on them.

A slice is a folder that holds everything it needs (see its README.md). For each of its windows, on this node:
  1. download  the window's InstCal files (image, weight map, mask per exposure) from the NOIRLab archive, md5 checked
               while streaming, 4 tries; one window ahead of each GPU. A file listed in --local-index is linked from
               disk instead of downloaded;
     build     the scenario folder the finder reads (exposures.csv with times and observer positions, meta.json,
               exposures/), in the RSI task's format, every exposure visible;
  2. run       the slice's program, one window per GPU:  finder.py --scenario S --out OUT.json --budget-s SECONDS
               in a fresh copy, without network when the node allows it, killed 30 s after the budget;
     measure   while the images are still here: forced photometry of the known bodies at their predicted places
               (private/known.json; the finder never sees it) and along the tracks of the window's best candidates,
               with the same-spot test on other nights. The reward and the lead screen (slice_score.py) use these;
     keep      WORKDIR/out/<sid>/{candidates.json, finder.log, info.json, exposures.csv, meta.json, known_phot.json,
               cand_phot.json, done.json};
     delete    the window's images (unless --keep-images).
Windows with done.json are skipped, so a rerun resumes. Writes WORKDIR/run.json at the end.

Usage: python -I lib/slice_run.py SLICE_DIR WORKDIR [--dry-run] [--local-index FILE] [--gpus N] [--keep-images]
       [--redo] [--windows KEY,KEY] [--program DIR] [--threads N]
--dry-run prints what would be fetched and run, checks the folder and the disk, and touches nothing.
"""
import argparse, csv, hashlib, json, os, queue, shutil, signal, socket, subprocess, sys, tempfile, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

API = "https://astroarchive.noirlab.edu/api"
COLS = ["exp_id", "image", "weight", "mask", "mjd_start", "exptime", "mjd_mid", "band", "airmass", "magzero",
        "seeing_arcsec", "ra_center", "dec_center", "obs_x", "obs_y", "obs_z"]
CAND_PHOT_TOP = 50          # candidates per window measured on the images (highest p, outside 25 au)
LOCK = threading.Lock()


def log(*a):
    with LOCK:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def budget_for(settings, n_exp):
    b = settings.get("budget_s", 900)
    if isinstance(b, (int, float)):
        return float(b)
    for nmax, sec in b:
        if n_exp <= nmax:
            return float(sec)
    return float(b[-1][1])


def fetch_file(name, md5, size, d, local):
    dst = os.path.join(d, name)
    if os.path.lexists(dst) and os.path.exists(dst) and os.path.getsize(dst) == size:
        return "skip", 0
    if name in local and os.path.exists(local[name]) and os.path.getsize(local[name]) == size:
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(local[name], dst)
        return "link", 0
    part = dst + ".part"
    for k in range(4):
        try:
            h = hashlib.md5()
            with urllib.request.urlopen(f"{API}/retrieve/{md5}/", timeout=600) as r, open(part, "wb") as o:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    h.update(b); o.write(b)
            if h.hexdigest() != md5:
                raise IOError("md5 mismatch")
            os.replace(part, dst)
            return "ok", size
        except Exception as e:
            log("  retry", name, k, repr(e)[:80]); time.sleep(15 * (k + 1))
    return "FAILED", 0


def observer(mjd):
    """Heliocentric ICRS equatorial position of DECam (au), as in rsi/bin/make_scenarios.py."""
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, get_body_barycentric, solar_system_ephemeris
    import astropy.units as u
    from astropy.utils import iers
    iers.conf.auto_download = False                    # compute nodes may have no network; UT1 to 0.9 s is 0.4 km
    iers.conf.iers_degraded_accuracy = "warn"
    site = EarthLocation(lat=-30.16606 * u.deg, lon=-70.81489 * u.deg, height=2215 * u.m)
    t = Time(mjd, format="mjd", scale="utc")
    with solar_system_ephemeris.set("builtin"):
        e = get_body_barycentric("earth", t).xyz.to(u.au).value
        s = get_body_barycentric("sun", t).xyz.to(u.au).value
    return e - s + site.get_gcrs_posvel(t)[0].xyz.to(u.au).value


def build_scenario(w, names, data, scen):
    """names: stamp -> {i, w, d: file name}. Returns the exposure rows (image paths relative to the scenario)."""
    import numpy as np
    from astropy.io import fits
    from common import PIX
    rows = []
    for st in sorted(names):
        g = names[st]
        with fits.open(os.path.join(data, g["i"]), memmap=False) as h:
            p = h[0].header
            fw = [x.header.get("FWHM") for x in h[1:] if x.header.get("FWHM")]
            mjd0, t = float(p["MJD-OBS"]), float(p["EXPTIME"])
            mid = mjd0 + t / 2 / 86400
            o = observer(mid)
            ra = (p.get("RA") or "0:0:0").split(":"); de = (p.get("DEC") or "0:0:0").split(":")
            rad = 15 * (float(ra[0]) + float(ra[1]) / 60 + float(ra[2]) / 3600)
            sg = -1 if de[0].strip().startswith("-") else 1
            ded = sg * (abs(float(de[0])) + float(de[1]) / 60 + float(de[2]) / 3600)
            rows.append(dict(exp_id=int(p.get("EXPNUM", 0)), image=g["i"], weight=g["w"], mask=g["d"],
                             mjd_start=round(mjd0, 8), exptime=t, mjd_mid=round(mid, 8),
                             band=str(p.get("FILTER", "")).split(" ")[0], airmass=p.get("AIRMASS"),
                             magzero=p.get("MAGZERO"), seeing_arcsec=round(float(np.median(fw)) * PIX, 3) if fw else None,
                             ra_center=round(rad, 6), dec_center=round(ded, 6),
                             obs_x=round(float(o[0]), 9), obs_y=round(float(o[1]), 9), obs_z=round(float(o[2]), 9)))
    rows.sort(key=lambda r: r["mjd_mid"])
    os.makedirs(os.path.join(scen, "exposures"), exist_ok=True)
    for r in rows:
        for k in ("image", "weight", "mask"):
            dst = os.path.join(scen, "exposures", r[k])
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(os.path.join(data, r[k])), dst)
    with open(os.path.join(scen, "exposures.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, COLS); wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"exposures/{r[k]}" if k in ("image", "weight", "mask") else r[k]) for k in COLS})
    nights = sorted({int(np.floor(r["mjd_mid"] - 0.5)) for r in rows})
    json.dump(dict(sid=w["sid"], ra=w["ra"], dec=w["dec"], first_mjd=min(r["mjd_mid"] for r in rows),
                   last_mjd=max(r["mjd_mid"] for r in rows), n_exposures=len(rows), n_nights=len(nights),
                   bands="".join(sorted({r["band"] for r in rows}))), open(os.path.join(scen, "meta.json"), "w"), indent=1)
    return rows, len(nights)


def netns_ok():
    try:
        return subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def run_finder(program, scen, out, cap, gpu, netns, threads):
    tmp = tempfile.mkdtemp(prefix=f"slice_{os.path.basename(scen)}_", dir=os.environ.get("SLICE_TMP", tempfile.gettempdir()))
    subc, home = os.path.join(tmp, "sub"), os.path.join(tmp, "home")
    shutil.copytree(program, subc, ignore=shutil.ignore_patterns("__pycache__", "work")); os.makedirs(home)
    py = sys.executable
    env = dict(PATH=os.path.dirname(py) + ":/usr/bin:/bin", HOME=home, TMPDIR=home, OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), MPLCONFIGDIR=home,
               PYTHONDONTWRITEBYTECODE="1", PYTORCH_ENABLE_MPS_FALLBACK="1",
               LD_LIBRARY_PATH=os.path.join(os.path.dirname(os.path.dirname(py)), "lib"))
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    res = os.path.join(tmp, "candidates.json")
    cmd = (["unshare", "-rn"] if netns else []) + [py, "-I", os.path.join(subc, "finder.py"), "--scenario", scen,
                                                  "--out", res, "--budget-s", str(cap)]
    t = time.time()
    with open(os.path.join(tmp, "finder.log"), "w") as lg:
        p = subprocess.Popen(cmd, cwd=subc, env=env, stdout=lg, stderr=subprocess.STDOUT, start_new_session=True)
        status = "ok"
        try:
            rc = p.wait(timeout=cap + 30)
            if rc != 0:
                status = f"exit {rc}"
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); status = "timeout"
    dt = time.time() - t
    if os.path.exists(res):
        shutil.copy(res, os.path.join(out, "candidates.json"))
    elif status == "ok":
        status = "no output"
    shutil.copy(os.path.join(tmp, "finder.log"), os.path.join(out, "finder.log"))
    tail = open(os.path.join(tmp, "finder.log"), errors="replace").read()[-3000:]
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(status=status, seconds=round(dt, 1), budget_s=cap, log_tail=tail, host=socket.gethostname(), gpu=gpu, netns=netns)


def measure(bodies, rows, scen, out):
    """Forced photometry while the window's images are on disk. Writes known_phot.json (the known bodies, for their
    measured tiers) and cand_phot.json (the best candidates, for the lead screen). Returns counts for info.json."""
    import warnings
    import numpy as np
    from astropy.wcs import FITSFixedWarning
    import common as C
    warnings.simplefilter("ignore", FITSFixedWarning); warnings.simplefilter("ignore", RuntimeWarning)
    stamps = [r["image"][:17] for r in rows]
    idx = {s: k for k, s in enumerate(stamps)}
    mjds = [float(r["mjd_mid"]) for r in rows]
    obs = [np.array([r["obs_x"], r["obs_y"], r["obs_z"]], float) for r in rows]
    night = [C.night_of(m) for m in mjds]
    n = len(rows)
    exps = [C.Exposure(scen, r) for r in rows]
    try:
        nk = 0
        if bodies:
            pts, tags = [], []
            for b in bodies:
                for x in b["recs"]:
                    if x["stamp"] in idx and x.get("ra") is not None:
                        pts.append((idx[x["stamp"]], x["ra"], x["dec"])); tags.append((b["pdes"], x["stamp"], x["night"]))
            res = {b["pdes"]: [] for b in bodies}
            for (pd, st, ng), m in zip(tags, C.measure_many(exps, pts)):
                res[pd].append(dict(stamp=st, night=ng, **m))
            json.dump(res, open(os.path.join(out, "known_phot.json"), "w"))
            nk = len(pts)
        # the window's best candidates: flux along the track in every exposure ...
        cands = [c for c in C.load_cands(os.path.join(out, "candidates.json")) if c["r"] >= C.RMIN_AU]
        todo = []
        for c in cands:
            pos = C.positions(c, mjds, obs)
            if pos is not None:
                todo.append((c, pos))
            if len(todo) >= CAND_PHOT_TOP:
                break
        pts = [(k, float(ra[k]), float(de[k])) for c, (ra, de) in todo for k in range(n)]
        M = C.measure_many(exps, pts)
        recs, spts, stag = [], [], []
        for j, (c, (ra, de)) in enumerate(todo):
            tr = M[j * n:(j + 1) * n]
            byn = {}
            for k, m in enumerate(tr):
                if C.good(m):
                    byn.setdefault(night[k], []).append(k)
            # ... and the same sky spot on up to 10 exposures of other nights (a static source shows there too)
            for ng, ks in byn.items():
                k0 = ks[0]
                ot = [q for q in range(n) if night[q] != ng]
                for q in ot[::max(1, len(ot) // 10)]:
                    spts.append((q, float(ra[k0]), float(de[k0]))); stag.append((j, ng))
            gm = [(tr[k]["flux"], tr[k]["err"]) for ks in byn.values() for k in ks]
            s1 = sorted(gm, key=lambda m: m[0] / m[1])
            recs.append(dict(rank=c["rank"], p=c["p"], r=round(c["r"], 2),
                             snr=[round(m["snr"], 2) if C.good(m) else None for m in tr],
                             flag=[m.get("flag") if m else None for m in tr],
                             nights={str(ng): round(C.stack(*zip(*[(tr[k]["flux"], tr[k]["err"]) for k in ks])), 2) for ng, ks in byn.items()},
                             static={}, tot=round(C.stack(*zip(*gm)), 2) if gm else 0.0,
                             drop=round(C.stack(*zip(*s1[:-1])), 2) if len(s1) > 1 else 0.0))
        S = C.measure_many(exps, spts)
        acc = {}
        for (j, ng), m in zip(stag, S):
            acc.setdefault((j, ng), [])
            if C.good(m):
                acc[(j, ng)].append((m["flux"], m["err"]))
        for (j, ng), ms in acc.items():
            recs[j]["static"][str(ng)] = round(C.stack(*zip(*ms)), 2) if ms else None
        json.dump(dict(stamps=stamps, nights=night, candidates=recs), open(os.path.join(out, "cand_phot.json"), "w"))
        return dict(known_points=nk, candidates_measured=len(recs), sections=all(e.section for e in exps))
    finally:
        for e in exps:
            e.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slice"); ap.add_argument("work")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--local-index", help="text file, one path per line, of InstCal files already on disk: linked, not downloaded")
    ap.add_argument("--gpus", type=int, help="GPUs to use (default: all that nvidia-smi lists; 0 = one run at a time on the CPU)")
    ap.add_argument("--keep-images", action="store_true"); ap.add_argument("--redo", action="store_true")
    ap.add_argument("--windows", help="only these window keys, comma separated")
    ap.add_argument("--program", help="program directory (default SLICE_DIR/program)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    sl = os.path.abspath(a.slice)
    card = json.load(open(os.path.join(sl, "slice.json")))
    W = json.load(open(os.path.join(sl, "windows.json")))["windows"]
    S = json.load(open(os.path.join(sl, "settings.json")))
    F = json.load(open(os.path.join(sl, "files.json")))
    program = os.path.abspath(a.program or os.path.join(sl, "program"))
    kp = os.path.join(sl, "private", "known.json")
    known = json.load(open(kp))["windows"] if os.path.exists(kp) else None
    local = {}
    if a.local_index:
        local = {os.path.basename(l.strip()): l.strip() for l in open(a.local_index) if l.strip()}
    sel = W
    if a.windows:
        keys = {int(k) for k in a.windows.split(",")}
        sel = [w for w in W if w["key"] in keys]
    out_root, data_root, scen_root = (os.path.join(a.work, x) for x in ("out", "data", "scen"))
    todo = [w for w in sel if a.redo or not os.path.exists(os.path.join(out_root, w["sid"], "done.json"))]
    if a.gpus is None:
        try:
            a.gpus = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
                          if l.startswith("GPU")])
        except Exception:
            a.gpus = 0
    nwork = max(1, a.gpus)

    def files_of(w):
        return {e["stamp"]: F[e["stamp"]] for e in w["exposures"] if e["stamp"] in F}

    gb = sum(f[p][2] for w in todo for f in files_of(w).values() for p in "iwd") / 1e9
    nexp = sum(len(files_of(w)) for w in todo)
    sec = sum(budget_for(S, len(files_of(w))) for w in todo)
    wmax = max([sum(f[p][2] for f in files_of(w).values() for p in "iwd") for w in todo] or [0]) / 1e9
    nlocal = sum(f[p][0] in local for w in todo for f in files_of(w).values() for p in "iwd")
    print(f"slice {card['id']} ({card['group_name']}): {len(todo)} windows to run of {len(sel)}, {nexp} exposures, "
          f"{gb:.1f} GB ({nlocal} of {3 * nexp} files are in the local index)")
    print(f"program {program}; finder budget {sec / 3600:.2f} GPU-hours at most, about {sec / 3600 / nwork:.2f} h on {nwork} "
          f"{'GPU' if a.gpus else 'CPU run'}{'s' if nwork > 1 else ''}; at most about "
          f"{wmax * (S.get('fetch_ahead_windows', 1) + 1) * nwork:.0f} GB of images on disk at once")
    print("known bodies: " + ("none listed: no reward for this slice" if known is None else
                              f"{sum(len(known.get(str(w['key']), [])) for w in sel)} in these windows (private/known.json)"))
    problems = []
    fp = os.path.join(program, "finder.py")
    if not os.path.exists(fp):
        problems.append(f"no finder.py in {program}")
    else:
        md5 = hashlib.md5(open(fp, "rb").read()).hexdigest()
        print(f"finder.py md5 {md5}" + ("" if a.program or md5 == card.get("program", {}).get("finder_md5") else "  PROBLEM: differs from slice.json"))
    miss = [e["stamp"] for w in todo for e in w["exposures"] if e["stamp"] not in F]
    if miss:
        problems.append(f"{len(miss)} exposures have no entry in files.json: {miss[:4]}")
    if a.dry_run or problems:
        for p in problems:
            print("PROBLEM:", p)
        try:
            os.makedirs(a.work, exist_ok=True)
            print(f"free disk at {a.work}: {shutil.disk_usage(a.work).free / 1e9:.0f} GB")
        except Exception as e:
            print("work dir:", e)
        print("dry run: nothing fetched, nothing run" if a.dry_run else "stopped")
        sys.exit(1 if problems else 0)

    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard if hard != resource.RLIM_INFINITY else 65536, hard))
    except Exception:
        pass
    for d in (out_root, data_root, scen_root):
        os.makedirs(d, exist_ok=True)
    netns = netns_ok()
    log(f"node {socket.gethostname()}: {nwork} at a time, network namespace {netns}, {len(local)} files in the local index")
    ready = queue.Queue()
    slots = threading.Semaphore((max(0, S.get("fetch_ahead_windows", 1)) + 1) * nwork)
    tot = dict(bytes=0, fetch_s=0.0, ok=0, failed=0)

    def finish(w, info):
        d = os.path.join(out_root, w["sid"])
        os.makedirs(d, exist_ok=True)
        json.dump(dict(info, key=w["key"], sid=w["sid"], slice=card["id"]), open(os.path.join(d, "info.json"), "w"), indent=1)
        json.dump(dict(status=info["status"], when=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(d, "done.json"), "w"))
        if not a.keep_images:
            shutil.rmtree(os.path.join(data_root, str(w["key"])), ignore_errors=True)
            shutil.rmtree(os.path.join(scen_root, w["sid"]), ignore_errors=True)
        with LOCK:
            tot["ok" if info["status"] == "ok" else "failed"] += 1

    def fetcher():
        for w in todo:
            slots.acquire()
            try:
                fl = files_of(w)
                d = os.path.join(data_root, str(w["key"]))
                os.makedirs(d, exist_ok=True)
                t = time.time()
                with ThreadPoolExecutor(S.get("fetch_streams", 16)) as ex:
                    res = list(ex.map(lambda x: fetch_file(x[0], x[1], x[2], d, local), [f[p] for f in fl.values() for p in "iwd"]))
                dt, nb = time.time() - t, sum(r[1] for r in res)
                with LOCK:
                    tot["bytes"] += nb; tot["fetch_s"] += dt
                if any(r[0] == "FAILED" for r in res):
                    finish(w, dict(status="fetch failed", seconds=0, fetch_s=round(dt, 1))); slots.release()
                    continue
                scen = os.path.join(scen_root, w["sid"])
                rows, nights = build_scenario(w, {st: {p: f[p][0] for p in "iwd"} for st, f in fl.items()}, d, scen)
                log(f"fetched {w['sid']} ({w['key']}): {len(rows)} exposures on {nights} nights, {nb / 1e9:.1f} GB in {dt:.0f} s"
                    + (f" ({nb / 1e6 / dt:.0f} MB/s)" if nb and dt > 0 else ""))
                ready.put((w, scen, rows, dict(fetch_s=round(dt, 1), bytes=nb, n_exposures=len(rows), n_nights=nights)))
            except Exception as e:
                finish(w, dict(status=f"setup failed: {type(e).__name__}: {e}"[:300], seconds=0)); slots.release()
        for _ in range(nwork):
            ready.put(None)

    def worker(gpu):
        while True:
            it = ready.get()
            if it is None:
                return
            w, scen, rows, st = it
            try:
                d = os.path.join(out_root, w["sid"])
                os.makedirs(d, exist_ok=True)
                for f in ("exposures.csv", "meta.json"):
                    shutil.copy(os.path.join(scen, f), os.path.join(d, f))
                info = run_finder(program, scen, d, budget_for(S, len(rows)), gpu if a.gpus else None, netns, a.threads)
                info.update(st)
                t = time.time()
                try:
                    info["measured"] = measure((known or {}).get(str(w["key"])), rows, scen, d)
                except Exception as e:
                    info["measure_error"] = f"{type(e).__name__}: {e}"[:300]
                info["measure_s"] = round(time.time() - t, 1)
                log(f"ran {w['sid']}: {info['status']} {info['seconds']:.0f} s of {info['budget_s']:.0f}, measured in {info['measure_s']:.0f} s"
                    + (f" ({info['measure_error']})" if "measure_error" in info else ""))
                finish(w, info)
            except Exception as e:
                finish(w, dict(status=f"run failed: {type(e).__name__}: {e}"[:300], seconds=0))
            finally:
                slots.release()

    t0 = time.time()
    th = [threading.Thread(target=fetcher)] + [threading.Thread(target=worker, args=(k,)) for k in range(nwork)]
    for t in th: t.start()
    for t in th: t.join()
    summ = dict(slice=card["id"], host=socket.gethostname(), gpus=a.gpus, program=program, windows=len(todo), **tot,
                wall_s=round(time.time() - t0, 1), when=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(summ, open(os.path.join(a.work, "run.json"), "w"), indent=1)
    log("RUN DONE", json.dumps(summ))


if __name__ == "__main__":
    main()
