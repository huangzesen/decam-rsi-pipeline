"""The group job: everything one group needs, from the archive to its candidates.

For every window of the group (or of one role), on this node:
  1. fetch   the window's InstCal files from the NOIRLab archive (md5 checked while streaming, 4 tries; a file found
             in --local-index is linked instead), a few windows ahead of the GPUs;
  2. build   the scenario the finder reads (exposures.csv, meta.json, exposures/), as rsi/bin/make_scenarios.py does,
             with every exposure visible (nothing is held back in production);
  3. run     the group's program, one window per GPU:  finder.py --scenario S --out OUT.json --budget-s SECONDS
             in a fresh copy, without network when unprivileged namespaces work, killed 30 s after the budget;
  4. measure the known bodies of the window (--known, from known_bodies.py): forced photometry at their predicted
             positions, so that the score uses what the images really show;
  5. keep    WORKDIR/out/<sid>/{candidates.json, info.json, exposures.csv, meta.json, known_phot.json, done.json};
  6. delete  the window's images (unless --keep-images).
Windows with done.json are skipped, so a rerun resumes. With several Slurm tasks (one per node) each takes its share.

Usage: python -I run_group.py GROUP_DIR WORKDIR [practice|validation|production|all] [--dry-run] [--known KNOWN.json]
       [--slices ID,ID] [--windows KEY,KEY] [--limit N] [--gpus N] [--local-index FILE] [--keep-images] [--redo]
--slices runs the windows of the named slices of the group (slices.json): the unit a job is given.
--dry-run prints what would be fetched and run, checks the program, the file list and the disk, and touches nothing.
"""
import argparse, csv, hashlib, json, os, queue, shutil, signal, socket, subprocess, sys, tempfile, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://astroarchive.noirlab.edu/api"
PIX = 0.263
COLS = ["exp_id", "image", "weight", "mask", "mjd_start", "exptime", "mjd_mid", "band", "airmass", "magzero",
        "seeing_arcsec", "ra_center", "dec_center", "obs_x", "obs_y", "obs_z"]
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
    tmp = tempfile.mkdtemp(prefix=f"grp_{os.path.basename(scen)}_", dir=os.environ.get("GROUP_TMP", tempfile.gettempdir()))
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


def known_phot(bodies, rows, scen, out):
    """Forced photometry of the known bodies (rsi/bin/make_truth.py): aperture 0.7 FWHM, sky from a 3-5 FWHM annulus."""
    import warnings
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS, FITSFixedWarning
    warnings.simplefilter("ignore", FITSFixedWarning); warnings.simplefilter("ignore", RuntimeWarning)
    by = {r["image"][:17]: r for r in rows}
    need = sorted({x["stamp"] for b in bodies for x in b["recs"] if x["stamp"] in by})
    res = {b["pdes"]: [] for b in bodies}
    for st in need:
        r = by[st]
        with fits.open(os.path.join(scen, "exposures", r["image"]), memmap=False) as hi, \
                fits.open(os.path.join(scen, "exposures", r["weight"]), memmap=False) as hw, \
                fits.open(os.path.join(scen, "exposures", r["mask"]), memmap=False) as hm:
            ccds = [(k, WCS(x.header), x.header.get("NAXIS1", 2046), x.header.get("NAXIS2", 4094), x.header.get("FWHM") or 4.0)
                    for k, x in enumerate(hi) if k and "CTYPE1" in x.header]
            for b in bodies:
                for x in b["recs"]:
                    if x["stamp"] != st or x.get("ra") is None:
                        continue
                    rec = dict(stamp=st, night=x["night"])
                    for k, wcs, nx, ny, fwhm in ccds:
                        px, py = wcs.all_world2pix(x["ra"], x["dec"], 0)
                        if 0 <= px < nx and 0 <= py < ny:
                            img, wt, msk = hi[k].data.astype(np.float32), hw[k].data.astype(np.float32), hm[k].data
                            ra_, r1, r2 = max(1.5, 0.7 * fwhm), 3 * fwhm, 5 * fwhm
                            x0, x1 = int(max(0, np.floor(px - r2))), int(min(img.shape[1], np.ceil(px + r2) + 1))
                            y0, y1 = int(max(0, np.floor(py - r2))), int(min(img.shape[0], np.ceil(py + r2) + 1))
                            yy, xx = np.mgrid[y0:y1, x0:x1]
                            d = np.hypot(xx - px, yy - py)
                            I, Wt, M = img[y0:y1, x0:x1], wt[y0:y1, x0:x1], msk[y0:y1, x0:x1]
                            ap, an = d <= ra_, (d >= r1) & (d <= r2) & (M == 0) & (Wt > 0)
                            rec.update(hdu=k, x=round(float(px), 2), y=round(float(py), 2))
                            if ap.sum() == 0 or an.sum() < 20:
                                rec.update(flag="edge")
                            else:
                                sky = np.median(I[an])
                                f = float(np.sum(I[ap] - sky))
                                var = float(np.sum(1.0 / np.where(Wt[ap] > 0, Wt[ap], np.inf))) + ap.sum() ** 2 * np.pi / 2 * np.var(I[an]) / an.sum()
                                err = float(np.sqrt(var)) if var > 0 else None
                                rec.update(flux=f, err=err, flag="bad" if ((M[ap] != 0) | (Wt[ap] <= 0)).any() else "",
                                           snr=(f / err) if err else None)
                            break
                    res[b["pdes"]].append(rec)
    json.dump(res, open(os.path.join(out, "known_phot.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group"); ap.add_argument("work")
    ap.add_argument("role", nargs="?", default="all", choices=["practice", "validation", "production", "all"])
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--known")
    ap.add_argument("--windows"); ap.add_argument("--limit", type=int)
    ap.add_argument("--slices", help="slice ids of this group (slices.json), comma separated: run only their windows")
    ap.add_argument("--gpus", type=int, help="GPUs to use (default: all that nvidia-smi lists; 0 = one run at a time on the CPU)")
    ap.add_argument("--local-index", help="text file, one path per line, of InstCal files already on disk")
    ap.add_argument("--program", help="program directory (default GROUP_DIR/program)")
    ap.add_argument("--keep-images", action="store_true"); ap.add_argument("--redo", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    g = os.path.abspath(a.group)
    card = json.load(open(os.path.join(g, "group.json")))
    W = json.load(open(os.path.join(g, "windows.json")))["windows"]
    S = json.load(open(os.path.join(g, "settings.json")))
    fp = os.path.join(g, "files.json")
    F = json.load(open(fp)) if os.path.exists(fp) else None
    program = os.path.abspath(a.program or os.path.join(g, "program"))
    known = json.load(open(a.known))["windows"] if a.known else {}
    local = {}
    if a.local_index:
        local = {os.path.basename(l.strip()): l.strip() for l in open(a.local_index) if l.strip()}
    sel = [w for w in W if w["role"] != "no data" and (a.role == "all" or w["role"] == a.role)]
    if a.windows:
        keys = {int(k) for k in a.windows.split(",")}
        sel = [w for w in sel if w["key"] in keys]
    if a.slices:
        want = set(a.slices.split(","))
        sel = [w for w in sel if w.get("slice") in want]
    out_root, data_root, scen_root = (os.path.join(a.work, x) for x in ("out", "data", "scen"))
    shard, nshards = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    sel = [w for k, w in enumerate(sel) if k % nshards == shard]          # a node's share never depends on what is done
    todo = [w for w in sel if a.redo or not os.path.exists(os.path.join(out_root, w["sid"], "done.json"))]
    if a.limit:
        todo = todo[:a.limit]
    if a.gpus is None:
        try:
            a.gpus = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
                          if l.startswith("GPU")])
        except Exception:
            a.gpus = 0
    nwork = max(1, a.gpus)

    def files_of(w):
        return {e["stamp"]: F[e["stamp"]] for e in w["exposures"] if F and e["stamp"] in F}

    gb = sum(f[p][2] for w in todo for f in files_of(w).values() for p in "iwd") / 1e9
    nexp, nfile = sum(len(w["exposures"]) for w in todo), sum(len(files_of(w)) for w in todo)
    sec = sum(budget_for(S, len(files_of(w))) for w in todo)
    wmax = max([sum(f[p][2] for f in files_of(w).values() for p in "iwd") for w in todo] or [0]) / 1e9
    print(f"group {card['id']} ({card['group']}), role {a.role}: {len(todo)} windows to run of {len(sel)} selected "
          f"(shard {shard} of {nshards}), {nexp} exposures, {nfile} with files, {gb:.1f} GB to fetch")
    print(f"program {program}; budget {sec / 3600:.1f} GPU-hours at most, about {sec / 3600 / nwork:.1f} h on {nwork} "
          f"{'GPU' if a.gpus else 'CPU run'}{'s' if nwork > 1 else ''}; at most about "
          f"{wmax * (S.get('fetch_ahead_windows', 1) + 1) * nwork:.0f} GB of images on disk at once")
    problems = []
    if F is None:
        problems.append("no files.json: run make_groups.py after the archive file list is fetched")
    if not os.path.exists(os.path.join(program, "finder.py")):
        problems.append(f"no finder.py in {program}")
    if a.known and not any(str(w["key"]) in known for w in todo):
        problems.append("--known holds none of these windows")
    few = [w["key"] for w in todo if len({int((e["mjd_start"] - 0.5) // 1) for e in w["exposures"] if F and e["stamp"] in F}) < 3]
    if few:
        print(f"{len(few)} windows have fewer than 3 nights with files and will be skipped: {few[:8]}")
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

    for d in (out_root, data_root, scen_root):
        os.makedirs(d, exist_ok=True)
    netns = netns_ok()
    log(f"node {socket.gethostname()}: {nwork} workers, network namespace {netns}, {len(local)} files in the local index")
    ready = queue.Queue()
    slots = threading.Semaphore((max(0, S.get("fetch_ahead_windows", 1)) + 1) * nwork)
    tot = dict(bytes=0, fetch_s=0.0, ok=0, failed=0, skipped=0)

    def finish(w, info):
        d = os.path.join(out_root, w["sid"])
        os.makedirs(d, exist_ok=True)
        json.dump(dict(info, key=w["key"], sid=w["sid"], role=w["role"], group=card["id"]), open(os.path.join(d, "info.json"), "w"), indent=1)
        json.dump(dict(status=info["status"], when=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(d, "done.json"), "w"))
        if not a.keep_images:
            shutil.rmtree(os.path.join(data_root, str(w["key"])), ignore_errors=True)
            shutil.rmtree(os.path.join(scen_root, w["sid"]), ignore_errors=True)
        with LOCK:
            tot["ok" if info["status"] == "ok" else "skipped" if info["status"].startswith("skipped") else "failed"] += 1

    def fetcher():
        for w in todo:
            slots.acquire()
            try:
                fl = files_of(w)
                nn = len({int((e["mjd_start"] - 0.5) // 1) for e in w["exposures"] if e["stamp"] in fl})
                if nn < 3:
                    finish(w, dict(status="skipped: fewer than 3 nights with files", seconds=0)); slots.release()
                    continue
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
                kb = known.get(str(w["key"]))
                if kb:
                    try:
                        known_phot(kb, rows, scen, d)
                    except Exception as e:
                        info["known_phot_error"] = f"{type(e).__name__}: {e}"[:300]
                log(f"ran {w['sid']}: {info['status']} {info['seconds']:.0f} s of {info['budget_s']:.0f}")
                finish(w, info)
            except Exception as e:
                finish(w, dict(status=f"run failed: {type(e).__name__}: {e}"[:300], seconds=0))
            finally:
                slots.release()

    t0 = time.time()
    th = [threading.Thread(target=fetcher)] + [threading.Thread(target=worker, args=(k,)) for k in range(nwork)]
    for t in th: t.start()
    for t in th: t.join()
    summ = dict(group=card["id"], role=a.role, host=socket.gethostname(), shard=shard, windows=len(todo), **tot,
                wall_s=round(time.time() - t0, 1), when=time.strftime("%Y-%m-%d %H:%M:%S"))
    tag = a.role if not a.slices else "slices_" + hashlib.sha1(a.slices.encode()).hexdigest()[:8]
    if a.slices:
        summ["slices"] = a.slices.split(",")
    json.dump(summ, open(os.path.join(a.work, f"summary_{tag}_{shard}.json"), "w"), indent=1)
    log("DONE", json.dumps(summ))


if __name__ == "__main__":
    main()
