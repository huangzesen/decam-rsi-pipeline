"""Grade one finder submission on a fixed set of sealed scenarios.

For each scenario the submission's finder.py runs in a macOS sandbox (no network; it may read only its own copy of the
submission, that one scenario and the Python install; it may write only its output dir and a scratch HOME), under a
wall-clock cap (the finder is told the cap and should write OUT.json before it):
    python finder.py --scenario SCEN_DIR --out OUT.json --budget-s SECONDS
Then:
  1. each scenario keeps at most --max-per-scen candidates (highest p); over all scenarios the --k-per-scen x
     n_scenarios candidates with the highest p are kept (the global slot budget);
  2. each kept candidate's state is propagated (two-body about the Sun, light time) to every exposure; it matches a
     truth body when it lies within max(--tol arcsec, the body's 3-sigma) on >= 3 distinct visible nights; bodies
     match once, in order of p (a second candidate on a body counts as nothing); a candidate that matches a scored
     body (tier 1-3) goes to it before any neutral one (2026-11-11: a tier-None body with a 1.66e9" sigma, first in
     its truth file, had taken the top candidate of its field although that candidate sat 0.05" from a tier-3 body);
  3. hit weights by truth tier: 1 -> 1, 2 -> 2, 3 -> 4; a body too faint for our own photometry in these data
     (tier 0) or with an imprecise orbit (tier None) is neutral, and the count of tier-0 matches is kept as a sign of
     a script that carries known orbits;
  4. unmatched candidates at >= 25 au are tested on the held-back night: forced photometry along the predicted path,
     inverse-variance stack S/N >= 5, and it must be a mover: the same sky spots faint on the other nights (S/N < 3),
     the orbit itself visible in the non-held exposures (S/N >= 5), no single held-back exposure carrying it
     (S/N >= 3 without the brightest) = confirmed new (weight --w-new); rejected ones are listed in grade.json;
  5. reward = (sum of hit weights + confirmed weights) / (sum of weights of all tier 1-3 bodies in the set).
Writes GRADE_DIR/grade.json (full, private) and GRADE_DIR/public.json (reward and run status only).
With --runs-dir the finder runs come from elsewhere (Lonestar6, see ls6/): RUNS_DIR/<sid>/{info.json,candidates.json};
truth and scoring stay here.

Usage: ~/anaconda3/bin/python -I grade.py ROOT SUBMISSION_DIR GRADE_DIR --scenarios s1,s2,... [--cap-s 600]
"""
import argparse, json, math, os, shutil, signal, subprocess, sys, tempfile, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "linker"))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
from orbits import rv_to_elements, elements_to_rv, GM, C_AU
from make_truth import Exposure, phot, stack

PY = os.path.expanduser("~/anaconda3/bin/python")
W_TIER = {1: 1.0, 2: 2.0, 3: 4.0, 0: 0.0}


def profile(root, sub, scen, out, home):
    pyroot = os.path.realpath(os.path.expanduser("~/anaconda3"))
    return f"""(version 1)
(allow default)
(deny network*)
(deny signal)
(allow signal (target same-sandbox))
(deny file-read* (subpath "{os.path.realpath(root)}"))
(deny file-read* (subpath "{os.path.realpath(os.path.expanduser('~'))}"))
(allow file-read* (subpath "{pyroot}"))
(allow file-read* (subpath "{os.path.realpath(sub)}"))
(allow file-read* (subpath "{os.path.realpath(scen)}"))
(allow file-read* (subpath "{os.path.realpath(out)}"))
(allow file-read* (subpath "{os.path.realpath(home)}"))
(allow file-read* (literal "{os.path.realpath(os.path.expanduser('~'))}"))
(deny file-write* (subpath "{os.path.realpath(root)}"))
(deny file-write* (subpath "{os.path.realpath(os.path.expanduser('~'))}"))
(allow file-write* (subpath "{os.path.realpath(out)}"))
(allow file-write* (subpath "{os.path.realpath(home)}"))
"""


def run_finder(root, sub, scen, cap):
    tmp = tempfile.mkdtemp(prefix="rsi_run_")
    subc, out, home = os.path.join(tmp, "sub"), os.path.join(tmp, "out"), os.path.join(tmp, "home")
    shutil.copytree(sub, subc); os.makedirs(out); os.makedirs(home)
    sb = os.path.join(tmp, "p.sb")
    open(sb, "w").write(profile(root, subc, scen, out, home))
    env = dict(PATH="/usr/bin:/bin", HOME=home, TMPDIR=home, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTORCH_ENABLE_MPS_FALLBACK="1", MPLCONFIGDIR=home, PYTHONDONTWRITEBYTECODE="1")
    cmd = ["sandbox-exec", "-f", sb, PY, "-I", os.path.join(subc, "finder.py"), "--scenario", scen,
           "--out", os.path.join(out, "candidates.json"), "--budget-s", str(cap)]
    t = time.time()
    log = open(os.path.join(tmp, "finder.log"), "w")
    p = subprocess.Popen(cmd, cwd=subc, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    status = "ok"
    try:
        rc = p.wait(timeout=cap + 30)
        if rc != 0:
            status = f"exit {rc}"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait(); status = "timeout"
    dt = time.time() - t
    log.close()
    cands, err = load_cands(os.path.join(out, "candidates.json"), status)
    tail = open(os.path.join(tmp, "finder.log")).read()[-3000:]
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(status=status, seconds=round(dt, 1), error=err, n_out=len(cands), log_tail=tail), cands


def load_cands(f, status):
    cands, err = [], None
    if os.path.exists(f):
        try:
            d = json.load(open(f))
            for c in (d.get("candidates", []) if isinstance(d, dict) else d)[:100000]:
                s = [float(x) for x in c["state"]]
                if len(s) == 6 and all(map(math.isfinite, s)) and math.isfinite(float(c["epoch_mjd"])):
                    cands.append(dict(epoch=float(c["epoch_mjd"]), state=s, p=float(c.get("p", 0.0)),
                                      mag=c.get("mag")))
        except Exception as e:
            err = f"bad output: {type(e).__name__}"
    elif status == "ok":
        err = "no output"
    return cands, err


def finder_from_runs(runs_dir, sid):
    """Use a finder run made elsewhere (ls6/finder_node.py): RUNS_DIR/SID/{info.json, candidates.json}."""
    d = os.path.join(runs_dir, sid)
    info = json.load(open(os.path.join(d, "info.json"))) if os.path.exists(os.path.join(d, "info.json")) \
        else dict(status="not run", seconds=0.0, log_tail="")
    cands, err = load_cands(os.path.join(d, "candidates.json"), info["status"])
    return dict(status=info["status"], seconds=info["seconds"], error=err, n_out=len(cands),
                log_tail=info.get("log_tail", ""), host=info.get("host"), netns=info.get("netns")), cands


def positions(c, mjds, obs):
    """Astrometric RA/Dec (deg) of a two-body orbit (equatorial heliocentric state) at each time, light time included."""
    r0, v0 = np.array(c["state"][:3]), np.array(c["state"][3:])
    a, e, i, Om, om, M0 = rv_to_elements(r0[None], v0[None])
    if not (e[0] < 1 and a[0] > 0):
        return None
    n = math.sqrt(GM / a[0] ** 3)
    ra, de = [], []
    for t, o in zip(mjds, obs):
        tau = 0.0
        for _ in range(3):
            r, _ = elements_to_rv(a, e, i, Om, om, M0 + n * (t - tau - c["epoch"]))
            g = r[0] - o
            tau = np.linalg.norm(g) / C_AU
        ra.append(math.degrees(math.atan2(g[1], g[0])) % 360); de.append(math.degrees(math.asin(g[2] / np.linalg.norm(g))))
    return np.array(ra), np.array(de)


def sep(ra1, de1, ra2, de2):
    d1, d2 = np.radians(de1), np.radians(de2)
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(np.radians(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("sub"); ap.add_argument("out")
    ap.add_argument("--scenarios", required=True); ap.add_argument("--cap-s", type=float, default=600)
    ap.add_argument("--k-per-scen", type=int, default=25); ap.add_argument("--max-per-scen", type=int, default=200)
    ap.add_argument("--tol", type=float, default=2.0); ap.add_argument("--w-new", type=float, default=2.0)
    ap.add_argument("--runs-dir", help="score finder runs made elsewhere instead of running the finder here")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sids = a.scenarios.split(",")
    per, allc = {}, []
    for sid in sids:
        P = json.load(open(os.path.join(a.root, "private", "scenarios", f"{sid}.json")))
        scen = os.path.join(a.root, f"scen_{P['split']}", sid)
        info, cands = finder_from_runs(a.runs_dir, sid) if a.runs_dir else run_finder(a.root, a.sub, scen, a.cap_s)
        cands = sorted(cands, key=lambda c: -c["p"])[:a.max_per_scen]
        for c in cands:
            c["sid"] = sid
        per[sid] = info
        allc += cands
        print(f"{sid}: {info['status']} {info['seconds']} s, {info['n_out']} candidates {info['error'] or ''}", flush=True)
    K = a.k_per_scen * len(sids)
    kept = sorted(allc, key=lambda c: -c["p"])[:K]

    total_w, got_w, conf_w, hits, confirmed, rejected, n_tier0 = 0.0, 0.0, 0.0, [], [], [], 0
    for sid in sids:
        P = json.load(open(os.path.join(a.root, "private", "scenarios", f"{sid}.json")))
        T = json.load(open(os.path.join(a.root, "private", "truth", f"{sid}.json")))
        E = P["all_exposures"]
        held = set(P["held_back"])
        mjds = [e["mjd_mid"] for e in E]
        obs = [np.array([e["obs_x"], e["obs_y"], e["obs_z"]]) for e in E]
        night = [int(np.floor(m - 0.5)) for m in mjds]
        bodies = [b for b in T["bodies"] if b["tier"] is not None]
        total_w += sum(W_TIER[b["tier"]] for b in bodies if b["tier"] in (1, 2, 3))
        taken = set()
        exps = {}
        for c in [c for c in kept if c["sid"] == sid]:
            pos = positions(c, mjds, obs)
            if pos is None or np.linalg.norm(c["state"][:3]) < 25:
                continue
            ra, de = pos
            match = None
            for b in sorted(T["bodies"], key=lambda b: b["tier"] not in (1, 2, 3)):   # scored bodies first, else file order
                if b["pdes"] in taken:
                    continue
                ok = set()
                for k, r in enumerate(b["recs"]):
                    if r["held"] or r.get("ra") is None:
                        continue
                    tol = max(a.tol, r.get("s_ra") or 0, r.get("s_dec") or 0)
                    if sep(ra[k], de[k], r["ra"], r["dec"]) <= tol:
                        ok.add(night[k])
                if len(ok) >= 3:
                    match = b
                    break
            if match is not None:
                taken.add(match["pdes"])
                if match["tier"] is not None:
                    w = W_TIER[match["tier"]]
                    n_tier0 += match["tier"] == 0
                    got_w += w
                    hits.append(dict(sid=sid, pdes=match["pdes"], tier=match["tier"], w=w, p=c["p"]))
                continue
            def meas(k, r_, d_):
                if k not in exps:
                    ex = E[k]; d = os.path.join(a.root, "data", str(P["window"]))
                    exps[k] = Exposure(os.path.join(d, ex["image"]), os.path.join(d, ex["weight"]), os.path.join(d, ex["mask"]))
                loc = exps[k].locate(r_, d_)
                if loc:
                    fl, er, flag = phot(*exps[k].data(loc[0]), loc[1], loc[2], loc[3])
                    if fl is not None and er and flag == "":
                        return fl, er
                return None
            hk = [k for k, ex in enumerate(E) if ex["image"] in held]
            vk = [k for k, ex in enumerate(E) if ex["image"] not in held]
            m_held = [m for m in (meas(k, ra[k], de[k]) for k in hk) if m]
            if not m_held or stack(*zip(*m_held)) < 5:
                continue
            # a confirmation must be a moving source: (1) the held-back positions are faint on the other nights (no
            # star or galaxy there), (2) the orbit itself shows on the visible exposures, (3) no single held-back
            # exposure carries it (cosmic ray, artefact)
            m_static = [m for m in (meas(j, ra[k], de[k]) for k in hk for j in vk[:: max(1, len(vk) // 12)]) if m]
            m_vis = [m for m in (meas(j, ra[j], de[j]) for j in vk) if m]
            snr_h = stack(*zip(*m_held))
            snr_s = stack(*zip(*m_static)) / max(1, len(hk)) ** 0.5 if m_static else 0.0
            snr_v = stack(*zip(*m_vis)) if m_vis else 0.0
            drop1 = sorted(m_held, key=lambda m: m[0] / m[1])[:-1]
            snr_h1 = stack(*zip(*drop1)) if drop1 else snr_h
            ok = snr_s < 3 and snr_v >= 5 and (len(m_held) < 2 or snr_h1 >= 3)
            (confirmed if ok else rejected).append(dict(sid=sid, p=c["p"], snr=round(snr_h, 1), snr_static=round(snr_s, 1),
                                                      snr_visible=round(snr_v, 1), snr_held_wo_max=round(snr_h1, 1),
                                                      state=c["state"], epoch=c["epoch"]))
            if ok:
                conf_w += a.w_new
        for x in exps.values():
            x.close()
    reward = (got_w + conf_w) / total_w if total_w else 0.0
    nbad = sum(1 for v in per.values() if v["status"] != "ok" or v["error"])
    status = "ok" if nbad == 0 else f"{nbad} of {len(sids)} scenarios failed or timed out"
    G = dict(reward=round(reward, 4), status=status, truth_weight=total_w, hit_weight=got_w, confirmed_weight=conf_w,
             n_kept=len(kept), K=K, tier0_matches=n_tier0, hits=hits, confirmed=confirmed, held_rejected=rejected, per_scenario=per,
             settings=dict(cap_s=a.cap_s, k_per_scen=a.k_per_scen, tol=a.tol, w_new=a.w_new, scenarios=sids,
                      runs_dir=a.runs_dir))
    json.dump(G, open(os.path.join(a.out, "grade.json"), "w"), indent=1)
    json.dump(dict(reward=G["reward"], status=status,
                   finder_seconds_total=round(sum(v["seconds"] for v in per.values()))),
              open(os.path.join(a.out, "public.json"), "w"), indent=1)
    print(f"reward {reward:.4f} (hits {got_w} + confirmed {conf_w} of {total_w}); {status}")


if __name__ == "__main__":
    main()
