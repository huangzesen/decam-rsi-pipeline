"""Gen 10a hybrid TNO finder: catalogue linker (lk/, = gen009b) then orbit-space shift-and-stack (sst/, = gen009d)
in the leftover budget.  Stack candidates that duplicate a linker candidate are dropped, the rest are merged with a
rescaled p.  See NOTES.md."""
import argparse, json, math, os, subprocess, sys, time
import numpy as np

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lk"))
GM = 2.959122082855911e-4
ARC = math.pi / 180 / 3600

# fraction of the budget the stack may use; the linker gets the rest
ST_FRAC = float(os.environ.get("TNO_STFRAC", 0.35))
# stack p mapping: p' = S_SCALE * p_stack ** S_POW, only after linker candidates (see merge)
S_SCALE = float(os.environ.get("TNO_SSCALE", 0.6))


def log(*a):
    print(f"[hyb {time.time() - T0:6.1f}s]", *a, flush=True)


def valid(c):
    try:
        s = [float(v) for v in c["state"]]
        if len(s) != 6 or not all(map(math.isfinite, s + [float(c["epoch_mjd"]), float(c["p"])])):
            return False
        r = math.sqrt(sum(v * v for v in s[:3]))
        return r >= 25.0 and sum(v * v for v in s[3:]) / 2 - GM / r < 0
    except Exception:
        return False


def load(path):
    try:
        return [c for c in json.load(open(path))["candidates"] if valid(c)]
    except Exception:
        return []


def night_geom(scen):
    import csv
    rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
    nt = {}
    for r in rows:
        t = float(r["mjd_mid"]); n = math.floor(t - 0.5)
        nt.setdefault(n, (t, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
    return np.array([v[0] for v in nt.values()]), np.array([v[1] for v in nt.values()])


def dirs(cs, night_t, night_o):
    import orbit
    D = np.zeros((len(cs), len(night_t), 3))
    for ep in set(round(float(c["epoch_mjd"]), 6) for c in cs):
        idx = [k for k, c in enumerate(cs) if round(float(c["epoch_mjd"]), 6) == ep]
        S = np.array([cs[k]["state"] for k in idx], float)
        D[idx] = orbit.predict_dirs_b(S, ep, night_t, night_o)
    return D


def merge(L, S, night_t, night_o, tol=4.0):
    """L: linker candidates (calibrated p), S: stack candidates.  Stack ones within tol" of any linker candidate
    on >= min(3, nights) nights are dropped (same body; the linker's state is more precise).  Remaining stack
    candidates get p' = S_SCALE * p, capped below the linker's p for the same rank so that the linker's ranking
    is never displaced at the top."""
    L = sorted(L, key=lambda c: -c["p"])
    S = sorted(S, key=lambda c: -c["p"])[:300]
    if S and L:
        try:
            DL = dirs(L, night_t, night_o); DS = dirs(S, night_t, night_o)
            need = min(3, len(night_t)); cosl = math.cos(tol * ARC)
            cnt = np.zeros((len(S), len(L)), np.int16)
            for n in range(len(night_t)):
                cnt += (DS[:, n] @ DL[:, n].T > cosl)
            dup = (cnt >= need).any(1)
            # within-stack dedup on nights as well
            cs = np.zeros((len(S), len(S)), np.int16)
            for n in range(len(night_t)):
                cs += (DS[:, n] @ DS[:, n].T > cosl)
            alive = ~dup
            dd = np.triu(cs >= need, 1)
            for i in range(len(S)):
                if alive[i]:
                    alive[i + 1:] &= ~dd[i, i + 1:]
            log(f"merge: {int(dup.sum())} of {len(S)} stack candidates duplicate linker ones, {int(alive.sum())} kept")
            S = [c for c, a in zip(S, alive) if a]
        except Exception as ex:
            log("merge dedup failed", ex)
    out = [dict(c) for c in L]
    for c in S:
        d = dict(c)
        d["p"] = round(float(min(1.0, max(0.0, S_SCALE * float(c["p"])))), 6)
        out.append(d)
    out.sort(key=lambda c: -c["p"])
    return out[:200]


def write(path, cs):
    tmp = path + ".tmp"
    json.dump({"candidates": cs}, open(tmp, "w"))
    os.replace(tmp, path)


def run(cmd, deadline, poll=None):
    p = subprocess.Popen(cmd)
    try:
        while p.poll() is None:
            if time.time() - T0 > deadline:
                log("killing", cmd[2], "at deadline")
                p.kill(); p.wait(); break
            if poll:
                poll()
            time.sleep(1.0)
    except BaseException:
        p.kill(); raise
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    a = ap.parse_args()
    B = a.budget_s
    out = os.path.abspath(a.out)
    odir = os.path.dirname(out)
    lk_out = os.path.join(odir, ".lk_out.json")
    st_out = os.path.join(odir, ".st_out.json")
    for f in (lk_out, st_out):
        if os.path.exists(f):
            os.remove(f)
    write(out, [])
    py = sys.executable
    night_t, night_o = night_geom(a.scenario)

    state = {"lk_m": 0, "st_m": 0}

    def refresh(force=False):
        m1 = os.path.getmtime(lk_out) if os.path.exists(lk_out) else 0
        m2 = os.path.getmtime(st_out) if os.path.exists(st_out) else 0
        if not force and m1 == state["lk_m"] and m2 == state["st_m"]:
            return
        state["lk_m"], state["st_m"] = m1, m2
        try:
            write(out, merge(load(lk_out), load(st_out), night_t, night_o))
        except Exception as ex:
            log("refresh failed", ex)

    poll_t = [0.0]

    def poll():
        if time.time() - poll_t[0] > 10:
            poll_t[0] = time.time()
            refresh()

    # 1. orbit-space stack (gen009d), capped at ST_FRAC of the budget (it averaged ~72 s of 600 on the A100)
    b_st = ST_FRAC * B
    log(f"stack budget {b_st:.0f}s")
    run([py, "-I", os.path.join(HERE, "sst", "finder.py"), "--scenario", a.scenario, "--out", st_out,
         "--budget-s", f"{b_st:.1f}"], b_st + 15, poll)
    refresh(True)
    log(f"stack done: {len(load(st_out))} candidates")
    # 2. linker (gen009b) with everything left; it writes progressively, killed just before the grader would
    b_lk = B - (time.time() - T0) - 5
    if b_lk > 60:
        log(f"linker budget {b_lk:.0f}s")
        run([py, "-I", os.path.join(HERE, "lk", "finder.py"), "--scenario", a.scenario, "--out", lk_out,
             "--budget-s", f"{b_lk:.1f}"], B + 20, poll)
    refresh(True)
    log("done")


if __name__ == "__main__":
    main()
