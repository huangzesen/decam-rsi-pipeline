"""Gen 10d hybrid: run the gen-9b detect-then-link pipeline (linker/), then the gen-9d orbit-space GPU
shift-and-stack (stackd/) in the remaining budget, and merge the two candidate lists.

OUT.json is written by the linker itself from the start (atomic replaces), so a crash or overrun of the stack stage
leaves the gen-9b output in place.  The merge is also an atomic replace.
"""
import argparse, csv, json, math, os, signal, subprocess, sys, time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker"))

# fraction of the budget given to the linker; the stack gets what is left after the linker finishes
LIN_FRAC = float(os.environ.get("TNO_LINFRAC", "0.72"))
STACK_MIN_S = 70.0          # do not start the stack stage with less than this
MERGE_RESERVE_S = 12.0
MATCH_ARCSEC = 3.0
# unmatched stack candidates: p = STACK_SCALE * p_stack (stack p is an uncalibrated sigmoid of a z-score)
STACK_SCALE = float(os.environ.get("TNO_STACKSCALE", "0.35"))


def log(*a):
    print(f"[wrap {time.time() - T0:6.1f}s]", *a, flush=True)


def run_child(script, scen, out, budget, timeout, env):
    cmd = [sys.executable, "-I", os.path.basename(script), "--scenario", scen, "--out", out, "--budget-s", f"{budget:.1f}"]
    p = subprocess.Popen(cmd, cwd=os.path.dirname(script), env=env, start_new_session=True)
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"{os.path.basename(os.path.dirname(script))} over time, killing")
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        p.wait()
    return p.returncode


def load(path):
    try:
        return list(json.load(open(path))["candidates"])
    except Exception:
        return []


def valid(c):
    try:
        s = [float(v) for v in c["state"]]
        if len(s) != 6 or not all(map(math.isfinite, s + [float(c["epoch_mjd"]), float(c["p"])])):
            return False
        r = math.sqrt(sum(v * v for v in s[:3]))
        v2 = sum(v * v for v in s[3:])
        return r >= 25.0 and v2 / 2 - 2.959122082855911e-4 / r < 0
    except Exception:
        return False


def merge(scen, lin, stk):
    import numpy as np
    import orbit
    rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
    nt = {}
    for r in rows:
        n = math.floor(float(r["mjd_mid"]) - 0.5)
        nt.setdefault(n, (float(r["mjd_mid"]), [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
    night_t = np.array([v[0] for v in nt.values()])
    night_o = np.array([v[1] for v in nt.values()])
    need = min(3, len(night_t))
    lin = sorted([c for c in lin if valid(c)], key=lambda c: -float(c["p"]))
    stk = sorted([c for c in stk if valid(c)], key=lambda c: -float(c["p"]))
    if not stk:
        return lin

    def dirs(cs):
        D = np.zeros((len(cs), len(night_t), 3))
        for ep in set(round(float(c["epoch_mjd"]), 6) for c in cs):
            idx = [k for k, c in enumerate(cs) if round(float(c["epoch_mjd"]), 6) == ep]
            S = np.array([cs[k]["state"] for k in idx], float)
            D[idx] = orbit.predict_dirs_b(S, ep, night_t, night_o)
        return D

    cosl = math.cos(MATCH_ARCSEC * orbit.ARC)
    out = [dict(c) for c in lin]
    if lin:
        DL, DS = dirs(lin), dirs(stk)
        cnt = np.zeros((len(stk), len(lin)), np.int16)
        for n in range(len(night_t)):
            cnt += (DS[:, n] @ DL[:, n].T > cosl)
        match = cnt >= need
    else:
        match = np.zeros((len(stk), 0), bool)
    nboost = nnew = 0
    used_lin = set()
    for i, c in enumerate(stk):
        ps = float(c["p"])
        js = np.nonzero(match[i])[0]
        if len(js):
            j = int(js[0])              # highest-p linker match (lin sorted)
            if j not in used_lin:
                used_lin.add(j)
                pl = float(out[j]["p"])
                out[j]["p"] = round(pl + (1 - pl) * 0.5 * ps, 6)   # independent confirmation by the pixel stack
                nboost += 1
            continue
        d = dict(c)
        d["p"] = round(STACK_SCALE * ps, 6)
        out.append(d)
        nnew += 1
    out.sort(key=lambda c: -float(c["p"]))
    # drop stack candidates duplicating each other / the linker on >= need nights (dedup by decreasing p)
    try:
        import finder as lf          # linker/finder.py
        out = lf.dedup_sky(out, night_t, night_o)
    except Exception as ex:
        log("dedup failed", ex)
    log(f"merge: {len(lin)} linker, {len(stk)} stack, {nboost} boosted, {nnew} stack-only added; out {len(out[:200])}")
    return out[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    a, _ = ap.parse_known_args()
    scen = os.path.abspath(a.scenario)
    out = os.path.abspath(a.out)
    B = a.budget_s
    env = dict(os.environ)
    tmpdir = env.get("TMPDIR") or os.path.dirname(out)
    try:
        json.dump({"candidates": []}, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    except Exception:
        pass

    b_lin = LIN_FRAC * B
    log(f"linker: budget {b_lin:.0f}s")
    rc = run_child(os.path.join(HERE, "linker", "finder.py"), scen, out, b_lin, b_lin + 25, env)
    log(f"linker exit {rc}")

    rem = B - (time.time() - T0) - MERGE_RESERVE_S
    if rem < STACK_MIN_S or os.environ.get("TNO_NOSTACK"):
        log(f"no stack stage ({rem:.0f}s left)")
        return
    sout = os.path.join(tmpdir, f"stack_{os.getpid()}.json")
    log(f"stack: budget {rem:.0f}s")
    rc = run_child(os.path.join(HERE, "stackd", "finder.py"), scen, sout, rem, rem + 8, env)
    log(f"stack exit {rc}")
    stk = load(sout)
    if not stk:
        return
    lin = load(out)
    try:
        json.dump({"candidates": lin}, open(os.path.join(tmpdir, f"lin_{os.getpid()}.json"), "w"))
    except Exception:
        pass
    try:
        cs = merge(scen, lin, stk)
    except Exception as ex:
        log("merge failed", ex)
        return
    json.dump({"candidates": cs}, open(out + ".tmp", "w"))
    os.replace(out + ".tmp", out)
    log("done")


if __name__ == "__main__":
    main()
