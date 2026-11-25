"""TNO finder, generation 10c: hybrid of two earlier programs, run one after the other.

1. link/  = gen009b (detect-then-link, HelioLinC/BK frame, null-calibrated p), given LINK_FRAC of the budget.
2. stack/ = gen009d (orbit-space GPU shift-and-stack + CEM refinement), given the rest of the budget.
3. Merge: stack candidates whose sky track matches a linker candidate on >= 3 nights confirm it (p boosted) and are
   dropped; the others are added with p scaled by STACK_SCALE. Sky-track dedup, top 200.
OUT.json is written after each stage, so a kill at any point leaves a valid file.
"""
import argparse, csv, json, math, os, subprocess, sys, time
import numpy as np

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "link"))
LINK_FRAC = float(os.environ.get("TNO_LINKFRAC", 0.7))
STACK_SCALE = float(os.environ.get("TNO_STACKSCALE", 0.5))
GM = 2.959122082855911e-4


def log(*a):
    print(f"[wrap {time.time() - T0:6.1f}s]", *a, flush=True)


def valid_state(c):
    try:
        s = [float(v) for v in c["state"]]
        if len(s) != 6 or not all(map(math.isfinite, s + [float(c["epoch_mjd"]), float(c["p"])])):
            return False
        r = math.sqrt(sum(v * v for v in s[:3]))
        return r >= 25.0 and sum(v * v for v in s[3:]) / 2 - GM / r < 0 and 0 <= float(c["p"]) <= 1
    except Exception:
        return False


def load(path):
    try:
        c = json.load(open(path))["candidates"]
        return sorted([x for x in c if valid_state(x)], key=lambda x: -float(x["p"]))
    except Exception:
        return []


def run_stage(name, script_dir, scen, out, budget, hard):
    env = dict(os.environ)
    cmd = [sys.executable, "-I", os.path.join(script_dir, "finder.py"), "--scenario", scen, "--out", out,
           "--budget-s", f"{budget:.1f}"]
    log(name, "start, budget", round(budget), "hard", round(hard))
    p = subprocess.Popen(cmd, cwd=script_dir, env=env, start_new_session=True)
    try:
        p.wait(timeout=max(5.0, hard))
        log(name, "exit", p.returncode)
    except subprocess.TimeoutExpired:
        log(name, "timeout: killing")
        try:
            os.killpg(p.pid, 9)
        except Exception:
            p.kill()
        p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    a = ap.parse_args()
    B = a.budget_s
    scen = os.path.abspath(a.scenario)
    out = os.path.abspath(a.out)
    odir = os.path.dirname(out)
    out_l = os.path.join(odir, ".link_out.json")
    out_s = os.path.join(odir, ".stack_out.json")

    rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
    nt = {}
    for r in rows:
        t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
        nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
    night_t = np.array([v[0] for v in nt.values()])
    night_o = np.array([v[1] for v in nt.values()])

    def write(L, S):
        cs = [dict(c, src="link") for c in L]
        try:
            cs = merge(cs, S, night_t, night_o)
        except Exception as ex:
            log("merge failed", ex)
        cs = [c for c in cs if valid_state(c)][:200]
        log("top 30 sources/p:", [(c.get("src", "?"), round(float(c["p"]), 3)) for c in cs[:30]])
        cs = [{"epoch_mjd": float(c["epoch_mjd"]), "state": [float(v) for v in c["state"]],
               "p": float(min(1.0, max(0.0, c["p"]))), **({"mag": c["mag"]} if c.get("mag") is not None else {})}
              for c in cs]

        tmp = out + ".tmp"
        json.dump({"candidates": cs}, open(tmp, "w"))
        os.replace(tmp, out)
        log("wrote", len(cs), "candidates", "(stack-only:", sum(c.get("src") == "stack" for c in cs), ")")

    write([], [])
    run_stage("link", os.path.join(HERE, "link"), scen, out_l, LINK_FRAC * B, LINK_FRAC * B + 25)
    L = load(out_l)
    write(L, [])
    left = B - (time.time() - T0) - 12
    if left > 45:
        run_stage("stack", os.path.join(HERE, "stack"), scen, out_s, left, min(left + 20, B + 15 - (time.time() - T0)))
        S = load(out_s)
        log("stack candidates", len(S), "top p", [round(float(c["p"]), 3) for c in S[:8]])
        write(L, S)
    log("done")


def merge(L, S, night_t, night_o, tol=3.0):
    import orbit
    import importlib.util
    spec = importlib.util.spec_from_file_location("link_finder", os.path.join(HERE, "link", "finder.py"))
    lf = importlib.util.module_from_spec(spec); spec.loader.exec_module(lf)
    dedup_sky = lf.dedup_sky
    if S:
        S = [dict(c, src="stack") for c in S[:400]]
        if L:
            def dirs(cs):
                D = np.zeros((len(cs), len(night_t), 3))
                for ep in set(round(float(c["epoch_mjd"]), 6) for c in cs):
                    idx = [k for k, c in enumerate(cs) if round(float(c["epoch_mjd"]), 6) == ep]
                    D[idx] = orbit.predict_dirs_b(np.array([cs[k]["state"] for k in idx], float), ep,
                                                  night_t, night_o)
                return D
            DL, DS = dirs(L[:1500]), dirs(S)
            need = min(3, len(night_t))
            cosl = math.cos(tol * orbit.ARC)
            cnt = np.zeros((len(S), len(DL)), np.int16)
            for n in range(len(night_t)):
                cnt += (DS[:, n] @ DL[:, n].T > cosl)
            keep = []
            for i, c in enumerate(S):
                js = np.nonzero(cnt[i] >= need)[0]
                if len(js):
                    j = int(js[0])            # highest-p linker match: independent confirmation
                    pl, ps = float(L[j]["p"]), float(c["p"])
                    L[j]["p"] = 1 - (1 - pl) * (1 - 0.5 * ps)
                    L[j]["src"] = "link+stack"
                else:
                    c["p"] = STACK_SCALE * float(c["p"])
                    keep.append(c)
            S = keep
        else:
            for c in S:
                c["p"] = STACK_SCALE * float(c["p"])
    cs = sorted(L + S, key=lambda c: -float(c["p"]))
    cs = dedup_sky(cs, night_t, night_o)
    return cs


if __name__ == "__main__":
    main()
