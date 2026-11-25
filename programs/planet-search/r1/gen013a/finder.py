"""TNO finder, gen 10e: hybrid of two earlier programs, run one after the other, outputs merged.

1. linker/  = gen009b (detect-then-link, HelioLinC/BK frame, null-calibrated p; reward 0.1128).
2. stackd/  = gen009d (orbit-space GPU shift-and-stack + CEM refinement; reward 0.0977, ~72 s/scenario on the A100).
   It runs in the budget the linker leaves unused.
3. Merge: candidates whose predicted sky tracks agree (< 2.5" on >= 3 nights) are one body: keep one, boost its p
   (two independent methods agree).  Stack-only candidates get p * STACK_W.  Top 200 by p.
OUT.json is written after each stage.
"""
import argparse, csv, json, math, os, shutil, signal, subprocess, sys, time
import numpy as np

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker"))
import orbit  # noqa: E402

ARC = math.pi / 180 / 3600
STACK_W = float(os.environ.get("TNO_STACK_W", "0.7"))
MATCH_AS = 4.0   # gen 12e: was 2.5; refined linker seeds may move a little


def log(*a):
    print(f"[wrap {time.time() - T0:6.1f}s]", *a, flush=True)


def run(sub, scen, out, budget, hard, env=None):
    cmd = [sys.executable, "-I", os.path.join(HERE, sub, "finder.py"), "--scenario", scen, "--out", out,
           "--budget-s", f"{budget:.1f}"]
    log("start", sub, f"budget {budget:.0f}s, hard stop {hard:.0f}s")
    t = time.time()
    p = subprocess.Popen(cmd, cwd=os.path.join(HERE, sub), start_new_session=True,
                         env=dict(os.environ, **(env or {})))
    try:
        p.wait(timeout=max(5.0, hard))
    except subprocess.TimeoutExpired:
        log(sub, "over its hard stop: killing")
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            p.kill()
        p.wait()
    log("done", sub, f"rc {p.returncode} in {time.time() - t:.0f}s")


def load(path):
    try:
        return list(json.load(open(path))["candidates"])
    except Exception:
        return []


def valid(c):
    try:
        s = np.array(c["state"], float)
        if s.shape != (6,) or not np.all(np.isfinite(s)) or not math.isfinite(float(c["epoch_mjd"])):
            return False
        r = np.linalg.norm(s[:3])
        return r >= 25 and 0.5 * s[3:] @ s[3:] - orbit.GM / r < 0
    except Exception:
        return False


def tracks(cs, night_t, night_o):
    """(K, Nn, 3) unit vectors at one time per night."""
    if not cs:
        return np.zeros((0, len(night_t), 3))
    out = []
    for c in cs:
        out.append(orbit.predict_dirs(np.array(c["state"], float), float(c["epoch_mjd"]), night_t, night_o))
    return np.array(out)


def merge(A, B, night_t, night_o):
    A = [dict(c, src="L") for c in A if valid(c)]
    B = [dict(c, src="S") for c in B if valid(c)]
    for c in B:
        c["p"] = float(c["p"]) * STACK_W
    if not B:
        return A
    ua, ub = tracks(A, night_t, night_o), tracks(B, night_t, night_o)
    need = min(3, len(night_t))
    keepB = np.ones(len(B), bool)
    nmatch = 0
    if len(A):
        cosm = math.cos(MATCH_AS * ARC)
        for j in range(len(B)):
            dots = np.einsum("knd,nd->kn", ua, ub[j])          # (K, Nn)
            close = (dots > cosm).sum(1)
            hit = np.nonzero(close >= need)[0]
            if len(hit):
                i = hit[np.argmax([A[k]["p"] for k in hit])]
                pa, pb = float(A[i]["p"]), float(B[j]["p"])
                A[i]["p"] = 1 - (1 - pa) * (1 - 0.5 * pb)
                A[i]["src"] = "LS"
                keepB[j] = False
                nmatch += 1
    log(f"merge: {len(A)} linker, {len(B)} stack, {nmatch} stack candidates matched a linker one")
    allc = A + [c for c, k in zip(B, keepB) if k]
    return allc


LEV_NG = int(os.environ.get("TNO_LEV_NG", "1"))
LEV_F = float(os.environ.get("TNO_LEV_F", "0.5"))


def apply_lev(A, path):
    """gen 13a: pixel evidence for linker candidates.  The stack stage samples its 1" S/N maps (all mapped exposures)
    at each linker candidate's own predicted positions and counts the nights with S/N >= 2.5 there (ngood).  A real
    body that the linker saw on >= 3 nights, with a state good to the grader's 2", has ngood >= 2-3; a candidate with
    ngood <= LEV_NG although >= 3 nights are covered is junk or has a state too poor to be graded a hit, so it is
    demoted (p x LEV_F), freeing global top-25N slots for other scenarios' bodies."""
    try:
        d = json.load(open(path))
    except Exception:
        return A
    A = [dict(c) for c in A]
    nd = 0
    for r in d.get("seeds", []):
        i = int(r["li"])
        if 0 <= i < len(A) and r.get("ncn", 0) >= 3 and r["ngood"] <= LEV_NG and float(A[i]["p"]) < 0.98:
            A[i]["p"] = float(A[i]["p"]) * LEV_F
            nd += 1
    log(f"pixel evidence: {len(d.get('seeds', []))} linker candidates scored, {nd} demoted "
        f"(<= {LEV_NG} nights with S/N >= 2.5 on their track)")
    return A


def write(path, cs):
    cs = sorted(cs, key=lambda c: -float(c["p"]))[:200]
    out = []
    for c in cs:
        d = {"epoch_mjd": float(c["epoch_mjd"]), "state": [float(x) for x in c["state"]],
             "p": float(min(max(float(c["p"]), 0.0), 1.0))}
        if c.get("mag") is not None:
            try:
                m = float(c["mag"])
                if math.isfinite(m):
                    d["mag"] = m
            except Exception:
                pass
        out.append(d)
    tmp = path + ".tmp"
    json.dump({"candidates": out}, open(tmp, "w"))
    os.replace(tmp, path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    a = ap.parse_args()
    B = a.budget_s
    scen = os.path.abspath(a.scenario)
    out = os.path.abspath(a.out)
    write(out, [])
    rows = list(csv.DictReader(open(os.path.join(scen, "exposures.csv"))))
    nt = {}
    for r in rows:
        t_ = float(r["mjd_mid"]); n_ = math.floor(t_ - 0.5)
        nt.setdefault(n_, (t_, [float(r["obs_x"]), float(r["obs_y"]), float(r["obs_z"])]))
    night_t = np.array([v[0] for v in nt.values()])
    night_o = np.array([v[1] for v in nt.values()])

    outL, outS = out + ".linker.json", out + ".stack.json"
    # 1. linker (best program); leaves >= 25 % of the budget for the stack stage
    bl = B - max(140.0, 0.25 * B)
    run("linker", scen, outL, bl, bl + 15)
    A = load(outL)
    try:
        write(out, merge(A, [], night_t, night_o))
    except Exception as ex:
        log("write A failed", ex)
        if os.path.exists(outL):
            shutil.copy(outL, out)
    # 2. stack (in what is left)
    bs = B - (time.time() - T0) - 12
    if bs >= 60:
        run("stackd", scen, outS, bs, bs + 6, env={"TNO_SEEDS": outL} if os.path.exists(outL) else None)
        S = load(outS)
        try:
            A = apply_lev(A, outS + ".lev.json")
            final = write(out, merge(A, S, night_t, night_o))
            log(f"final {len(final)} candidates; top p {[round(c['p'], 3) for c in final[:8]]}")
        except Exception as ex:
            import traceback
            traceback.print_exc()
            log("merge failed; keeping linker output", ex)
    else:
        log("no time left for the stack stage")
    for f in (() if os.environ.get("TNO_KEEP") else (outL, outS, outS + ".lev.json")):
        try:
            os.remove(f)
        except Exception:
            pass


if __name__ == "__main__":
    main()
