"""Gen 10b hybrid TNO finder (see NOTES.md).

1. link_finder.py (gen 9b detect-then-link, reward 0.1128) runs as a subprocess with ~62 % of the budget and writes
   OUT.json itself as it goes, so a valid output exists early.
2. ss/finder.py (gen 9d orbit-space GPU shift-and-stack, reward 0.0977 standalone) runs as a second subprocess with
   the remaining time, writing its own JSON.
3. Merge: shift-and-stack candidates that duplicate a linker track (sky track within 4" on >= 3 nights) only boost
   the linker candidate's p; the best new ones are appended with a down-weighted p.
"""
import json, math, os, subprocess, sys, time
import numpy as np

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LINK_FRAC = float(os.environ.get("TNO_LINK_FRAC", 0.62))
SS_SCALE = float(os.environ.get("TNO_SS_SCALE", 0.45))   # p of stack-only candidates = SS_SCALE * p_stack
SS_KEEP = int(os.environ.get("TNO_SS_KEEP", 30))          # stack-only candidates kept at that p per scenario


def log(*a):
    print(f"[wrap {time.time() - T_START:6.1f}s]", *a, flush=True)


def run(cmd, timeout):
    log("run", " ".join(os.path.basename(c) for c in cmd[1:3]), f"timeout {timeout:.0f}s")
    p = subprocess.Popen(cmd, cwd=os.path.dirname(cmd[1]), start_new_session=True)
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log("timeout; killing")
        try:
            os.killpg(p.pid, 9)
        except Exception:
            p.kill()
        p.wait()
    return p.returncode


def load(path):
    try:
        return json.load(open(path))["candidates"]
    except Exception:
        return []


def night_geometry(scen):
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


def merge(lc, sc, scen):
    from link_finder import valid_state, dedup_sky
    lc = sorted([c for c in lc if valid_state(c)], key=lambda c: -c["p"])
    sc = sorted([c for c in sc if valid_state(c)], key=lambda c: -c["p"])[:200]
    if not sc:
        return lc[:200]
    night_t, night_o = night_geometry(scen)
    need = min(3, len(night_t))
    cosl = math.cos(4.0 / 206264.806)
    match = np.zeros(len(sc), bool)
    if lc:
        DL = dirs(lc[:400], night_t, night_o)
        DS = dirs(sc, night_t, night_o)
        cnt = np.zeros((len(sc), len(DL)), np.int16)
        for n in range(len(night_t)):
            cnt += (DS[:, n] @ DL[:, n].T > cosl)
        m = cnt >= need
        match = m.any(1)
        for j in np.flatnonzero(match):
            i = int(np.argmax(m[j]))           # best-p linker candidate it matches
            pl, ps = lc[i]["p"], sc[j]["p"]
            lc[i]["p"] = float(min(0.999, 1 - (1 - pl) * (1 - 0.3 * ps)))
            lc[i]["ss"] = 1
        log(f"{int(match.sum())} stack candidates match linker tracks")
    new = [dict(c) for c, mm in zip(sc, match) if not mm]
    for k, c in enumerate(new):
        c["p"] = float(c["p"] * (SS_SCALE if k < SS_KEEP else 0.02))
        c["src"] = "ss"
    allc = sorted(lc + new, key=lambda c: -c["p"])
    try:
        allc = dedup_sky(allc, night_t, night_o)
    except Exception as ex:
        log("dedup failed", ex)
    return allc[:200]


def main():
    args = sys.argv[1:]
    def arg(name):
        return args[args.index(name) + 1]
    scen = os.path.abspath(arg("--scenario")); out = os.path.abspath(arg("--out"))
    budget = float(arg("--budget-s"))
    py = sys.executable
    b_link = LINK_FRAC * budget
    run([py, os.path.join(HERE, "link_finder.py"), "--scenario", scen, "--out", out, "--budget-s", f"{b_link:.1f}"],
        timeout=b_link + 15)
    lc = load(out)
    log(f"linker done: {len(lc)} candidates")
    rem = budget - (time.time() - T_START) - 25
    if rem < 60 or os.environ.get("TNO_NOSS"):
        log("no time for shift-and-stack"); return
    ss_out = out + ".ss.json"
    run([py, os.path.join(HERE, "ss", "finder.py"), "--scenario", scen, "--out", ss_out, "--budget-s", f"{rem:.1f}"],
        timeout=rem + 8)
    sc = load(ss_out)
    log(f"shift-and-stack done: {len(sc)} candidates")
    if not sc:
        return
    try:
        cs = merge(lc, sc, scen)
    except Exception as ex:
        import traceback; traceback.print_exc()
        log("merge failed; keeping linker output", ex); return
    for c in cs:
        c.pop("ss", None); c.pop("src", None)
    tmp = out + ".tmp"
    json.dump({"candidates": cs}, open(tmp, "w"))
    os.replace(tmp, out)
    try:
        os.remove(ss_out)
    except Exception:
        pass
    log(f"merged output {len(cs)} candidates; top p {[round(c['p'], 3) for c in cs[:8]]}")


if __name__ == "__main__":
    main()
