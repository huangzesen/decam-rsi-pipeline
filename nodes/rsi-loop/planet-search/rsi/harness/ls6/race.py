"""Grade a generation on whichever site gives it a GPU first (2026-11-02).

Queue on Lonestar6 and NERSC at the same time; the first job to start wins and the others are cancelled. Outputs come
back to gens/<name>/ls6_runs/ as with ls6.run_remote; scoring stays on the Mac.
"""
import glob, os, subprocess, time

import ls6, nersc


class LS6Site:
    label = "A100 (Lonestar6)"

    def __init__(self, run, gens_dir, names, sids, cap_s, log):
        self.gens_dir, self.names, self.log = gens_dir, names, log
        self.jid, self.R = ls6.submit(run, gens_dir, names, sids, cap_s, log)

    def tick(self):
        pass

    def state(self):
        try:
            r = ls6.ssh(f"squeue -h -j {self.jid} -o %T", timeout=60)
        except subprocess.TimeoutExpired:
            return "?"
        if r.returncode == 255:
            return "?"
        st = r.stdout.strip()
        return "GONE" if not st else "RUNNING" if st in ("RUNNING", "COMPLETING") else "PENDING"

    def cancel(self):
        ls6.ssh(f"scancel {self.jid}")

    def fetch(self):
        ls6.fetch(self.jid, self.R, self.gens_dir, self.names, self.log)


def available(sids):
    """(True, '') when at least one site can take the job."""
    res = [m.available(sids) for m in (ls6, nersc)]
    return (True, "") if any(r[0] for r in res) else (False, "; ".join(r[1] for r in res))


def once(run_, gens_dir, names, sids, cap_s, log):
    sites = []
    for mod, cls in ((nersc, nersc.Site), (ls6, LS6Site)):
        ok, why = mod.available(sids)
        if not ok:
            log(f"{mod.__name__} not used this time: {why}")
            continue
        try:
            sites.append(cls(run_, gens_dir, names, sids, cap_s, log))
        except Exception as e:
            log(f"{mod.__name__} not used this time: {type(e).__name__}: {e}")
    if not sites:
        raise RuntimeError("no grading site could take the job")
    t0, win = time.time(), None
    while win is None:
        for s in sites:
            try:
                s.tick()
            except subprocess.TimeoutExpired:
                pass
        time.sleep(60)
        states = [s.state() for s in sites]
        run = [s for s, st in zip(sites, states) if st == "RUNNING"]
        if run:
            win = run[0]
            log(f"{win.label} started first, after {(time.time() - t0) / 60:.0f} min in the queue")
            for s in sites:
                if s is not win:
                    s.cancel()
        elif time.time() - t0 > 24 * 3600:
            for s in sites:
                s.cancel()
            raise RuntimeError("no site started the job within 24 h; cancelled")
    down = None
    while True:
        time.sleep(60)
        st = win.state()
        if st == "?":
            if not down:
                down = time.time(); log(f"{win.label} unreachable; waiting for the master to come back")
            continue
        if down:
            log(f"{win.label} reachable again after {(time.time() - down) / 60:.0f} min"); down = None
        if st == "GONE":
            break
    win.fetch()
    return win.label


def run(run_, gens_dir, names, sids, cap_s, log):
    """Returns the label of the site that ran the job. One retry when a job ended without every run's info.json."""
    for attempt in (1, 2):
        label = once(run_, gens_dir, names, sids, cap_s, log)
        n = [len(glob.glob(os.path.join(gens_dir, g, "ls6_runs", "*", "info.json"))) for g in names]
        if all(k == len(sids) for k in n):
            break
        log(f"{label}: only {n} of {len(sids)} runs came back" + ("; trying once more" if attempt == 1 else ""))
    return label
