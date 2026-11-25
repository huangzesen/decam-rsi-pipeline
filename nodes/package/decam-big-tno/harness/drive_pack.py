#!/usr/bin/env python3
"""Package the decam-big-tno hill climb (run r2) with every trial's full trace into a Google Drive folder (2026-11-22). Same pattern as harbor_tasks_projects/tbsci-mock/datacard/*/upload.py: build on
Crucial, copy into the Drive for Desktop folder My Drive/projects/<NAME>/, then read Drive's SHA-256 back with rclone
(hashes only) until every file matches.

Contents (rule of harbor_tasks_projects/AGENTS.md, "every task zip holds full traces"):
  decam-big-tno-r2-hillclimb.zip
    task/                       the Harbor task base as on the trial host (instruction, task.toml, environment with the
                                1.6 GB sample patch, tools, solution, tests)
    submission/rollouts/<trial>/  every trial of the run (81 = 75 rounds, 6 relaunches): agent sessions, trajectory,
                                result.json, config.json, artifacts (the program, validation); the Harbor job files in _job/.
                                Left out for safety only: trial.log, lock.json, agent/setup/; Codex's encrypted reasoning and
                                key-like base64 blobs become a marker ($GCP_PROJECT/tools/pack.py redact_encrypted).
    run/                        the climb's own record: per round grade (grade.json, public.json), graded runs on the four
                                patches (grade_runs/), reward, the climb log, config, state, review.json (our reading of
                                every program), the extra history given to agents, the per-body loss tables
  decam-big-tno-open-regions.zip  the 23 labelled patches mounted read-only at /app/open in every trial (49 GB, stored)
  README.md, trials.csv, SHA256SUMS.txt

Stages (each one resumable): fetch (task base and open patches from the trial host), stage, zip, check, upload, verify.
    python3 -I drive_pack.py all          # or one stage name
"""
import csv, glob, hashlib, json, os, re, shutil, subprocess, sys, time, zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))         # rsi/decam-big-tno
RUN = os.path.join(HERE, "runs", "r2")
NAME = "decam-big-tno-hillclimb-r2-2026-11-22"
WORK = f"$DATA_DRIVE/{NAME}"
DST = os.path.expanduser(f"$DRIVE_FOLDER/projects/{NAME}")
REMOTE = f"gdrive-share:projects/{NAME}"
HOST = "$VM_HOST"
SSH = "ssh -o ControlPath=none -o BatchMode=yes"                          # never touch the shared mux socket
TASK_SRC = "$VM_WORK/tasks-ale/decam-big-tno-r2/"
OPEN_SRC = "$VM_WORK/tasks-ale/decam-big-tno-open/"
MAIN = "decam-big-tno-r2-hillclimb.zip"; OPEN = "decam-big-tno-open-regions.zip"
RUN_FILES = ["climb.log", "config.json", "state.json", "review.json", "history_extra_note.md", "continue_to_38.sh",
             "continue_to_75.sh"]
RUN_DIRS = ["history_extra", "eng", "loss", "aborted", "video", "test"]
DROP = {"trial.log", "lock.json"}


def say(*a):
    line = " ".join(str(x) for x in (time.strftime("%m-%d %H:%M:%S"), *a))
    print(line, flush=True)
    os.makedirs(WORK, exist_ok=True)
    with open(f"{WORK}/run.log", "a") as f:
        f.write(line + "\n")


def sh(cmd):
    say("$", cmd if len(cmd) < 200 else cmd[:200] + " ...")
    subprocess.run(cmd, shell=True, check=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def redact(run_dir):
    """$GCP_PROJECT/tools/pack.py's redact_encrypted, copied so that importing pack.py (which needs the delivery skill) is not needed."""
    rx = re.compile(r'("encrypted_content"\s*:\s*")[^"]*(")')
    blob = re.compile(r"[A-Za-z0-9_\-+/=]{200,}")
    keyish = re.compile(r"sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|[a-z0-9]{3,8}p_[A-Za-z0-9_-]{20,}|AK[A-Z]{2}[A-Za-z0-9_\-+/=]{20,}")
    n = 0
    for root, _, fs in os.walk(run_dir):
        for f in fs:
            p = os.path.join(root, f)
            if f.endswith((".jsonl", ".json", ".txt")):
                s = open(p, errors="replace").read()
                t = rx.sub(r"\g<1>[encrypted reasoning omitted]\g<2>", s) if "encrypted_content" in s else s
                if keyish.search(t):
                    t = blob.sub(lambda m: f"[opaque base64 blob omitted, {len(m.group(0))} chars]"
                                 if keyish.search(m.group(0)) else m.group(0), t)
                if t is not s:
                    open(p, "w").write(t); n += 1
    return n


def trials():
    """Every trial folder of the run: runs/r2/gens/<gen>/job-<a>/<trial>__<id>/."""
    out = []
    for jd in sorted(glob.glob(f"{RUN}/gens/gen*/job-*")):
        for td in sorted(glob.glob(f"{jd}/*__*")):
            if os.path.isdir(td):
                out.append((os.path.basename(os.path.dirname(jd)), os.path.basename(jd), td))
    return out


def fetch():
    os.makedirs(f"{WORK}/stage", exist_ok=True)
    sh(f"rsync -a --delete --exclude='*.bak*' -e '{SSH}' {HOST}:{TASK_SRC} {WORK}/stage/task/")
    os.makedirs(f"{WORK}/open", exist_ok=True)
    sh(f"rsync -a --partial -e '{SSH}' {HOST}:{OPEN_SRC} {WORK}/open/decam-big-tno-open/")
    say("fetched", subprocess.run(f"du -sh {WORK}/stage/task {WORK}/open", shell=True, capture_output=True, text=True).stdout.replace("\n", "; "))


def stage():
    S = f"{WORK}/stage"
    for d in ("submission", "run"):
        shutil.rmtree(f"{S}/{d}", ignore_errors=True)
    rows, red = [], 0
    for gen, job, td in trials():
        tid = os.path.basename(td)
        dst = f"{S}/submission/rollouts/{tid}"
        shutil.copytree(td, dst, symlinks=True, ignore=lambda d, names: [n for n in names if n in DROP
                        or (n == "setup" and os.path.basename(d) == "agent")])
        os.makedirs(f"{dst}/_job", exist_ok=True)
        for f in ("config.json", "result.json", "job.log"):
            if os.path.exists(f"{os.path.dirname(td)}/{f}"):
                shutil.copy2(f"{os.path.dirname(td)}/{f}", f"{dst}/_job/{f}")
        for root, ds, fs in os.walk(dst):                     # dangling symlinks out
            for n in fs + ds:
                p = os.path.join(root, n)
                if os.path.islink(p) and not os.path.exists(p):
                    os.unlink(p)
        red += redact(dst)
        r = json.load(open(f"{td}/result.json")) if os.path.exists(f"{td}/result.json") else {}
        cfg = json.load(open(f"{td}/config.json")) if os.path.exists(f"{td}/config.json") else {}
        ag = (cfg.get("agent") or {})
        pub = f"{RUN}/gens/{gen}/grade/public.json"
        p = json.load(open(pub)) if os.path.exists(pub) else {}
        rows.append(dict(generation=gen, job=job, trial=tid, result_id=r.get("id", ""), agent=ag.get("name", ""),
                         model=ag.get("model_name", ""), started=r.get("started_at", ""), finished=r.get("finished_at", ""),
                         reward=p.get("reward", ""), graded_status=p.get("status", ""),
                         exception=((r.get("exception_info") or {}).get("exception_type") or "")))
    for g in sorted(glob.glob(f"{RUN}/gens/gen*")):
        shutil.copytree(g, f"{S}/run/gens/{os.path.basename(g)}", symlinks=True,
                        ignore=lambda d, names: [n for n in names if n.startswith("job-") and d == g])
    for f in RUN_FILES:
        if os.path.exists(f"{RUN}/{f}"):
            shutil.copy2(f"{RUN}/{f}", f"{S}/run/{f}")
    for d in RUN_DIRS:
        if os.path.isdir(f"{RUN}/{d}"):
            shutil.copytree(f"{RUN}/{d}", f"{S}/run/{d}", symlinks=False)
    with open(f"{WORK}/trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(f"{WORK}/trials.json", "w"), indent=1)
    say(f"staged {len(rows)} trials ({red} files redacted)")


def keyscan():
    """No live credential in the staged traces: the relay keys' exact values and key-shaped strings."""
    keys = []
    for kf in glob.glob(os.path.expanduser("$RELAY_KEY_DIR/*.key")):
        k = open(kf).read().strip()
        if len(k) >= 16:
            keys.append(k)
    pat = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,200}(?![A-Za-z0-9_-])|AKIA[0-9A-Z]{16}")
    hits = 0
    for root, _, fs in os.walk(f"{WORK}/stage"):
        if "/task/environment/data" in root:
            continue
        for n in fs:
            p = os.path.join(root, n)
            if os.path.islink(p) or os.path.getsize(p) > 200_000_000:
                continue
            try:
                s = open(p, errors="replace").read()
            except Exception:
                continue
            if any(k in s for k in keys) or pat.search(s):
                hits += 1; say("KEY-LIKE STRING in", os.path.relpath(p, WORK))     # the path only, never the string
    say("key scan:", hits, "files flagged")
    return hits


def zip_all(only=None):
    S = f"{WORK}/stage"
    for name, root, comp in ((MAIN, S, zipfile.ZIP_DEFLATED), (OPEN, f"{WORK}/open", zipfile.ZIP_STORED)):
        if only and name != only:
            continue
        out = f"{WORK}/out/{name}"
        if os.path.exists(out) and name == OPEN:
            say("exists, kept:", name); continue
        os.makedirs(f"{WORK}/out", exist_ok=True)
        part = out + ".part"
        with zipfile.ZipFile(part, "w", comp, allowZip64=True, compresslevel=6 if comp == zipfile.ZIP_DEFLATED else None) as z:
            for r, ds, fs in os.walk(root):
                ds.sort()
                for n in sorted(fs):
                    p = os.path.join(r, n)
                    if os.path.islink(p) and not os.path.exists(p):
                        continue
                    a = os.path.relpath(p, root)
                    c = zipfile.ZIP_STORED if (comp == zipfile.ZIP_DEFLATED and n.endswith((".npy", ".npz", ".gz", ".zip", ".png", ".mp4"))
                                               and os.path.getsize(p) > 50_000_000) else comp
                    z.write(p, a, compress_type=c)
        os.replace(part, out)
        say("zipped", name, f"{os.path.getsize(out) / 1e9:.2f} GB")


def check():
    """Every trial of the run is in the zip (by result.json id), and every member reads back (CRC)."""
    want = json.load(open(f"{WORK}/trials.json"))
    with zipfile.ZipFile(f"{WORK}/out/{MAIN}") as z:
        names = set(z.namelist())
        miss = [t["trial"] for t in want if f"submission/rollouts/{t['trial']}/result.json" not in names]
        got_ids = {json.loads(z.read(f"submission/rollouts/{t['trial']}/result.json")).get("id") for t in want
                   if f"submission/rollouts/{t['trial']}/result.json" in names}
        bad = z.testzip()
    ids = {t["result_id"] for t in want if t["result_id"]}
    say(f"check {MAIN}: {len(want)} trials listed, missing {len(miss)} {miss[:5]}, ids matched {len(ids & got_ids)}/{len(ids)}, crc {'ok' if bad is None else 'BAD ' + bad}")
    with zipfile.ZipFile(f"{WORK}/out/{OPEN}") as z:
        bad2 = z.testzip()
    say(f"check {OPEN}: {len(z.namelist())} files, crc {'ok' if bad2 is None else 'BAD ' + bad2}")
    if miss or bad or bad2 or len(ids & got_ids) != len(ids):
        sys.exit("check failed")


def top():
    shutil.copy2(os.path.join(HERE, "harness", "drive_pack_README.md"), f"{WORK}/out/README.md")
    shutil.copy2(f"{WORK}/trials.csv", f"{WORK}/out/trials.csv")
    with open(f"{WORK}/out/SHA256SUMS.txt", "w") as f:
        for n in (MAIN, OPEN, "README.md", "trials.csv"):
            f.write(f"{sha(f'{WORK}/out/{n}')}  {n}\n")
    say("top files written")


def ship(name):
    """2026-11-22: zip one part and copy it into the Drive folder at once."""
    os.makedirs(DST, exist_ok=True)
    if name == "top":
        for src, n in ((os.path.join(HERE, "harness", "drive_pack_README.md"), "README.md"), (f"{WORK}/trials.csv", "trials.csv")):
            shutil.copyfile(src, f"{DST}/{n}"); say("copied to the Drive folder:", n)
        return
    zip_all(only=name)
    d = f"{DST}/{name}"
    shutil.copyfile(f"{WORK}/out/{name}", d + ".part"); os.replace(d + ".part", d)
    say("copied to the Drive folder:", name)


def upload():
    os.makedirs(DST, exist_ok=True)
    for n in ("README.md", "trials.csv", "SHA256SUMS.txt", MAIN, OPEN):
        d = f"{DST}/{n}"
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(f"{WORK}/out/{n}"):
            say("already in the Drive folder:", n); continue
        shutil.copyfile(f"{WORK}/out/{n}", d + ".part"); os.replace(d + ".part", d)
        say("copied to the Drive folder:", n)


def verify(wait=True):
    want = {l.split()[1]: l.split()[0] for l in open(f"{WORK}/out/SHA256SUMS.txt")}
    want["SHA256SUMS.txt"] = sha(f"{WORK}/out/SHA256SUMS.txt")
    while True:
        p = subprocess.run(["rclone", "lsjson", "--hash", "--hash-type", "sha256", "--files-only", REMOTE,
                            "--low-level-retries", "20"], capture_output=True, text=True)
        got = {}
        if p.returncode == 0:
            for x in json.loads(p.stdout or "[]"):
                got.setdefault(x["Name"], []).append((x.get("Hashes") or {}).get("sha256"))
        ok = [n for n in want if want[n] in got.get(n, [])]
        dup = [n for n in got if len(got[n]) > 1]
        say(f"verify: {len(ok)}/{len(want)} match on Drive" + (f"; duplicates {dup}" if dup else "") +
            ("" if p.returncode == 0 else f"; rclone error {p.stderr.strip()[:120]}"))
        if len(ok) == len(want) or not wait:
            return len(ok) == len(want)
        time.sleep(600)


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = dict(fetch=fetch, stage=stage, keyscan=keyscan, zip=zip_all, check=check, top=top, upload=upload, verify=verify)
    if st == "all":
        fetch(); stage()
        if keyscan():
            sys.exit("key-like strings found: look before zipping")
        zip_all(); check(); top(); upload(); verify()
    elif st == "ship":
        ship(sys.argv[2])
    else:
        steps[st]()
