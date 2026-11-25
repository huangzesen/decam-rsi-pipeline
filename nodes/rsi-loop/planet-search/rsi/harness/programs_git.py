#!/usr/bin/env python3
"""Keep a git history of every program the planet RSI agents submitted (2026-11-04).

One commit per generation, in the bare repository runs/<run>/programs.git. A commit's parent is the program that round
really started from, so the history is a tree: a round that scored worse stays as a side branch and the next round
starts from the best program again. Tags: `gen071` for every round, `record-0.1880-gen071` for each new best.
Branches: `main` is the best program so far (ties go to the newest, as in rsi_loop.best), `latest` the newest round.

Only finished rounds are committed (a reward or skipped.txt exists). Running it again adds the new rounds and touches
nothing else. Sealed truth and grade.json never enter the repository: only submission/ is stored, plus the reward and
the weighted hit count in the commit message.

Browse:  git -C runs/r1/programs.git log --graph --oneline --decorate main
         git -C runs/r1/programs.git diff gen070 gen071 -- stackd/finder.py
Usage:   python3 -I programs_git.py [--run r1]
"""
import glob, hashlib, json, os, re, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else "r1"
PRIV = os.path.join(HERE, "..", "runs", RUN)
GENS = os.path.join(PRIV, "gens")
REPO = os.path.join(PRIV, "programs.git")
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
BASE = re.compile(r"copy of the best earlier program, `history/(gen\w+)`|the bare starter program")


def git(*args, env=None, cwd=None, stdin=None):
    e = dict(os.environ, GIT_DIR=REPO)
    e.update(env or {})
    return subprocess.run(["git", *args], env=e, cwd=cwd, check=True, capture_output=True, text=True, input=stdin).stdout.strip()


BIG = 5e6   # since 2026-11-05 submissions may be 500 MB (network weights): bigger files stay out of git, a stub stands in


def add_submission(sub, env):
    """Stage sub/ in the temporary index; each file over BIG becomes `<name>.omitted.txt` with its size and SHA-256."""
    big = []
    for r, _, fs in os.walk(sub):
        for f in fs:
            p = os.path.join(r, f)
            if not os.path.islink(p) and os.path.getsize(p) > BIG:
                big.append(os.path.relpath(p, sub))
    git("add", "-A", "--", ".", *[f":(exclude,literal){b}" for b in big], env=env, cwd=sub)
    for b in big:
        p, h = os.path.join(sub, b), hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        note = (f"{b}: {os.path.getsize(p)} bytes, sha256 {h.hexdigest()}\n"
                f"Left out of git because it is over {BIG / 1e6:.0f} MB; the file is in the private run record.\n")
        blob = git("hash-object", "-w", "--stdin", stdin=note)
        git("update-index", "--add", "--cacheinfo", f"100644,{blob},{b}.omitted.txt", env=env, cwd=sub)
    return big


def order(g):
    return int(re.match(r"gen(\d+)", g).group(1)), g


def read_json(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def base_from_prompt(g):
    """The base named in the round's own prompt, or None when its prompt does not name one (rounds up to gen008)."""
    try:
        with open(os.path.join(GENS, g, "transcript.jsonl"), errors="replace") as f:
            for i, line in enumerate(f):
                m = BASE.search(line)
                if m:
                    return m.group(1) or "gen000"
                if i > 30:
                    break
    except OSError:
        pass
    return None


def best_before(g, rewards):
    """What rsi_loop.best would have picked before generation g: highest reward, ties to the newest name."""
    earlier = [(r, name) for name, r in rewards.items() if order(name)[0] < order(g)[0] and r is not None]
    return max(earlier)[1] if earlier else None


def message(g, reward, base, inferred):
    d = os.path.join(GENS, g)
    grade = read_json(os.path.join(d, "grade", "grade.json")) or {}
    rw = read_json(os.path.join(d, "reward.json")) or {}
    ss = read_json(os.path.join(d, "session.json")) or {}
    try:
        notes = open(os.path.join(d, "submission", "NOTES.md"), errors="replace").read(4000)
    except OSError:
        notes = ""
    change = re.search(r"^Change:\s*(.*)$", notes, re.M)
    head = f"{g}: reward {reward:.4f}" if reward is not None else f"{g}: not graded"
    if grade.get("truth_weight"):
        head += f" ({grade.get('hit_weight'):g} of {grade.get('truth_weight'):g})"
    lines = [head, ""]
    if base:
        lines.append(f"Base: {base}" + (" (inferred: this round's prompt does not name it)" if inferred else ""))
    if change:
        lines.append("Change: " + change.group(1).strip())
    extra = []
    if ss.get("seconds"):
        extra.append(f"session {round(ss['seconds'] / 60)} min")
    if rw.get("graded_on"):
        extra.append(f"graded on {rw['graded_on']}")
    if rw.get("status") and rw["status"] != "ok":
        extra.append(f"status {rw['status']}")
    if os.path.exists(os.path.join(d, "skipped.txt")):
        extra.append("skipped")
    if extra:
        lines.append(", ".join(extra))
    return "\n".join(lines + ["", TRAILER, ""])


def main():
    if not os.path.isdir(REPO):
        subprocess.run(["git", "init", "--quiet", "--bare", "--initial-branch=main", REPO], check=True)
        with open(os.path.join(REPO, "info", "exclude"), "a") as f:
            f.write("__pycache__/\n*.pyc\n.DS_Store\n")
    have = set(git("tag", "--list", "gen*").split())
    names = sorted((g for g in os.listdir(GENS) if os.path.isdir(os.path.join(GENS, g)) and re.match(r"gen\d", g)), key=order)
    rewards = {}
    for g in names:
        rw = read_json(os.path.join(GENS, g, "reward.json"))
        if rw is not None:
            rewards[g] = rw.get("reward") if rw.get("reward") is not None else -1.0
    added, inferred_n, newest = 0, 0, None
    for g in names:
        d = os.path.join(GENS, g)
        sub = os.path.join(d, "submission")
        done = g in rewards or os.path.exists(os.path.join(d, "skipped.txt")) or g == "gen000"
        if not os.path.isdir(sub) or not done:
            continue
        newest = g
        if g in have:
            continue
        base = base_from_prompt(g)
        inferred = base is None and g != "gen000"
        if inferred:
            base = best_before(g, rewards)
        if base is not None and base not in have:
            print(f"{g}: base {base} is not in the repository; skipped")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            env = {"GIT_INDEX_FILE": os.path.join(tmp, "index"), "GIT_WORK_TREE": sub}
            if add_submission(sub, env):
                print(f"{g}: files over {BIG / 1e6:.0f} MB stored as stubs")
            tree = git("write-tree", env=env, cwd=sub)
        stamp = next((p for p in (os.path.join(d, "reward.json"), os.path.join(d, "skipped.txt"), os.path.join(d, "session.json"), sub) if os.path.exists(p)))
        t = os.path.getmtime(stamp)
        date = f"{int(t)} {time.strftime('%z', time.localtime(t))}"
        who = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
        if g != "gen000":
            who.update(GIT_AUTHOR_NAME=f"{g} agent (Claude Opus 5.5)", GIT_AUTHOR_EMAIL="noreply@anthropic.com")
        reward = rewards.get(g)
        msg = message(g, None if reward == -1.0 else reward, base, inferred)
        parents = ["-p", git("rev-parse", base)] if base else []
        commit = subprocess.run(["git", "commit-tree", tree, *parents], input=msg, text=True, capture_output=True, check=True,
                                env=dict(os.environ, GIT_DIR=REPO, **who)).stdout.strip()
        git("tag", g, commit)
        have.add(g)
        added += 1
        inferred_n += inferred
    # record tags and the two branches are worked out afresh, so an interrupted run heals itself
    top, best = -1.0, None
    for g in sorted((g for g in rewards if g in have), key=order):
        r = rewards[g]
        if r >= top:
            if r > top + 1e-9:
                git("tag", "--force", f"record-{r:.4f}-{g}", g)
            top, best = r, g
    if best:
        git("update-ref", "refs/heads/main", git("rev-parse", best))
    if newest and newest in have:
        git("update-ref", "refs/heads/latest", git("rev-parse", newest))
    if added:
        git("gc", "--quiet")
    print(f"added {added} rounds ({inferred_n} with an inferred base); best {best} {top:.4f}; newest {newest}")


if __name__ == "__main__":
    main()
    if "--run" not in sys.argv:   # the refresher job calls it bare: keep every other run's own history too (r2 from 2026-11-09)
        runs = os.path.join(HERE, "..", "runs")
        for r in sorted(os.listdir(runs)):
            if r != RUN and glob.glob(os.path.join(runs, r, "gens", "gen*", "reward.json")):   # once a round is graded
                subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--run", r])
