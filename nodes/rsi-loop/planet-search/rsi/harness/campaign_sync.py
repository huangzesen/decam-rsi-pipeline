#!/usr/bin/env python3
"""Bring the planet-campaign repository up to date with finished rounds (2026-11-04). Run by hand; nothing schedules it.

  1. programs_git.py adds new rounds to runs/r1/programs.git
  2. each finished round's trace files are checked for keys and passwords; any hit stops everything before a push
  3. those files are committed to this repository, and main and the program refs are pushed

Usage: python3 -I campaign_sync.py [--no-push]
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RSI = os.path.dirname(HERE)
ROOT = os.path.dirname(RSI)
GENS = os.path.join(RSI, "runs", "r1", "gens")
PROGRAMS = os.path.join(RSI, "runs", "r1", "programs.git")
FILES = ("trace.md", "transcript.jsonl", "session.json", "reward.json", "subagents")
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PATTERNS = [re.compile(p) for p in (r"sk-ant-[A-Za-z0-9_-]{20,}", r"\bsk-[A-Za-z0-9_-]{20,}", r"gh[pousr]_[A-Za-z0-9]{20,}",
                                    r"\bhf_[A-Za-z0-9]{20,}", r"\bAKIA[0-9A-Z]{12,}", r"-----BEGIN [A-Z ]*PRIVATE KEY",
                                    r"Bearer [A-Za-z0-9._-]{20,}", r"ANTHROPIC_AUTH_TOKEN[\"'= :]+[A-Za-z0-9_-]{16,}",
                                    r"\bgt-[A-Za-z0-9_.-]{20,}")]


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], check=True, capture_output=True, text=True).stdout.strip()


def exact_secrets():
    home = os.path.expanduser("~")
    out = []
    for path, read in ((home + "/<relay-settings-file>", lambda t: json.loads(t)["env"]["ANTHROPIC_AUTH_TOKEN"]),
                       (home + "/<pages-password-file>", str.strip), (home + "/<rubin-token-file>", str.strip)):
        try:
            out.append(read(open(path).read()))
        except (OSError, ValueError, KeyError):
            pass
    return [s for s in out if s]


def has_secret(path, exact):
    text = open(path, errors="replace").read()
    return any(s in text for s in exact) or any(p.search(text) for p in PATTERNS)


def round_paths(g):
    d = os.path.join(GENS, g)
    if not (os.path.exists(os.path.join(d, "reward.json")) or os.path.exists(os.path.join(d, "skipped.txt"))):
        return []                                       # still running
    return [os.path.join(d, f) for f in FILES if os.path.exists(os.path.join(d, f))]


def main():
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "programs_git.py")], check=True)
    paths, rounds = [], []
    for g in sorted(os.listdir(GENS)):
        if os.path.isdir(os.path.join(GENS, g)) and re.match(r"gen\d", g):
            p = round_paths(g)
            if p:
                paths += p
                rounds.append(g)
    exact = exact_secrets()
    bad = []
    for p in paths:
        files = [os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs] if os.path.isdir(p) else [p]
        bad += [f for f in files if has_secret(f, exact)]
    if bad:
        raise SystemExit(f"stopped: {len(bad)} trace files look like they hold a key; first {os.path.relpath(bad[0], ROOT)}. Nothing was pushed.")
    git("add", "--", *[os.path.relpath(p, ROOT) for p in paths])
    if git("diff", "--cached", "--name-only", "--", "rsi/runs"):
        newest = sorted(rounds, key=lambda g: (int(re.match(r"gen(\d+)", g).group(1)), g))[-1]
        git("commit", "-m", f"Add round traces through {newest}\n\n{TRAILER}", "--", "rsi/runs")
    if "--no-push" in sys.argv:
        print("committed; not pushed")
        return
    url = git("remote", "get-url", "origin")
    git("push", "--quiet", "origin", "main")
    subprocess.run(["git", "-C", PROGRAMS, "push", "--quiet", "--force", url, "refs/heads/main:refs/heads/programs/best",
                    "refs/heads/latest:refs/heads/programs/latest"], check=True)
    subprocess.run(["git", "-C", PROGRAMS, "push", "--quiet", url, "--tags"], check=True)
    git("fetch", "--quiet", "origin", "--tags")
    print(f"pushed: {len(rounds)} rounds, newest {rounds[-1]}")


if __name__ == "__main__":
    main()
