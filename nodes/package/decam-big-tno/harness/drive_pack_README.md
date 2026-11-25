# decam-big-tno: hill climb run r2, task and full traces

Packaged 2026-11-22 by the project lead. (This README went with a private package of the grader's labels and candidate positions; the package itself is not in this repository.)

## The task

Agents write a program that finds large trans-Neptunian and more distant bodies in thirteen years (2013-2026) of public
DECam single-exposure detections of one sky patch (a "5 % search region", about 64 million detections). The program
runs on one GPU per patch with a 1,800 s budget. Score: weighted recall of the findable known bodies in the program's
global top list (weight by size), on four graded patches (REG_A, REG_B, R009, R013), with GPU-hours reported as a second,
separate number. Confident candidates that match no known body are leads, checked afterwards as possible discoveries.

The agent works in a Harbor trial: 2 h, 8 CPUs, no GPU, no web. `tools/check.py` runs the program on the sample patch
and checks the output format. Grading ran outside Harbor on NERSC Perlmutter (one A100 per patch, no network).

**All open.** Nothing was hidden from the agents: `/app/open` held all 23 labelled patches, the four graded ones
included, and `/app/history/` held every earlier program of this run with its notes, reward and per-body feedback, plus
the 37 programs of the earlier run r1. Results were trusted and every program was read by hand for cheating
(`run/review.json`).

## The run

- 75 generations, gen001 to gen075, 2026-11-14 to 2026-11-22; 81 Harbor trials. Six generations needed a second
  attempt (gen057 a third), when an attempt left no program or the climb was restarted mid-trial; every attempt is
  included (`job-a1`, `job-a2`, ... in `_job/`).
- Agents: Claude Code on Claude Fable 5.2 (effort xhigh, 20 trials), Codex on GPT 6 Astra (xhigh, 18) and
  GPT 6.2 Astra (max, 43). From gen003 for a time two agents worked side by side, each on its own generation, with
  the same history; later one agent at a time.
- Starting point gen000 = r1's gen019, recall 0.479 at 1.94 GPU-hours.
- Best: **gen067, recall 0.8383**; gen059, gen063, gen065, gen073 and gen075 at 0.8323. Last round gen075: 0.8323 at
  1.90 GPU-hours. gen054 and gen055 left the best program unchanged and were not graded.
- `trials.csv`: one row per trial (round, trial id, Harbor result id, agent, model, start/finish, round reward).

## Files

| File | What |
|---|---|
| `decam-big-tno-r2-hillclimb.zip` | `task/`: the Harbor task base (instruction, task.toml, environment with the 1.6 GB sample patch and tools, solution = the starter program, tests). `submission/rollouts/<trial>/`: every trial, complete: agent sessions, `trajectory.json`, `result.json`, `config.json`, `artifacts/` (the submitted program), Harbor job files in `_job/`. `run/`: the climb's record: `gens/<gen>/grade/` (grade.json per patch, public.json), `gens/<gen>/grade_runs/` (graded runs' outputs and logs), `climb.log`, `config.json`, `state.json`, `review.json`, `history_extra/` (earlier results given to the agents), `loss/`, `eng/`, `video/` |
| `trials.csv` | the trial table |

Left out of the traces, for safety only: `trial.log`, `lock.json`, `agent/setup/`, dangling symlinks. Codex's
provider-encrypted reasoning blobs are replaced by `[encrypted reasoning omitted]`.

## Rerunning a trial

The per-round task copy is not kept: each round's `/app/history/` was built from the earlier rounds' programs, notes
and rewards (all in `run/gens/` and in the rollouts) and `run/history_extra/`. To rerun: unpack `task/`, put a
`history/` folder next to the Dockerfile (`COPY history /app/history`), mount the 23 labelled patches (49 GB, not in this
folder; on request) with a Docker compose override (`volumes: ["<open>:/app/open:ro"]`), and run `harbor run` on the task.
