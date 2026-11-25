# tailor: port the best program to each GPU device for maximum efficiency

2026-11-17.

Built 2026-11-18. **No session has run.** Nothing here spends GPU allocation unless the optional grader is used.

## The flow: one command per device

    python3 -I tailor/tailor.py port --target gb200-vista --agent astra           # ~2 h; run it with nohup
    python3 -I tailor/tailor.py port --target gb200-vista --agent astra --dry     # writes the brief only

One Harbor trial on the trial host (2 h, CPU only, no web, the r2 task with the all-open regions mounted, exactly as the
r2 climb's agents run), starting from the hill climb's best program (read from `runs/r2/state.json` at launch; now
gen041, recall 0.7904, 1.87 GPU-h on the A100). Its instruction is the r2 task plus `harness/task_r2/instruction_hw.md`:
single goal, the least GPU time on the device; the device's facts from `targets.json`; the program's own seconds and log
tails on the A100 (`history/a100/`). The only test is the task's CPU sample check (`tools/check.py`). The agent declares a
budget per region in `BUDGET_S.txt` (the program is anytime) and writes `PORT.md` (what changed for the device, expected
effect per stage, what to check first on the device).

Result: `tailor/ports/<target>/program/` (the port), `PORT.md`, `port.json` (device, agent, base, declared budget, scan
result, session minutes); an earlier port moves to `ports/<target>/old/<time>/`; the session's job folder is in
`ports/<target>/sessions/`. A program that reaches the network or names known bodies is still kept but flagged in
`port.json` `scan`.

    python3 -I tailor/tailor.py status       sessions and ports per device
    python3 -I tailor/tailor.py table        one row per device -> tailor/efficiency.json (for organising runs)

The table's reference row is the base on the A100: 1,684 s per region, 0.694 GPU-h per 1,000 exposures (regions in
`regions.json`: 2,698 exposures, 246 M detection rows). Device rows get measured numbers only from the optional grader.

## Plugs

| File | What |
|---|---|
| `agents.json` | `astra` = Codex, openai/gpt-6.2-astra, effort max; `fable` = Claude Code, claude-fable-5-2, effort xhigh. Add a key to plug another |
| `targets.json` | `gb200-vista` (Blackwell, B200-class), `gh200-vista` (Hopper, H100-class), `h100-ls6` (H100 80 GB), `a100-nersc`; `h200` disabled (no H200 in any of our allocations) |

## Optional tools, not part of the flow (they spend allocation; only on the user's explicit word)

| File | What |
|---|---|
| `tailor.py grade --target T --go "<approvals>" [--budget S] [--dry]` | runs `ports/T/program` on the device: the r2 climb's four graded regions at its declared budget, one GPU per region, no network, scored on the Mac against `private/truth_r2x`; pass = recall >= the base's A100 recall (0.7904); GPU-seconds, GPU-h and SU per 1,000 exposures go to the table. `tailor.py cost` prints the SU of one such grading (gb200 ~0.6, gh200 ~2.3, h100-ls6 ~6.3) |
| `sites/tacc_grade.sbatch` | the TACC grader: one region per GPU, a hardware probe per node, refuses where `unshare -rn` (no network) fails |
| `stage.py` | copies the four regions from NERSC to a TACC device with Globus (`--go`); Vista lacks 2 of 4 (5.7 GB), Lonestar6 all 4 (11 GB) |
| `ls6_env.sh`, `nersc_env_freeze.txt` | builds the grading env on Lonestar6 with NERSC's package set; not run |

Untested for the optional grader: `unshare -rn` on Lonestar6 gpu-h100 and Vista gh compute nodes (works on Vista gb), the
4-node gh job, and the Lonestar6 Globus collection with our login.
