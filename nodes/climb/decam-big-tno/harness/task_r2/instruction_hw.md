

## This session: port the program to TARGET_NAME for maximum efficiency

This session is not a round of the open hill climb above. `/app/submission` holds the hill climb's best program
(BASE_GEN: recall BASE_RECALL on the four graded regions, BASE_GPUH GPU-hours on the A100 at 1,800 s per region). Your
one goal: **port it to TARGET_NAME so that it costs as little GPU time as possible there** (the fewest GPU-seconds per
region, and so per 1,000 exposures) for the whole-archive search we will run on this device. Everything above about
the task, the finder interface, the data and the rules still holds; the grading paragraphs about rewards and rounds do
not apply here. Keep what the program finds; spend the session on speed on this device.

### The device

HARDWARE_TEXT

Each region runs on one GPU of this device (CUDA_VISIBLE_DEVICES set to it; the node's other GPUs run other regions at
the same time), no network. You cannot run on it: your machine has 8 CPU cores and no GPU, and nothing will be run on
the device during this session. Work from the facts above, `/app/history/HARDWARE.md`, and the program's own behaviour
on the A100: `/app/history/a100/` has its seconds per region and the last LOG_LINES lines of its log for each graded region.

Use what this device has: its memory size (larger tiles and batches, whole regions resident, fewer host round trips),
its tensor cores and lower precision where results do not change, compiled kernels (Triton, torch.compile, CUDA graphs,
fused steps, streams overlapping copies and compute), the node's CPU cores and architecture (compiled C++ parts built
for it; aarch64 on Grace nodes), and a split of the budget between stages that fits their speed here. The program fills
whatever budget it is given, so:

- **Declare a budget**: one integer in `/app/submission/BUDGET_S.txt`, the seconds per region you expect this device
  needs to find what the base finds in 1,800 s on the A100 (at most MAX_BUDGET).
- Make the program print, briefly, per-stage wall time and GPU memory peak, so its first run on the device shows where
  the time goes.
- It must still run on a CPU-only machine and other GPUs. The only test here is the CPU sample check:
  `python3 /app/tools/check.py` must pass before you finish.

Before you finish, write `/app/submission/PORT.md`: the device, what you changed for it and why, the expected effect on
each stage's time, the budget you declared and why, and what to check first when it runs on the device. Also replace
`NOTES.md` (`Method:`, `Base:`, `Change:` lines first). You have AGENT_H hours; whatever is in `/app/submission` then is
kept.
