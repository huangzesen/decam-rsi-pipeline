#!/usr/bin/env python3
"""conveyor (NERSC): reorder a unit's NOIRLab download list region by region (2026-11-20).

The unit's fetch jobs split missing.tsv into blocks of 32 exposures in file order (fetch.py --part k/N), so the order
decides which regions become complete first. This writes missing.tsv anew with only the files still to fetch (not in any
share's done log, not already on disk at full size), the regions with the fewest missing exposures first, each region's
exposures together, so the N fetchers finish one region after another and each can be built and searched at once
(early build). SHARE_DONE markers of shares that get files again are removed. The old list is kept as
missing.tsv.bak-<time>. Prints the plan: regions in order with their gaps and the cumulative files.
Usage: python3 -I prio_missing.py UNIT_DIR REGIONS_EXPOSURES.json N_SHARES [--dry]"""
import glob, json, os, sys, time
U, RX, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); dry = "--dry" in sys.argv
rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(U, "missing.tsv")) if l.strip()]
done = set()
for f in glob.glob(os.path.join(U, "p[0-9]*", "log", "done_*.txt")):
    done.update(x.strip() for x in open(f))
present = {}
for p in glob.glob(os.path.join(U, "p[0-9]*", "img", "*", "c4d_*.fits.fz")):
    present[os.path.basename(p)] = os.path.getsize(p)
rem = [r for r in rows if r[2] not in done and present.get(r[2]) != int(r[1])]
by_stem = {}
for r in rem:
    by_stem.setdefault(r[2][:17], []).append(r)
reg = json.load(open(RX))
gap = {rid: sorted({s for s in stems if s in by_stem}) for rid, stems in reg.items()}
order, seen, plan, cum = [], set(), [], 0
for rid in sorted((r for r in gap if gap[r]), key=lambda r: (len(gap[r]), r)):
    new = [s for s in gap[rid] if s not in seen]
    for s in new:
        order += by_stem[s]; seen.add(s); cum += len(by_stem[s])
    plan.append((rid, len(gap[rid]), len(new), cum))
left = [r for s, rs in by_stem.items() if s not in seen for r in rs]          # exposures in no region (should be none)
order += left
complete_now = sorted(r for r in reg if not gap[r])
print(f"{len(rows)} files listed, {len(rows) - len(rem)} already fetched, {len(rem)} to fetch "
      f"({sum(int(r[1]) for r in rem) / 1e12:.2f} TB); regions complete already: {len(complete_now)}; to complete: {len(plan)}; "
      f"files outside regions: {len(left)}")
for rid, g, nn, c in plan[:12]:
    print(f"  {rid}: {g} exposures missing ({nn} new), done after {c} files")
if len(plan) > 12:
    print(f"  ... last: {plan[-1][0]} after {plan[-1][3]} files")
if not dry:
    t = time.strftime("%Y%m%d-%H%M")
    os.replace(os.path.join(U, "missing.tsv"), os.path.join(U, f"missing.tsv.bak-{t}"))
    with open(os.path.join(U, "missing.tsv"), "w") as o:
        o.write("".join("\t".join(r) + "\n" for r in order))
    # every share k < N gets files again: clear its SHARE_DONE so the conveyor resubmits it
    nexp = -1; last = None; got = set()
    for r in order:
        if r[2][:17] != last:
            last = r[2][:17]; nexp += 1
        got.add((nexp // 32) % N)
    for k in sorted(got):
        f = os.path.join(U, f"p{k}", "log", "SHARE_DONE")
        if os.path.exists(f):
            os.remove(f); print(f"  cleared p{k}/log/SHARE_DONE")
    print(f"written {U}/missing.tsv ({len(order)} files); old list {U}/missing.tsv.bak-{t}")
