"""One-off test of the NERSC site: run a graded program on two small eval fields under `interactive` (or `debug`) and
compare its candidates with the same program's Lonestar6 outputs. Usage: python -I test_nersc.py GEN WORKDIR"""
import json, os, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import nersc, rsi_loop as L

g, work = sys.argv[1], sys.argv[2]
sids = json.load(open(L.priv("r1", "config.json")))["eval"][-2:]
gd = os.path.join(work, "gens")
shutil.rmtree(gd, ignore_errors=True)
shutil.copytree(L.priv("r1", "gens", g, "submission"), os.path.join(gd, g, "submission"))
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
print(nersc.available(sids))
s = nersc.Site("test", gd, [g], sids, 900.0, log)
t0 = time.time()
while s.state() != "RUNNING":
    s.tick(); time.sleep(30)
    if time.time() - t0 > 3600:
        s.cancel(); sys.exit("not started in 1 h")
log(f"started after {(time.time() - t0) / 60:.1f} min")
while s.state() != "GONE":
    time.sleep(20)
s.fetch()
print(nersc.ssh(f"tail -6 {s.R}/int.{s.k_int}.log; ls {s.R}").stdout[-900:])
for sid in sids:
    i = json.load(open(os.path.join(gd, g, "ls6_runs", sid, "info.json")))
    j = json.load(open(L.priv("r1", "gens", g, "ls6_runs", sid, "info.json")))
    a = json.load(open(os.path.join(gd, g, "ls6_runs", sid, "candidates.json")))["candidates"]
    b = json.load(open(L.priv("r1", "gens", g, "ls6_runs", sid, "candidates.json")))["candidates"]
    pa, pb = sorted(x["p"] for x in a)[::-1][:5], sorted(x["p"] for x in b)[::-1][:5]
    print(sid, "NERSC", i["status"], i["seconds"], "s", i["host"], "netns", i["netns"], len(a), "| LS6", j["status"], j["seconds"], "s", len(b))
    print("   top p NERSC", [round(x, 3) for x in pa], "LS6", [round(x, 3) for x in pb])
print("TEST DONE")
