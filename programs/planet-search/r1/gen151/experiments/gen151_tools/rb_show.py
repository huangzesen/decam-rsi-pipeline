"""print per-set body ranks/p under base and shipped RBSKIP, and the global-cut membership of bodies"""
import sys
sys.argv = [sys.argv[0]] + sys.argv[1:] + ["--diag-none"]
exec(open(__file__.replace("rb_show.py", "rb_replay.py")).read().split("if \"--diag\" in sys.argv:")[0])
for name, env in [("base", dict(RB_SKIP=0.0)), ("ship", dict(RB_SKIP=1.0, RB_Z=3.5, RB_C=0.0, RB_MINF=0.6, RB_RMAX=60, RB_DS=0.0))]:
    print(name, run(env, verbose=True))
