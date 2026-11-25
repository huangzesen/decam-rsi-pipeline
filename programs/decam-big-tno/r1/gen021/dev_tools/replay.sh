#!/bin/bash
# replay.sh OUT BUDGET [env...] : growth + carry only, from the dumps of /tmp/w/outA
out=$1; bud=$2; shift 2
mkdir -p $out
cd /app/submission
env FINDER_DEVCACHE=/tmp/w/cache_p4.npz FINDER_RR=39.2,41.8 FINDER_DEBUG=1 FINDER_LOADRAW=/tmp/w/outA/raw6.pkl FINDER_LOAD3=/tmp/w/outA/g3.pkl FINDER_LOADGROWN=/tmp/w/outA/grown.pkl "$@" \
  python3 ${FINDER_PY:-finder.py} --scen /tmp/p4 --out $out --budget-s $bud > $out/log.txt 2>&1
python3 /app/submission/dev_tools/cls.py $out/candidates.json /tmp/p4 $out/found.npy > $out/cls.txt 2>&1
python3 /tmp/w/ana_c.py $out/candidates.json /tmp/p4 > $out/ana.txt 2>&1
