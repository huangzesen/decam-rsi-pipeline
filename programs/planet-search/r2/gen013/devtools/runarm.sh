#!/bin/zsh
# runarm.sh ARM SUBDIR BUDGET [ENV=VAL ...] : run the four dev scenarios from cache with injection
T=$RSI_SANDBOX/r2/gen010/tmp; DEV=$DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev
arm=$1; sub=$2; bud=$3; shift 3
for s in ${=SCENS:-scen01 scen02 scen03 scen04}; do mkdir $T/run/${arm}_$s 2>/dev/null
(cd $sub && env "$@" FINDER_INJECT=$T/inj/$s.json FINDER_CACHE=$T/cache/$s FINDER_DUMPC=$T/run/${arm}_$s OMP_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 nohup $HOME/anaconda3/bin/python finder.py --scenario $DEV/$s --out $T/run/${arm}_$s/OUT.json --budget-s $bud > $T/run/${arm}_$s/log 2>&1 &)
done
