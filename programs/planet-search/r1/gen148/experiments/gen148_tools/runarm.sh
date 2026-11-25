#!/bin/bash
# usage: runarm.sh NAME SCEN BUDGET [ENV=VAL ...]
name=$1; scen=$2; B=$3; shift 3
D=$RSI_SANDBOX/r1/gen148/tmp/runs/$name
rm -rf $D; mkdir $D; cp -R $RSI_SANDBOX/r1/gen148/workspace/submission $D/sub; rm -rf $D/sub/experiments
cd $D/sub
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 TNO_KEEP=1 "$@" /usr/bin/time -l $HOME/anaconda3/bin/python -I finder.py --scenario $DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev/$scen --out $D/OUT.json --budget-s $B > $D/log.txt 2>&1
echo "exit $?" >> $D/log.txt
