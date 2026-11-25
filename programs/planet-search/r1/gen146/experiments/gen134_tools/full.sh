#!/bin/bash
# full.sh TAG SCEN  (runs $T/code_TAG copy of a submission on $T/scen/SCEN, budget 900, dump vet sidecar)
T=$RSI_SANDBOX/r1/gen134/tmp
TAG=$1; S=$2
cd $T/code_$TAG
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 TNO_VET_DUMP=$T/runs/v_${TAG}_$S.json \
  $HOME/anaconda3/bin/python -I finder.py --scenario $T/scen/$S --out $T/runs/o_${TAG}_$S.json --budget-s 900 > $T/runs/l_${TAG}_$S.log 2>&1
echo done $TAG $S
