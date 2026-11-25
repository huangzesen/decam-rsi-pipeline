#!/bin/zsh
# runarm.sh ARMDIR SCEN(R|Q) TAG [env...]
T=$RSI_SANDBOX/r1/gen147/tmp
A=$1; S=$2; G=$3; shift 3
cd $T/$A
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 TNO_FSTK_DMIN=40 TNO_FSTK_DMAX=44 TNO_FSTK_F=0.5 \
 TNO_FSTK_DUMP=$T/runs/fdump_$G.json TNO_VET_DUMP=$T/runs/vdump_$G.json "$@" $HOME/anaconda3/bin/python -I finder.py \
 --scenario $T/scen_inj$S --out $T/runs/out_$G.json --budget-s 900 > $T/runs/log_$G.txt 2>&1
echo done > $T/runs/done_$G
