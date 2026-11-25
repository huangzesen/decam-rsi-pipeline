#!/bin/bash
T=$RSI_SANDBOX/r1/gen134/tmp; D=$DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev
cd $T/tools
while [ $# -ge 3 ]; do
  INJ_COLOR=1 INJ_SNR_SCALE=1.43 INJ_NAB=12 $HOME/anaconda3/bin/python inject_gen105.py $D/$1 $T/scen/$3 --n 40 --seed $2 --procs 2 > $T/scen/$3.log 2>&1 &
  shift 3
done
wait
