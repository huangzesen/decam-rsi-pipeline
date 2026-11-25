#!/bin/sh
# mkshort.sh DEV_SCENARIO_DIR OUT_DIR 'SED_ROWS' : a short scenario from some rows of a dev scenario's exposures.csv
# (gen009). Example: mkshort.sh $RSI_DEV_DIR/scen04 tmp/fake/S1 '2,5p'   (rows 2-5 of the file = exposures 1-4)
mkdir -p "$2"; ln -sfn "$1/exposures" "$2/exposures"; cp "$1/meta.json" "$2/meta.json"
(head -1 "$1/exposures.csv"; sed -n "$3" "$1/exposures.csv") > "$2/exposures.csv"; wc -l "$2/exposures.csv"
