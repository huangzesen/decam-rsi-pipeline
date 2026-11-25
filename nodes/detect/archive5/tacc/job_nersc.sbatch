#!/bin/bash
# archive5 (NERSC Perlmutter CPU copy of job.sbatch, 2026-11-13): one CPU node downloads its regions' DECam InstCal files to scratch (fetch.py, md5-checked) and runs
# single-exposure detection on each exposure as soon as its three files are complete (detect.py --watch).
# Images stay on scratch (TACC purges files not accessed for 10 days). Detection lists: $D/det/<YYMMDD>/<stem>.npz.
# Submit from $D with -A, -p, -t and --export=ALL,PY=<python>,MAN=<manifest>,PART=k/N,STREAMS=64,WORKERS=48,HOURS=<h>
# (several nodes share one manifest with PART=k/N, each in its own $D so their detection does not overlap)
# Stop: touch $D/log/STOP (fetch stops starting files; detection finishes what is on disk).
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -n 1
#SBATCH -J archive5
#SBATCH -o log/job-%j.out
D=$SLURM_SUBMIT_DIR
cd $D || exit 1
mkdir -p log det img
rm -f log/FETCH_DONE
[ -n "$MODS" ] && { module purge >/dev/null 2>&1; module load $MODS; }
$PY -c "import numpy, scipy, astropy" || { echo "python $PY cannot import numpy/scipy/astropy"; exit 2; }
echo "node $(hostname) $(date); manifest $MAN; $STREAMS streams; $WORKERS detect workers; python $PY"
# the archive answers slowly under load (2026-11-13/14: jobs died on one 30 s check), so retry for ~10 min first
ok=0; for t in 1 2 3 4 5 6 7 8 9 10; do
  c=$(curl -s -o /dev/null -m 30 -w "%{http_code}" https://astroarchive.noirlab.edu/); echo "archive from this node: http $c (try $t)"
  [ "$c" = "200" ] && { ok=1; break; }; sleep 30
done
[ $ok = 1 ] || { echo "this node cannot reach the archive"; exit 2; }
$PY -I detect.py $D/img $D/det --workers $WORKERS --watch > log/detect_${SLURM_JOB_ID}.out 2>&1 &
DET=$!
$PY -I fetch.py $MAN $D --part ${PART:-0/1} --streams $STREAMS --hours $HOURS > log/fetch_${SLURM_JOB_ID}.log 2>&1
echo "fetch ended $(date) (exit $?)"
touch log/FETCH_DONE
wait $DET
echo "detect ended $(date); $(ls det/*/*.npz 2>/dev/null | wc -l) detection lists"
