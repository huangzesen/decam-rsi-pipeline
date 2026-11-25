#!/bin/bash
# search38: run one program over the 5 % search regions on one NERSC GPU node (4 x A100), four lanes, one A100 each, no
# network, BUDGET seconds per region (node38.sh: the hill climb's node script plus a wall-time guard). Several nodes may
# share one ROOT: node38.sh takes each (tag, region) once through lock directories.
# ROOT/list.txt holds "TAG SID" lines, ROOT/TAG/sub the program. Outputs ROOT/TAG/out/SID/candidates.json, logs, timing.
# Submit from $D/search38: sbatch --export=ALL,ROOT=<dir>,BUDGET=<s>,WALL_S=<s> -t <wall> run38.sbatch
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -A $NERSC_GPU_ACCOUNT
#SBATCH -J dbt-run38
#SBATCH -o log/run38-%j.out
S=$NERSC_SCRATCH/dbt/search38
echo "node $(hostname) $(date) root $ROOT budget $BUDGET wall_s $WALL_S"
srun -N 1 -n 1 bash $S/node38.sh $ROOT $BUDGET $WALL_S
echo "done $(date)"
