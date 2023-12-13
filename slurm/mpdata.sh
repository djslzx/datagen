#!/bin/bash
project_dir=/home/djl328/prob-repl
script=$project_dir/slurm/data.sub

for name in NSCA NSE WW WD CA
do
    for n in 00 01 02
    do
	filename=$name/chunk-$n.jsonl
	echo "Processing file ${filename}"
	sbatch --requeue $script $filename
    done
done
