#!/bin/bash
project_dir=/home/djl328/prob-repl
data_dir=$project_dir/datasets/wiz
script=$project_dir/slurm/data.sub
n=10

for name in NSCA NSE WW WD CA
do
    in_dir=$data_dir/pre-eval-$n/$name
    out_dir=$data_dir/post-eval-$n/$name
    mkdir -p $out_dir

    for i in {0000..0009}
    do
	echo "Processing file $in_dir/chunk-$i.jsonl"
	sbatch --requeue $script \
	       --dataset $in_dir/chunk-$i.jsonl \
	       --out $out_dir/chunk-$i.jsonl
    done
done
