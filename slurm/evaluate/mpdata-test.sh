#!/bin/bash
project_dir=/home/djl328/prob-repl
data_dir=$project_dir/datasets/wiz
script=$project_dir/slurm/data.sub
n=1000
i=0001
name=NSCA
sbatch --requeue $script \
       --dataset $data_dir/pre-eval-$n/$name/chunk-$i.jsonl \
       --out $data_dir/post-eval-$n/$name/chunk-$i.jsonl
