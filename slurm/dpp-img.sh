#!/bin/bash
PROJECT_DIR="/home/djl328/prob-repl"
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export 'TOKENIZERS_PARALLELISM=false'
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src
source $PROJECT_DIR/venv/bin/activate
cd $PROJECT_DIR/src
DATA_DIR="$PROJECT_DIR/out/dpp/2024-03-01_09-20-06/N=100,fit=all,accept=energy,steps=10000,spread=1,run=0/"

mkdir -p $DATA_DIR/batched-imgs

python \
  -u dpp.py \
  --mode npy-to-images \
  --domain lsystem
  --npy-dir $DATA_DIR/data \
  --img-dir $DATA_DIR/batched-imgs \
  --batched
