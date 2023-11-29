#!/bin/bash
prefix="/home/djl328/prob-repl"
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export 'TOKENIZERS_PARALLELISM=false'
source $prefix/venv/bin/activate
cd $prefix/src
python3 -u finetune/tune.py $@

