#!/bin/bash
PROJECT_DIR="/home/djl328/prob-repl"
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export 'TOKENIZERS_PARALLELISM=false'
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src
source $PROJECT_DIR/venv/bin/activate
cd $PROJECT_DIR/src
python -u dpp.py --mode=search
