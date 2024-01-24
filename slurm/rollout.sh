#!/bin/bash
PROJECT_DIR=/home/djl328/prob-repl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src
source $PROJECT_DIR/venv/bin/activate
cd $PROJECT_DIR/src

for dataset in NSCA # NSE WW WD CA
do
    python3 -u finetune/main.py \
	    --mode rollout \
	    --dataset $PROJECT_DIR/datasets/wiz/hf-20\:30k/$dataset \
	    --model-name "codellama/CodeLLama-7b-Python-hf" \
	    --kbit 8 \
	    --max-seq-length 512
done
