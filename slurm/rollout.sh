#!/bin/bash
PROJECT_DIR=/home/djl328/prob-repl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src
source $PROJECT_DIR/venv/bin/activate
cd $PROJECT_DIR/src

DATASETS="NSCA NSE WW WD CA"
MODEL_PATHS=$(grep -v '^#'  $PROJECT_DIR/slurm/models.txt)

for MODEL_PATH in $MODEL_PATHS
do
  echo "Loading model from ${MODEL_PATH}"
  for DATASET in $DATASETS
  do
    echo "Testing model on dataset ${DATASET}"
    # CUDA_LAUNCH_BLOCKING=1 
    python3 -u finetune/main.py \
            --mode rollout \
            --dataset $PROJECT_DIR/datasets/wiz/hf-20\:30k/$DATASET \
            --model-name $MODEL_PATH \
            --kbit 8 \
            --max-seq-length 512 \
            --peft \
            --out-dir $PROJECT_DIR/datasets/wiz/rollouts/
  done
done

