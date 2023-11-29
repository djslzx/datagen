#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate tenv
prefix="/home/djl328/prob-repl"
cd $prefix/src
PYTHONUNBUFFERED=1;TOKENIZERS_PARALLELISM=false;OPENAI_API_KEY=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 python3 -u solve.py "$1" "$2"
