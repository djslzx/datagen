#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tenv
prefix="/home/djl328/prob-repl"
cd $prefix/src
python3 -u evaluate.py

