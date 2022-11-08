#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tenv
cd /home/djl328/prob-repl/src/
make evo
