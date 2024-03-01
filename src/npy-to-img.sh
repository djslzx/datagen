#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

dir="$1"

convert_npy_to_images() {
    local dir="$1"

    mkdir -p "$dir/imgs/"
    python dpp.py \
           --mode npy-to-images \
           --npy-dir "$dir/data/" \
           --img-dir "$dir/imgs/"
}

convert_npy_to_images "$dir"

