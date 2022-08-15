#!/bin/bash

RENDER_DIR='../imgs'
for file in ${RENDER_DIR}/*/*.ps
do 
  echo "Processing file ${file}..."
  ps2png $file ${file}.png 2> /dev/null
done

