#!/bin/bash

RENDER_DIR='../imgs/'
for file in ${RENDER_DIR}*.ps
do 
  ps2png $file ${file}.png
done

