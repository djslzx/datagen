#!/bin/bash
RENDER_DIR="$1"
if [[ -z "$RENDER_DIR" ]]; then
   echo "Usage: convert-ps.sh RENDER_DIR"
   exit 1
fi

# for file in "${RENDER_DIR}/*/*.ps"
# do 
#   echo "Processing file ${file}..."
#   ./scripts/ps2png.sh $file ${file}.png 2> /dev/null
# done
for file in "${RENDER_DIR}"/*.ps
do 
  echo "Processing file ${file}..."
  ./scripts/ps2png.sh $file ${file}.png 2> /dev/null
done

