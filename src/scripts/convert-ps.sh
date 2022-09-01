#!/bin/bash
DIR="$1"
if [[ -z "$DIR" ]]; then
   echo "Usage: convert-ps.sh DIR"
   exit 1
fi

# for file in "${DIR}/*/*.ps"
# do 
#   echo "Processing file ${file}..."
#   ./scripts/ps2png.sh $file ${file}.png 2> /dev/null
# done
if ls "${DIR}/*.ps" > /dev/null 2>&1; then
  for file in "${DIR}"/*.ps
   do 
     echo "Processing file ${file}..."
     ./scripts/ps2png.sh "$file" "${file}.png" 2> /dev/null
   done
else
  echo "No postscript files found"
fi
