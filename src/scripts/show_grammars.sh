#!/bin/bash 
DIR="$1"
for grammar in "${DIR}/"*.grammar
do 
  echo "grammar $grammar"
  cat $grammar
  echo
done | less
