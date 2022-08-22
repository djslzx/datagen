#!/bin/bash 
for grammar in ../imgs/*.grammar
do 
  echo "grammar $grammar"
  cat $grammar
  echo
done | less
