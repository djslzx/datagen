#!/bin/bash 
for grammar in ../imgs/*grammar.txt
do 
  echo "grammar $grammar"
  cat $grammar
  echo
done | less
