#!/bin/bash
ENV=.env
REQS=requirements.txt

rm -rf $ENV
python3 -m venv $ENV
source $ENV/bin/activate
# pip install -r $REQS
