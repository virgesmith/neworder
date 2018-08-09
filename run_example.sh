#!/bin/bash

if [ "$#" != "1" ]; then
  echo "usage: $0 example-dir"
  exit 1
fi

# for ARC3 non-conda env
PYTHONPATH=~/.local/lib/python3.6/site-packages:$PYTHONPATH

LD_LIBRARY_PATH=src/lib:$LD_LIBRARY_PATH src/bin/neworder examples/$1 examples/shared
