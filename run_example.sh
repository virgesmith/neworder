#!/bin/bash

if [ "$#" != "1" ]; then
  echo "usage: $0 example-dir"
  exit 1
fi

LD_LIBRARY_PATH=src/lib:$LD_LIBRARY_PATH src/bin/neworder examples/$1
