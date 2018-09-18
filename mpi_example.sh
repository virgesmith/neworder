#!/bin/bash

if [ "$#" != "2" ]; then
  echo "usage: $0 example-name size"
  exit 1
fi

# for ARC3 non-conda env
#PYTHONPATH=~/.local/lib/python3.6/site-packages:$PYTHONPATH

LD_LIBRARY_PATH=src/lib:$LD_LIBRARY_PATH mpirun -n $2 src/bin/neworder_mpi examples/$1 examples/shared
