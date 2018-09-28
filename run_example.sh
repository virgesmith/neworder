#!/bin/bash

if [ "$#" != "1" ] && [ "$#" != "2" ] && [ "$#" != "3" ]; then
  echo "usage: $0 example-dir [size [-c]]"
  echo where size is the number of processes for a parallel run, which will be independently seeded unless the -c flag is set
  exit 1
fi

# serial mode
if [ "$#" != "1" ]; then
  mpi="mpirun -n $2"
  suffix=_mpi
  indep=1
  if [ "$3" == "-c" ]; then
    indep=0
  fi
fi
# for ARC3 non-conda env
#PYTHONPATH=~/.local/lib/python3.6/site-packages:$PYTHONPATH

LD_LIBRARY_PATH=src/lib:$LD_LIBRARY_PATH $mpi src/bin/neworder$suffix $indep examples/$1 examples/shared
