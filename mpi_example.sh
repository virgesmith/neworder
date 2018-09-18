#!/bin/bash

if [ "$#" != "2" ] && [ "$#" != "3" ]; then
  echo "usage: $0 example-name size [-c]"
  echo where size is the number of processes, which will be independently seeded unless the -c flag is set
  exit 1
fi

indep=1
if [ "$3" == "-c" ]; then
  indep=0
fi

# for ARC3 non-conda env
#PYTHONPATH=~/.local/lib/python3.6/site-packages:$PYTHONPATH

LD_LIBRARY_PATH=src/lib:$LD_LIBRARY_PATH mpirun -n $2 src/bin/neworder_mpi $indep examples/$1 examples/shared
