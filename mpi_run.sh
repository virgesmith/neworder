#!/bin/bash

# for ARC3 non-conda env
PYTHONPATH=~/.local/lib/python3.6/site-packages:$PYTHONPATH

LD_LIBRARY_PATH=src/lib:$LD_LIBRARY_PATH mpirun -n 8 src/bin/neworder_mpi examples/people_big examples/shared
