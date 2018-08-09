#!/bin/sh 
# Use borne shell  
# Export variables and use current working directory 
#$ -cwd -V 
# Request 1 hour of runtime 
#$ -l h_rt=1:00:00 
# Request 80 CPU cores (processes) 
#$ -pe ib 80 

# for ARC3 non-conda env
PYTHONPATH=~/.local/lib/python3.6/site-packages:$PYTHONPATH

LD_LIBRARY_PATH=src/lib:$LD_LIBRARY_PATH mpirun src/bin/neworder_mpi examples/people_big examples/shared
