#!/bin/bash 
# Use borne shell  
#$ -m e
#$ -M a.p.smith@leeds.ac.uk
##$ -l h_vmem=2G
# Export variables and use current working directory 
#$ -cwd -V 
# Request 1 hour of runtime 
#$ -l h_rt=1:00:00 
# Request 24 CPU cores (seems to be the max allowed) 
#$ -pe ib 24 

# for ARC3 non-conda env
PYTHONPATH=~/.local/lib/python3.6/site-packages:$PYTHONPATH

echo Start: $(date) > mpi_out.log

LD_LIBRARY_PATH=src/lib:$LD_LIBRARY_PATH mpirun src/bin/neworder_mpi examples/people_big examples/shared >> mpi_out.log 2> mpi_err.log

echo Finish: $(date) >> mpi_out.log
