
#$ -cwd -V 
# Request runtime 
#$ -l h_rt=1:00:00 
# Request CPU cores 
#$ -pe ib 48 

# Martin tip:
##$ -l nodes=2 
# gives me 2 CPUs = 48 cores, & all their memory

#$ -m e
#$ -M a.p.smith@leeds.ac.uk

# for ARC3 non-conda env
export PYTHONPATH=~/.local/lib/python3.6/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=src/lib:$LD_LIBRARY_PATH 

#echo Start: $(date) > mpi_out.log

mpirun src/bin/neworder_mpi examples/people_big examples/shared >> mpi_out.log 2> mpi_err.log

#echo Finish: $(date) >> mpi_out.log

