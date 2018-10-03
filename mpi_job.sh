
#$ -cwd -V 
# Memory
#$ -l h_vmem=2G
# Request runtime 
#$ -l h_rt=2:00:00 
# Request CPU cores 
#$ -pe ib 48 

# Martin tip:
##$ -l nodes=2 
# gives me 2 CPUs = 48 cores, & all their memory

#$ -m be
#$ -M a.p.smith@leeds.ac.uk
#$ -o logs/
#$ -e logs/

# for ARC3 non-conda env
export PYTHONPATH=~/.local/lib/python3.6/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=src/lib:$LD_LIBRARY_PATH 

mpirun src/bin/neworder_mpi 48 1 examples/people_multi examples/shared

