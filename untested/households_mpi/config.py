
""" config.py
Microsimulation config for MPI-only implementation of household microsimulation prototype
"""
import os
import glob
import numpy as np
import neworder

assert neworder.mpi.SIZE() > 1 and neworder.INDEP, "This example requires MPI with independent RNG streams"

# THIS could be very, very useful
#https://stackoverflow.com/questions/47297585/building-a-transition-matrix-using-words-in-python-numpy

# define the outer sequence loop (optional)
# run 4 sims
#neworder.sequence = np.array([3,1,2,0])
# define the evolution
neworder.timeline = neworder.Timeline(2011, 2019, [8])

# define where the starting populations come from
data_dir = "examples/households/data"
# TODO this should probably not be same as above
cache_dir = "examples/households/data"
file_pattern = "hh_*_OA11_2011.csv"

# MPI split initial population files over threads
def partition(arr, count):
  return [arr[i::count] for i in range(count)]

initial_populations = partition(glob.glob(os.path.join(data_dir, file_pattern)), neworder.mpi.SIZE())

# household type transition matrix
ht_trans = os.path.join(data_dir, "w_hhtype_dv-tpm.csv")

# running/debug options
neworder.log_level = 1

# initialisation
neworder.initialisations = {
  "households": { "module": "households", "class_": "Households", "args": (initial_populations[neworder.mpi.RANK()], ht_trans, cache_dir) }
}

# timestep must be defined in neworder
neworder.dataframe.transitions = {
  "age": "households.age(timestep)"
}

# checks to perform after each timestep. Assumed to return a boolean
neworder.do_checks = True # Faith
# assumed to be methods of class_ returning True if checks pass
neworder.checks = {
  "check": "households.check()"
}

# Generate output at each checkpoint
neworder.checkpoints = {
  "write_table" : "households.write_table()"
}