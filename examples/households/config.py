
""" config.py
Microsimulation config
"""
import os
import numpy as np
import neworder

# THIS could be very, very useful
#https://stackoverflow.com/questions/47297585/building-a-transition-matrix-using-words-in-python-numpy

# define the outer sequence loop (optional)
# run 4 sims
#neworder.sequence = np.array([3,1,2,0])
# define the evolution
neworder.timeline = (2011, 2012, [1])

areas = os.getenv("LADS").split(" ")

# define where the starting populations come from
data_dir = "examples/households/data"
# 
file_pattern = "hh_%s_OA11_%d.csv"

# running/debug options
neworder.log_level = 1
# this model isnt meant for parallel execution
assert neworder.size() == 1, "This example is configured to be run as a single process only"

# initialisation
neworder.initialisations = {
  "households": { "module": "households", "class_": "Households", "args": (data_dir, file_pattern, areas) }
}

# timestep must be defined in neworder
neworder.transitions = { 
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