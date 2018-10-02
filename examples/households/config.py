
""" config.py
Microsimulation config
"""
import numpy as np
import neworder

# THIS could be very, very useful
#https://stackoverflow.com/questions/47297585/building-a-transition-matrix-using-words-in-python-numpy

# define some global variables describing where the starting population and the parameters of the dynamics come from
initial_population = "examples/households/data/ssm_hh_E08000021_OA11_2011.csv"

# running/debug options
neworder.log_level = 1
# this model isnt meant for parallel execution
assert neworder.size() == 1, "This example is configured to be run as a single process only"

# define the outer sequence loop (optional)
# run 4 sims
#neworder.sequence = np.array([3,1,2,0])
# define the evolution
neworder.timeline = (2011.25, 2012.25, 1)

# initialisation
neworder.initialisations = {
  "households": { "module": "households", "class_": "Households", "parameters": [initial_population] }
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