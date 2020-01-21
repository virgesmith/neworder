import numpy as np
import neworder

population_size = 100
p_trans = 0.01

# must be MPI enabled
assert neworder.size() > 1

neworder.timeline = (0, 100, 100)

# running/debug options
neworder.log_level = 1
neworder.do_checks = False
 
# initialisation
neworder.initialisations = {
  "test": { "module": "test", "class_": "Test", "args": (p_trans, population_size) }
}

neworder.transitions = {
  "test": "test.test()",
  #"redist": "TODO..." 
}

# Finalisation 
neworder.checkpoints = {
  "stats": "test.stats()",
}
