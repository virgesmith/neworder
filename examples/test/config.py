import numpy as np
import neworder

population_size = 100
p_trans = 0.01

# must be MPI enabled
assert neworder.size() > 1

timeline = neworder.Timeline(0, 100, [100])

# running/debug options
#neworder.log_level = 1
 
from test import Test
# initialisation
initialisations = {
  "test": Test(p_trans, population_size)
}

transitions = {
  "test": "test.test()",
}

# Finalisation 
checkpoints = {
  "stats": "test.stats()",
}

neworder.model = neworder.Model(timeline, [], initialisations, transitions, {}, checkpoints)

