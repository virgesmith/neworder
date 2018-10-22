import numpy as np
import neworder

# must be MPI enabled
assert neworder.size() == 1

# category 0 is empty
gridsize = [40,50]
categories = [0.3, 0.3, 0.3, 0.1]
similarity = 0.4

neworder.timeline = (0, 100, 100)

# running/debug options
neworder.log_level = 1
neworder.do_checks = False
 
# initialisation
neworder.initialisations = {
  "model": { "module": "model", "class_": "Schelling", "parameters": [gridsize, categories, similarity] }
}

neworder.transitions = {
  "step": "model.step()",
  #"redist": "TODO..." 
}

# Finalisation 
neworder.checkpoints = {
  "stats": "model.stats()",
}
