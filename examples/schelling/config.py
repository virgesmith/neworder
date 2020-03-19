import numpy as np
import neworder

# must be MPI enabled
assert neworder.size() == 1

# category 0 is empty
gridsize = [100,125]
categories = np.array([1.02, 0.19, 0.19, 0.6])
# normalise
categories = categories / sum(categories)
similarity = 0.5

neworder.timeline = neworder.Timeline(0, 500, [5000])

# running/debug options
neworder.log_level = 1
neworder.do_checks = False
 
# initialisation
neworder.initialisations = {
  "model": { "module": "model", "class_": "Schelling", "args": (gridsize, categories, similarity) }
}

neworder.transitions = {
  "step": "model.step()",
  #"redist": "TODO..." 
}

# Finalisation 
neworder.checkpoints = {
  "stats": "model.stats()",
#  "anim": "model.animate()"
}
