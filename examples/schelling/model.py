import numpy as np
import neworder
from schelling import Schelling

#neworder.verbose()

# category 0 is empty
gridsize = [480,360]
categories = np.array([0.36, 0.12, 0.12, 0.4])
# normalise if necessary
# categories = categories / sum(categories)
similarity = 0.6

# open-ended timeline with arbitrary timestep
# the model halts when all agents are satisfied, rather than at a specific time
timeline = neworder.LinearTimeline(0, 1.0)

schelling = Schelling(timeline, gridsize, categories, similarity)

neworder.run(schelling)
