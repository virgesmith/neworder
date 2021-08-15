import numpy as np
import neworder
from schelling import Schelling

#neworder.verbose()

# category 0 is empty
gridsize = [640,480]
categories = np.array([0.56, 0.19, 0.19, 0.6])
# normalise
categories = categories / sum(categories)
similarity = 0.5

# open-ended timeline with arbitrary timestep
# the model halts when all agents are satisfied, rather than at a specific time
timeline = neworder.LinearTimeline(0, 1.0)

schelling = Schelling(timeline, gridsize, categories, similarity)

neworder.run(schelling)
