import numpy as np
import neworder
from schelling import Schelling

#neworder.verbose()

# category 0 is empty
gridsize = [100,125]
categories = np.array([0.56, 0.19, 0.19, 0.6])
# normalise
categories = categories / sum(categories)
similarity = 0.5

# unit timeline up to max_steps
max_steps = 1000
timeline = neworder.LinearTimeline(0, max_steps, [max_steps])

schelling = Schelling(timeline, gridsize, categories, similarity)

neworder.run(schelling)
