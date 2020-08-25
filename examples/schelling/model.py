import numpy as np
import neworder
from schelling import Schelling

# serial mode
neworder.verbose()

# category 0 is empty
gridsize = [100,125]
categories = np.array([0.56, 0.19, 0.19, 0.6])
# normalise
categories = categories / sum(categories)
similarity = 0.5

timeline = neworder.Timeline(0, 500, [5000])

schelling = Schelling(timeline, gridsize, categories, similarity)

neworder.run(schelling)
