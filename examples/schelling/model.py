import numpy as np
import neworder
from schelling import Schelling

# must not be MPI enabled
assert neworder.mpi.size() == 1

# category 0 is empty
gridsize = [100,125]
categories = np.array([0.56, 0.19, 0.19, 0.6])
# normalise
categories = categories / sum(categories)
similarity = 0.5

timeline = neworder.Timeline(0, 500, [5000])

neworder.model = Schelling(timeline, gridsize, categories, similarity)
