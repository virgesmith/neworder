# !setup!
from parallel_mpi import ParallelMPI  # import our model definition  # ty:ignore[import-not-found]

import neworder

# neworder.verbose()
# neworder.checked(False)

# must be MPI enabled
assert neworder.mpi.SIZE > 1, "This configuration requires MPI with >1 process"
# !setup!

# !run!
population_size = 100
p = 0.01
timeline = neworder.timeline.linear(0, end=10)
model = ParallelMPI(timeline, p, population_size)
neworder.run(model)
#!run!
