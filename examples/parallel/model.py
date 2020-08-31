import neworder
from parallel import Parallel

neworder.verbose()

# must be MPI enabled
assert neworder.mpi.size() > 1, "This configuration requires MPI with >1 process"

population_size = 100
p_trans = 0.01
timeline = neworder.Timeline(0, 10, [10])

model = Parallel(timeline, p_trans, population_size)

neworder.run(model)