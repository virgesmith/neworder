#!setup!
from mpi4py import MPI
import neworder
from parallel import Parallel # import our model definition

comm = MPI.COMM_WORLD

#neworder.verbose()
#neworder.checked(False)

# must be MPI enabled
assert neworder.mpi.size() > 1, "This configuration requires MPI with >1 process"
#!setup!

#!run!
population_size = 100
p_trans = 0.01
timeline = neworder.Timeline(0, 10, [10])

model = Parallel(timeline, p_trans, population_size)

neworder.run(model)
#!run!

