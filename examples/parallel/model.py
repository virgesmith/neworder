# !setup!
import neworder
from parallel import Parallel # import our model definition

#neworder.verbose()
#neworder.checked(False)

# must be MPI enabled
assert neworder.mpi.size > 1, "This configuration requires MPI with >1 process"
# !setup!

# !run!
population_size = 100
p = 0.01
timeline = neworder.LinearTimeline(0, 10, 10)
model = Parallel(timeline, p, population_size)
neworder.run(model)
#!run!
