import neworder
from mpi4py import MPI
from parallel import Parallel

comm = MPI.COMM_WORLD

neworder.verbose()
#neworder.checked(False)

# must be MPI enabled
assert neworder.mpi.size() > 1, "This configuration requires MPI with >1 process"

population_size = 100
p_trans = 0.01
timeline = neworder.Timeline(0, 10, [10])

model = Parallel(timeline, p_trans, population_size)

neworder.run(model)

# now process 0 assembles all the data (this could just as easily be done inside the model)
pops = comm.gather(model.pop, root=0)
if neworder.mpi.rank() == 0:
  for r in range(1, neworder.mpi.size()):
    pops[0] = pops[0].append(pops[r])
  neworder.log("State counts (total %d):" % len(pops[0]))
  neworder.log(pops[0]["state"].value_counts())

