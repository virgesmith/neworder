import neworder
from test import Test
from mpi4py import MPI

comm = MPI.COMM_WORLD

neworder.module_init(comm.Get_rank(), comm.Get_size(), False, True)

# must be MPI enabled
assert neworder.mpi.size() > 1, "This configuration requires MPI with >1 process"

population_size = 100
p_trans = 0.01
timeline = neworder.Timeline(0, 100, [100])

test_model = Test(timeline, p_trans, population_size)

neworder.run(test_model)