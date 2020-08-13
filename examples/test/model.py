import neworder
from test import Test

# must be MPI enabled
assert neworder.mpi.size() > 1, "This configuration requires MPI"

population_size = 100
p_trans = 0.01
timeline = neworder.Timeline(0, 100, [100])

test_model = Test(timeline, p_trans, population_size)

neworder.run(test_model)