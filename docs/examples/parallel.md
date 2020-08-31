# Parallel execution

This example illustrates how data can be exchanged and synchronised between processes. The code for this examples can be found [here](https://github.com/virgesmith/neworder/tree/master/examples/parallel).

The basic idea is that we have a population with an arbitrary state property that changes randomly, and each process controls the population with one particular state.

Each process starts with a population with a specific state, which in this example is the rank of the process. The population is stored in a pandas `Dataframe`. As the model evolves, indiviuals' state changes randomly, and individuals are exchanged between processes so that each process holds all the individuals with the corresponding state.

To run the model,

```bash
mpiexec -n 2 python examples/parallel/model.py
```

adjusting the path as necessary

## Input

Firstly we import the necessary modules switch on verbose mode, and check we are running in parallel mode:

```python
import neworder
from parallel import Parallel

neworder.verbose()

# must be MPI enabled
assert neworder.mpi.size() > 1, "This configuration requires MPI with >1 process"
```

As always, the neworder framework expects an instance of a Model class, subclassed from `neworder.Model`, which in turn requires a `neworder.Timeline` object. 

```python

population_size = 100
p_trans = 0.01
timeline = neworder.Timeline(0, 10, [10])

model = Parallel(timeline, p_trans, population_size)
```

Now each process has a population of 100 individuals, each of which has a probability of changing state of 1% at each of the ten (unit) timesteps.

