# Parallel execution

This example illustrates how data can be exchanged and synchronised between processes. The code for this examples can be found [here](https://github.com/virgesmith/neworder/tree/master/examples/parallel).

The basic idea is that we have a population with a single arbitrary state property which can take one of `N` values, and each process holds the part of the population in one of the possible states.

Each population is stored in a pandas `Dataframe`. At the start these is an equal population in each state (and thus for each process).

The state changes randomly to another states with a fixed probability `p_trans`. At each timestep.

Each process starts with a population in a specific state, which in this example just corresponds to the rank of the process.  As the model evolves and individuals' state changes they are exchanged between processes so that each process holds all the individuals with the corresponding state.

Finally, one process acquires the entire population and prints a summary of the state counts.

To run the model,

```bash
mpiexec -n 2 python examples/parallel/model.py
```

adjusting the path as necessary, and optionally increasing the number of processes from 2.

## Input

Firstly we import the necessary modules switch on verbose mode, and check we are running in parallel mode:

```python
import neworder
from parallel import Parallel

# turn this on for (lots) more output
#neworder.verbose()

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

Here's the model constructor:

```python
class Parallel(neworder.Model):
  def __init__(self, timeline, p, n):
    # initialise base model (essential!)
    super().__init__(timeline, neworder.MonteCarlo.deterministic_independent_seed)

    # enumerate possible states
    self.s = np.array(range(neworder.mpi.size()))

    # create transition matrix
    self.p = np.identity(neworder.mpi.size()) * (1 - neworder.mpi.size() * p) + p

    # record initial population size
    self.n = n

    # individuals use the index as a unique id and their initial state is the MPI rank
    self.pop = pd.DataFrame({"id": np.array(range(neworder.mpi.rank() * n, (neworder.mpi.rank() + 1) * n)),
                             "state": np.full(n, neworder.mpi.rank()) }).set_index("id")
```

and the step method models the transitions:

```python
  def step(self):
    # generate some movement
    neworder.df.transition(self, self.s, self.p, self.pop, "state")

    # send emigrants to other processes
    for s in range(neworder.mpi.size()):
      if s != neworder.mpi.rank():
        emigrants = self.pop[self.pop.state == s]
        neworder.log("sending %d emigrants to %d" % (len(emigrants), s))
        comm.send(emigrants, dest=s)

    # remove the emigrants
    self.pop = self.pop[self.pop.state == neworder.mpi.rank()]

    # receive immigrants
    for s in range(neworder.mpi.size()):
      if s != neworder.mpi.rank():
        immigrants = comm.recv(source=s)
        if len(immigrants):
          neworder.log("received %d immigrants from %d" % (len(immigrants), s))
          self.pop = self.pop.append(immigrants)
```


```python
  def check(self):
    """ Ensure we havent lost (or gained) anybody """
    totals = comm.gather(len(self.pop), root=0)
    if neworder.mpi.rank() == 0:
      return sum(totals) == self.n * neworder.mpi.size()
    return True
```