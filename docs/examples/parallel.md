# Parallel Execution

This example illustrates how data can be exchanged and synchronised between processes. The code for this examples can be found [here](https://github.com/virgesmith/neworder/tree/master/examples/parallel).

The basic idea is that we have a population with a single arbitrary state property which can take one of `N` values, where `N` is the number of processes, and each process initially holds the part of the population in the corresponding state. As time evolves, indvidual's states change at random, and the processes exchange individuals to keep their own population homegeneous.

Each population is stored in a pandas `Dataframe`. At the start these is an equal population in each process (and thus in each state).

The states transition randomly with a fixed probability `p_trans`.

Finally, one process acquires the entire population and prints a summary of the state counts.

## Setup

Firstly we import the necessary modules and check we are running in parallel mode:

```python
import neworder
from parallel import Parallel

# turn this on for (lots) more output
#neworder.verbose()

# must be MPI enabled
assert neworder.mpi.size() > 1, "This configuration requires MPI with >1 process"
```

As always, the neworder framework expects an instance of a Model class, subclassed from `neworder.Model`, which in turn requires a `neworder.Timeline` object:

```python
population_size = 100
p_trans = 0.01
timeline = neworder.Timeline(0, 10, [10])

model = Parallel(timeline, p_trans, population_size)
```

Each process has an initial population of 100 individuals, each of which has a probability of changing state of 1% at each of the ten (unit) timesteps.

## The Model

Here's the model constructor:

```python
class Parallel(neworder.Model):
  def __init__(self, timeline, p, n):
    # initialise base model (essential!)
    super().__init__(timeline, neworder.MonteCarlo.nondeterministic_stream)

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

and the step method, which performs the state transitions:

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

And, just to be sure we account for everyone:

```python
  def check(self):
    # Ensure we haven't lost (or gained) anybody
    totals = comm.gather(len(self.pop), root=0)
    if neworder.mpi.rank() == 0:
      if sum(totals) != self.n * neworder.mpi.size():
        return False

    # And check we only have individuals that we should have
    return len(self.pop[self.pop.state != neworder.mpi.rank()]) == 0
```

Finally, the (single) checkpoint aggregates the populations 

```python
  def checkpoint(self):
    # wait until any slower-running processes catch up
    comm.Barrier()
    # then process 0 assembles all the data and prints a summary
    pops = comm.gather(self.pop, root=0)
    if neworder.mpi.rank() == 0:
      for r in range(1, neworder.mpi.size()):
        pops[0] = pops[0].append(pops[r])
      neworder.log("State counts (total %d):" % len(pops[0]))
      neworder.log(pops[0]["state"].value_counts())
```

## Execution

As usual, to run the model we just call

```python
neworder.run(model)
```

Or from the command line, something like

```bash
mpiexec -n 8 python examples/parallel/model.py
```

adjusting the path as necessary, and optionally changing the number of processes.

## Output

Results will vary as the random streams are not deterministic in this example, but you should see something like:

```text
...
[py 6/8] received 2 immigrants from 1
[py 6/8] received 1 immigrants from 2
[py 6/8] received 2 immigrants from 3
[py 6/8] received 1 immigrants from 4
[py 0/8] State counts (total 800):
[py 0/8] 6    118
2    118
5    106
0    101
7     92
4     89
3     88
1     88
Name: state, dtype: int64
```
