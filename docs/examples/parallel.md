# Parallel Execution

This example illustrates how data can be exchanged and synchronised between processes. It uses the `mpi4py` package for interprocess communication. If you're unfamiliar with this package, or with MPI, check out the documentation [here](https://mpi4py.readthedocs.io/en/stable/).

The basic idea is that we have a population with a single arbitrary state property which can take one of `N` values, where `N` is the number of processes, and each process initially holds the part of the population in the corresponding state. As time evolves, indvidual's states change at random, and the processes exchange individuals to keep their own population homegeneous.

Each population is stored in a pandas `Dataframe`. At the start these is an equal population in each process (and thus in each state).

The states transition randomly with a fixed probability `p_trans`.

Finally, one process acquires the entire population and prints a summary of the state counts.

## Setup

Firstly we import the necessary modules and check we are running in parallel mode:

{{ include_snippet("examples/parallel/model.py", "setup") }}

`neworder` caches the MPI rank and size, which are assumed to be constant, for efficiency, and the functions `neworder.mpi.rank()` and `neworder.mpi.size()` can be used to inspect these values. As always, the neworder framework expects an instance of a Model class, subclassed from `neworder.Model`, which in turn requires a `neworder.Timeline` object:

{{ include_snippet("examples/parallel/model.py", "run") }}

Each process has an initial population of 100 individuals, each of which has a probability of changing state of 1% at each of the ten (unit) timesteps.

## The Model

Here's the model constructor:

{{ include_snippet("examples/parallel/parallel.py", "constructor") }}

The `step` method performs the state transitions at each timestep:

{{ include_snippet("examples/parallel/parallel.py", "step") }}

Note that `neworder.df.transition` modifies the dataframe in-place. The `check` method accounts for everyone:

{{ include_snippet("examples/parallel/parallel.py", "check") }}

Finally, the (single) checkpoint aggregates the populations and prints a summary of the populations in each state.

{{ include_snippet("examples/parallel/parallel.py", "checkpoint") }}

## Execution

As usual, to run the model we just execute the model script, but via MPI, e.g. from the command line, something like

```bash
mpiexec -n 8 python examples/parallel/model.py
```

adjusting the path as necessary, and optionally changing the number of processes.

## Output

Results will vary as the random streams are not deterministic in this example, but you should see something like:

```text
...
[py 0/8]  sending 2 emigrants to 7
[py 0/8]  received 2 immigrants from 1
[py 0/8]  received 1 immigrants from 4
[py 0/8]  received 1 immigrants from 5
[py 0/8]  received 1 immigrants from 6
[py 0/8]  received 2 immigrants from 7
[py 0/8]  State counts (total 800):
2    109
4    106
6    105
7     99
1     99
0     99
5     96
3     87
```

## [Examples Source Code](https://github.com/virgesmith/neworder/tree/master/examples/parallel)
