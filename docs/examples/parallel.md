# Parallel Execution

This example illustrates how data can be exchanged and synchronised between processes. It uses the `mpi4py` package for interprocess communication. If you're unfamiliar with this package, or with MPI, check out the documentation [here](https://mpi4py.readthedocs.io/en/stable/).

The basic idea is that we have a population with a single arbitrary state property which can take one of `N` values, where `N` is the number of processes, and each process initially holds the part of the population in the corresponding state. As time evolves, indvidual's states change at random, and the processes exchange individuals to keep their own population homegeneous.

Each population is stored in a *pandas* `DataFrame`. At the start these is an equal population in each process (and thus in each state).

The states transition randomly with a fixed probability \(p\) at each timestep, and those that change are redistributed amongst the processes.

Finally, one process acquires the entire population and prints a summary of the state counts.

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

!!! note "Optional dependencies"
    This example requires optional dependencies, see [system requirements](../..#system-requirements) and use:

    `pip install neworder[parallel]`

## Setup

Firstly, we import the necessary modules and check we are running in parallel mode:

{{ include_snippet("examples/parallel/model.py", "setup") }}

!!! note "MPI"
    *neworder* uses the `mpi4py` package to provide MPI functionality, which in turn requires an MPI installation on the host (see [system requirements](../..#system-requirements)). The attributes `neworder.mpi.COMM` (the MPI communicator), `neworder.mpi.RANK` and `neworder.mpi.SIZE` are provided for convenience.

As always, the neworder framework expects an instance of a model class, subclassed from `neworder.Model`, which in turn requires a timeline, in this case a `neworder.LinearTimeline` object:

{{ include_snippet("examples/parallel/model.py", "run") }}

So each process has an initial population of 100 individuals, each of which has a 1% probability of changing to another given state at each of the ten (unit) timesteps.

## The Model

Here's the model constructor:

{{ include_snippet("examples/parallel/parallel.py", "constructor") }}

The `step` method, which is called at every timestep performs the state transitions. Note that `neworder.df.transition` modifies the dataframe in-place. Then, sends individuals with changed state to the appropriate process and receives appropriate individuals from the other processes:

{{ include_snippet("examples/parallel/parallel.py", "step") }}

!!! warning "Blocking communication"
    The above implementation uses *blocking communication*, which means that all processes send and receive from each other, even if they send an empty dataframe: a given process cannot know in advance if it's not going to receive data from another process, and will deadlock if it tries to receive data from a process that hasn't sent any. MPI does have non-blocking communication protocols, which are more complex to implement. For more info see the mpi4py [documentation](https://mpi4py.readthedocs.io/en/stable/overview.html?highlight=nonblocking#nonblocking-communications).

The `check` method accounts for everyone being present and in the right place (i.e. process):

{{ include_snippet("examples/parallel/parallel.py", "check") }}

For an explanation of why it's implemented like this, see [here](../tips.md#deadlocks). The `finalise` method aggregates the populations and prints a summary of the populations in each state.

{{ include_snippet("examples/parallel/parallel.py", "finalise") }}

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
