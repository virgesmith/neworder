# Tips and Tricks

## Model Initialisation

!!! danger "Memory Corruption"
    When instantiating the model class, it is imperative that the base class is explicitly initialised. Python does not enforce this, and memory corruption can occur.

Use this initialisation pattern:

```python
class MyModel(neworder.Model):
  def __init__(self, timeline, seeder, args...):
    # this line is essential
    super().__init__(timeline, seeder)
    # now initialise the subclass...
```

## Custom Seeding Strategies

!!! note "Note"
    `neworder` random streams use the Mersenne Twister generator, as implemented in the C++ standard library. 

`neworder` provides three seeding strategy functions which initialise the model's random stream so that they are either identical, independent, or non-reproducible (and independent).

If necessary, you can supply your own seeding strategy, for instance if you required some processes to have independent streams, and some identical streams. 

!!! note ""
    The seeder function must accept an `int` (even if unused) and return an `int`

```
def hybrid_seeder(rank):
  return (rank % 2) + 12345
```

or, as a lambda:

```
hybrid_seeder = lambda r: (r % 2) + 12345
```

which returns the same seed for all odd-ranked processes and a different seed for the even-ranked ones. The use your seeder when you instantiate the `Model`, e.g.

```python
class MyModel(neworder.Model):
  def __init__(self, timeline, args...): 
    super().__init__(timeline, lambda r: (r % 2) + 12345)
    ...
```

## Identical Streams

!!! warning "Synchronisation"
    Identically initialised random streams only stay in sync if the same number of samples are taken from each one .

The "option" example relies on identical streams to reduce noise when computing differences. It implements a `check` step that compares a single sample from each process and fails if any are different (see below).

!!! note "Stream Comparisons"
    Comparing output of each generator only gives a (very) probable assessment. The only surefire way to determine identical streams is to compare the internal states of each generator. This functionality is not currently implemented.

## Deadlocks

!!! danger "Failure is All-Or-Nothing"
    If checks fail, or any other error occurs in a parallel run, other processes must be notified, otherwise deadlocks can occur. 

Blocking communications between processes will deadlock if, for instance, the receiving process has ended due to an error. This will cause the entire run to hang (and may impact your HPC bill). The option example, as described above, has a check for random stream synchronisation that looks like this:

```python
  def check(self):
    # check the rng streams are still in sync by sampling from each one, comparing, and broadcasting the result
    # if one process fails the check and exits without notifying the others, deadlocks can result
    r = self.mc().ustream(1)[0]
    # send the value to process 0)
    a = comm.gather(r, 0)
    # process 0 checks the values
    if neworder.mpi.rank() == 0:
      ok = all(e == a[0] for e in a)
      neworder.log("check() ok: %s" % ok)
    else:
      ok = True
    # broadcast process 0's ok to all processes
    ok = comm.bcast(ok, root=0)
    return ok
```

The key here is that there is only one result, shared between all processes. In this case only one process is performing the check and broadcasting the result to the others. 

!!! note "Tip"
    In general, the return value of `check()` should be the logical "and" of the results from each process.