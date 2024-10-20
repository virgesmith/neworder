# Tips and Tricks

## Model Initialisation

!!! warning "Base Model Initialisation"
    When instantiating the model subclass, it is essential that the `neworder.Model` base class is explicitly initialised. It must be supplied with a `Timeline` object and (optionally) a seeding function for the Monte-Carlo engine. Failure to do this will result in a runtime error.

Ensure the `neworder.Model` base class is properly initialised:

```python title="Model initialisation"
class MyModel(neworder.Model):
  def __init__(self, args...) -> None:
    timeline = ... # initialise an appropriate timeline
    super().__init__(timeline) # (1)!
    ... # now initialise the subclass
```

1.  :material-alert: this line is essential

## Custom Seeding Strategies

!!! note "Random number generator"
    *neworder* random streams use the Mersenne Twister pseudorandom generator, as implemented in the C++ standard library.

*neworder* provides three basic seeding functions which initialise the model's random stream so that they are either non-reproducible (`neworder.MonteCarlo.nondeterministic_stream`), or reproducible and either identical (`neworder.MonteCarlo.deterministic_identical_stream`) or independent across parallel runs (`neworder.MonteCarlo.deterministic_independent_stream`). Typically, a user would select identical streams (and perturbed inputs) for sensitivity analysis, and independent streams (with identical inputs) for convergence analysis.

If necessary, you can supply your own seeding strategy, for instance if you required half the processes to have identical streams:

!!! note "Seeder function signature"
    The seeder function must take no arguments and return an `int`. When the function is called by the neworder runtime, the "rank" (in MPI parlance) of each process is available to it. For serial execution, the rank will always be zero.

```python
import neworder
def hybrid_seeder() -> int:
  return (neworder.mpi.RANK % 2) + 12345
```

or, as a lambda:

```python
hybrid_seeder: Callable[[], int] = lambda r: (r % 2) + 12345
```

which returns the same seed for all odd-ranked processes and a different seed for the even-ranked ones. You can define your seeder inline when you instantiate the `Model`, e.g.

```python
class MyModel(neworder.Model):
  def __init__(self, timeline: neworder.Timeline) -> None:
    super().__init__(timeline, lambda: (neworder.mpi.RANK % 2) + 12345)
    ...
```

If there was a requirement for multiple processes to all have the same nondeterministic stream, you could implement a seeding strategy like so:

```python
def nondeterministic_identical_stream() -> int:
  # only process 0 gets a seed
  seed = neworder.MonteCarlo.nondeterministic_stream(0) if neworder.mpi.RANK == 0 else None
  # then broadcasts it to the other processes
  seed = neworder.mpi.COMM.bcast(seed, root=0)
  return seed

```

## Identical Streams

!!! warning "Synchronisation"
    Identically initialised random streams only stay in sync if the same number of samples are taken from each one .

The "option" example relies on parallel processes with identical random streams to reduce noise when computing differences for sensitivity analysis. It implements a `check` step that compares the internal states of the random stream in each process and fails if any are different (see the example code).

## External Sources of Randomness

Other libraries, such as *numpy*, contain a much broader selection of random number functionality than *neworder* does, and it makes no sense to reimplement such functionality. If you are using a specific seeding strategy within neworder, and are also using an external random generator, it is important to ensure they are also following the same strategy, otherwise reproducibility may be compromised.

In your model constructor, you can seed the *numpy* generator like so

```python
ext_seed = self.mc.raw()
self.nprand = np.random.Generator(np.random.MT19937(ext_seed))
# ...get some values
x = self.nprand.normal(size=5)
```

If you've chosen a deterministic seedng strategy, then `ext_seed` will be reproducible, and if you've chosen an independent strategy, then `ext_seed` will be different for each process, thus propagating your chosen seeding strategy to the external generator.

!!! note "Seeding external generators"
    Wherever possible, explicitly seed any external random generators using *neworder*'s MonteCarlo engine. This will effectively propagate your seeding strategy to the external generator.

### Using neworder's random generator with numpy

It is now possible to use the RNG from the neworder model's Monte-Carlo engine as a `numpy` generator. In this way all of numpy's functionality is available with neworder's RNG. To achieve this use the adapter function `as_np`. Similarly to the example above, in your model constructor create the numpy generator:

```py
self.nprand = no.as_np(self.mc)
# ...get some values
x = self.nprand.normal(size=5)
```

NB as there is only one RNG state, you can safely get independent variates when calling both the RNG directly and via numpy.

## Ending the model run

Models will continue to run until the end of their timeline is reached, unless explicitly told otherwise (see next section).

!!! Note "Finalisation"
    The model's `finalise` method can be optionally implemented as necessary, for example to write results to a file. It is automatically called by the *neworder* runtime **only** when the end of the timeline is reached.

## Open-ended timelines and Conditional Halting

In some models, rather than (or as well as) evolving the population over a fixed timeline, it may make more sense to iterate timesteps until some condition is met. The "Schelling" example illustrates this - it runs until all agents are in a satisfied state. Currently, the inbuilt `LinearTimeline` and `CalendarTimeline` classes support both fixed and open-ended timelines. In other cases it may be useful to temporarily exit the model for later resumption.

The model's `halt` method can be used to stop the model run. In these situations, the `step` method should have some logic to (conditionally) call the `halt` method.

!!! note "`Model.halt()`"
    This function *does not* end execution immediately, it signals to the *neworder* runtime not to iterate any further timesteps. Calling `halt` means that:

    - the entire body of the `step` method (and the `check` method, if implemented) will still run for the current timestep,
    - the `finalise` method, even if implemented, will **not** be excuted,
    - the `neworder.run` method will then exit.

Overriding the `halt` method should not be necessary and is not recommended. The `finalise` method, if needed, must be called explicitly for models with open-ended timelines that have been `halt`ed.

!!! Note "Resuming execution"
    A model that has previously been `halt`ed but has not reached the end of its timeline can be resumed by passing it to `neworder.run` again. Attempting to resume a model that has reached the end of it's timeline will result in a `StopIteration` exception.

## Deadlocks

!!! danger "Failure is All-Or-Nothing"
    If checks fail, or any other error occurs in one process in a parallel run, other processes must be notified, otherwise deadlocks can occur.

Blocking communications between processes will deadlock if, for instance, the receiving process has ended due to an error. This will cause the entire run to hang (and may impact your HPC bill). The option example, as described above, has a check for random stream synchronisation that looks like this:

{{ include_snippet("examples/option/black_scholes.py", "check") }}

The key here is that there is only one result, shared between all processes. In this case only one process is performing the check and broadcasting the result to the others.

!!! note "Tip"
    In general, the return value of `check()` should be the logical "and" of the results from each process.

## Time Comparison

*neworder* uses 64-bit floating-point numbers to represent time, and the values `-inf`, `+inf` and `nan` respectively to represent the concepts of the distant past, the far future and never. This allows users to define, or compare against, values that are:

- unequal to any time value, including itself (`neworder.time.NEVER`),
- before any other (non-never) time value (`neworder.time.DISTANT_PAST`) , or
- after any other (non-never) time value (`neworder.time.FAR_FUTURE`)

!!! warning "NaN comparisons"
    Due to the rules of [IEEE754 floating-point](https://en.wikipedia.org/wiki/NaN#Comparison_with_NaN), care must be taken when comparing to `NaN`/`NEVER`, since a direct comparison will always be false, i.e.: `NEVER != NEVER`.

To compare time values with "never", use the supplied function `isnever()`:

```python
import neworder
n = neworder.time.NEVER
neworder.log(n == n) # False!
neworder.log(neworder.time.isnever(n)) # True
```

## Data Types

!!! warning "Static typing"
    Unlike python, C++ is a *statically typed* language and so *neworder* is strict about types. We strongly encourage the use of type annotations and a type checker (e.g. mypy) in python.

If an argument to a *neworder* method or function is not the correct type, it will fail immediately (as opposed to python, which will fail only if an invalid operation for the given type is attempted (a.k.a. "duck typing")). This applies to contained types (numpy's `dtype`) too. In the example below, the function is expecting an integer, and will complain if you pass it a floating-point argument:

```python
>>> import neworder
>>> neworder.df.unique_index(3.0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unique_index(): incompatible function arguments. The following argument types are supported:
    1. (n: int) -> numpy.ndarray[int64]

Invoked with: 3.0
```

## Project Structure

Although obvious to many users, in order to promote reusability, it is recommended to separate out functionality into logical units, for example:

- model definition - the actual model implementation
- model data - loading and preprocessing of input data
- model execution - defining the parameters of the model and running it
- result postprocessing and visualisation

This makes life much easier when you want to:

- use the same model with different parameters and/or input data,
- run the model on different plaforms without modification (think desktop vs HPC cluster vs web service).
- have visualisations tailored to the platform you are working on.
- run multiple models from one script.

The examples use canned (i.e. already preprocessed) data but otherwise largely adhere to this pattern.
