# Reference

TODO update...

## Classes and Modules

The neworder python module defines the following symbols within the `neworder` namespace:

name                        | type        | description
----------------------------|-------------|--------------
[`Timeline`](#timeline)     | class       | defines the timeline over which the model is run
[`MonteCarlo`](#montecarlo) | class       | random number generation and sampling functionality
`Model`                     | class       | the base class from which all models should be derived from
`time`                      | module      | time-related functionality
`stats`                     | module      | statistical functions
`dataframe`                 | module      | direct, fast manipulation of pandas DataFrames
`mpi`                       | module      | inter-process communication
`log`                       | function    | prints output with annotations
`version`                   | function    | reports the version of the embedded environment
`python`                    | function    | reports the python version used in the embedded environment
`shell`                     | function    | invokes an interactive python shell for debugging (embedded mode only)


## Timeline

The `Timeline` class describes a timeline containing one of more checkpoints.

The final value in checkpoints must be a list, e.g. Timeline(2020, 2050, [10, 20, 30]) will start the simulation at 2020 and end at 2050 with a yearly timestep and checkpoints at 2030, 2040 and 2050.

To generate a 'null' timeline, which is useful for case-based models use

```
timeline = Timeline.null()
```

which is equivalent to `Timeline(0, 0, [1])`.

The timeline is incremented internally by the neworder Model object and is not modifiable in python code once instantiated. The following accessors are provided:

name                | description
--------------------|------------------------------------
`null()`            | construct null timeline, as above
`at_end()`          | returns `True` if the end of the timeline has been reaches, `False` otherwise
`dt()`              | returns the size of the timestep
`index()`           | returns the index of the current timestepMRS233p_
`next()`            | jumps to the next timestep
`steps()`           | returns the total number of steps
`time()`            | returns the time at the current timestep
`__repr__()`        | prints the object's string representation
`__str__()`         | prints the object's string representation

### MonteCarlo

### Model


## Modules

### time

TODO



### Model

### Timeline


__NB the following are works-in-progress and subject to change, and the documentation may not reflect the current code__

## Functions

The `neworder` module exposes the following functions to python:

### General, Utility and Diagnostics

name                | description
--------------------|------------------------------------
`version()`         | returns the module version
`python()`          | returns the embedded python version
`log(msg)`          | prints `msg`
`shell()`           | starts an interactive shell (serial mode only)
`reseed()`          | resets the random number stream for the current process

### Time-related

name                  | description
----------------------|------------------------------------
`distant_past()`      | returns a floating-point number that compares less than any other floating point number (i.e. always before)
`far_future()`        | returns a floating-point number that compares greater than any other floating point number (i.e. always after)
`never()`             | returns a floating point number that compares unequal to (and unordered w.r.t) any other number
`isnever(t)`          | returns true if `t` is `never()`. (Direct comparison will always return false)

### Monte-Carlo

name                | description
--------------------|------------------------------------
`ustream(n)`        | returns a numpy 1d array of `n` uniform [0,1) random variables.
`hazard(r, n)`      | returns a numpy 1d array of `n` Bernoulli trials given a scalar hazard rate `r`.
`hazard(r)`         | return a numpy 1d array of Bernoulli trials given a vector of hazard rates `r`.
`stopping(r, n)`    | returns a numpy array of `n` stopping times given a flat hazard rate `r`.
`stopping(r)`       | returns a numpy array of stopping times given a vector of hazard rates `r`.
`arrivals(...)`     | returns sampled arrival times from an "open" non-homogeneous Poisson process (must be a finite probability of no event i.e. 0 or more events per input)
`first_arrival(...)`| returns sampled arrival times from a non-homogeneous Poisson process (can be "open", i.e. 0 or 1 events per input)
`next_arrival(...)` | returns subsequent sampled arrival times from an "open" non-homogeneous Poisson process (i.e. 0 or 1 events per input)

Each process has its own random number stream (Mersenne Twister), which by default is seeded independently. In most cases this is the preferred configuration. However, for sensitivity analysis, e.g. to gauge the impact perturbing the dynamics of the system in multiple runs, it makes more sense for each run to re-use the same sequence in order to eliminate noise from the Monte-Carlo simulation.

### Data Frames
name                           | description
-------------------------------|------------------------------------
`transition(c, t, df, colname)`| Modifies the values in `df.colname` according to a set of possible states `c` and a matrix `t` that specifies the transition probabilities between the states.

### Parallel Execution

Constants

Functions
name                | description
--------------------|------------------------------------
`rank()`            | identifies process for parallel runs (0 if serial)
`size()`            | total number of processes in simulation. (1 if serial)
`INDEP`             | `True` if each process is using an independent same random stream, `False` otherwise
`send(x, n)`        | send object `x` to process `n`
`receive(n)`        | accept object from process `n`
`send_csv(df, n)`   | send pandas DataFrame in csv format to process `n`
`receive_csv(n)`    | accept pandas DataFrame transmitted in csv format from process `n`
`broadcast(x, n)`   | send object `x` from process `n` to all other processes
`gather(x, n)`      | aggregate (numeric) `x` from the current process (i) to the i'th element of a numpy array in process `n`
`scatter(a, n)`     | disperse (numeric) `a` elements from the n'th process to current process (rank is index).
`allgather(a)`      | assemble `a` using elements from each process (rank is index).
`sync()`            | suspend execution of the current process until all processes have reached this point

