# API Reference

TODO update...

The _neworder_ module defines the following classes and modules symbols within the `neworder` namespace:

name                                 | type        | description
-------------------------------------|-------------|--------------
[`Timeline`](#neworder.timeline)     | class       | defines the timeline over which the model is run
[`MonteCarlo`](#neworder.montecarlo) | class       | random number generation and sampling functionality
[`Model`](#neworder.model)           | class       | the base class from which all models should be derived from
[`time`](#neworder.time)             | module      | time-related functionality
[`stats`](#neworder.stats)           | module      | statistical functions
[`dataframe`](#neworder.dataframe)   | module      | direct, fast manipulation of pandas DataFrames
[`mpi`](#neworder.mpi)               | module      | inter-process communication<sup>*</sup>
<p>

and the following functions:

name              | description
------------------|------------------------------------
`log(x)`          | prints x, annotated with process information 
`version()`       | reports the module version
`python()`        | reports the python version
`verbose(v=True)` | sets logging level
`run(m)`          | executes a neworder model
<p>

## Classes

### `neworder.Timeline`

The `Timeline` class describes a timeline containing one of more checkpoints. The timeline can refer to either absolute time, or to the age of the cohort being modelled for case-based models.

The final value in checkpoints must be a list, e.g. `Timeline(2020, 2050, [10, 20, 30])` will start the simulation at 2020 and end at 2050 with a yearly timestep and checkpoints at 2030, 2040 and 2050. Likewise, `Timeline(0.0, 100.0, [100])` could represent an age range of 0-100 with one-year steps, and a single checkpoint at age 100.

To generate a 'null' timeline, which is useful for continuous-time models use

```python
timeline = Timeline.null()
```

which is equivalent to `Timeline(0, 0, [1])`.

The timeline is incremented internally by the neworder Model object during the model run and is not directly modifiable in python code once instantiated. The following accessors are provided:

name                | description
--------------------|------------------------------------
`null()`            | construct null timeline, as above
`at_checkpoint()`   | returns `True` if the current step is a checkpoint, `False` otherwise
`at_end()`          | returns `True` if the end of the timeline has been reaches, `False` otherwise
`dt()`              | returns the size of the timestep
`index()`           | returns the index of the current timestep
`steps()`           | returns the total number of steps
`time()`            | returns the time at the current timestep
`__repr__()`        | prints the object's string representation

### `neworder.MonteCarlo`

#### Static Methods

name                                | description
------------------------------------|------------------------------------
`deterministic_identical_stream(_)`   | Seeds all streams with the same value
`deterministic_independent_stream(r)` | Seeds all streams with a value based on the rank of the process
`nondeterministic_stream(_)`          | Seeds the stream with 

#### Class Methods 

name                | description
--------------------|------------------------------------
`MonteCarlo(seeder)`| construct a MonteCarlo object with random stream using the supplied seeder function 
`seed()`            | returns the seed
`reset()`           | resets the stream  
`ustream(n)`        | returns a numpy array of `n` uniform [0,1) random values
`hazard(p,n)`         | returns a numpy array of `n` booleans simulating outcomes from a flat hazard rate `p` 
`hazard(a)`           | returns a numpy array of `len(a)` booleans simulating outcomes for each hazard rate in array `a`
`stopping(p, n)`        | returns a numpy array of `n` sampled stopping times for a constant hazard rate of `p`
`stopping(a)`        | returns a numpy array of sampled stopping times for an each hazard rate in array `a`
`arrivals(p, dt, gap, n)` | returns a numpy array of length n containing arrival times for a time-dependent hazard rate p with time interval `dt` with a minimum separation of `gap` between events
`first_arrival(p, dt, n)`    |
`first_arrival(p, dt, n, gap)` | as above with a minimum time 
`next_arrival()`     |
`next_arrival()`     |
`next_arrival()`     |
`__repr__()`         | prints the object's string representation


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


### neworder.Model

TODO

## Modules

### `neworder.time`

name                  | description
----------------------|------------------------------------
`distant_past()`      | returns a floating-point number that compares less than any other floating point number (i.e. always before)
`far_future()`        | returns a floating-point number that compares greater than any other floating point number (i.e. always after)
`never()`             | returns a floating point number that compares unequal to (and unordered w.r.t) any other number
`isnever(t)`          | returns true if `t` is `never()`. (Direct comparison will always return false)
`isnever(a)`          | returns a boolean array containing True for each element of a that `never()`. 

TODO

### `neworder.mpi`

TODO

### `neworder.dataframe`

name                           | description
-------------------------------|------------------------------------
`transition(c, t, df, colname)`| Modifies the values in `df.colname` according to a set of possible states `c` and a matrix `t` that specifies the transition probabilities between the states.

