# ![module](https://img.shields.io/badge/-module-blue) `neworder`
---

## ![class](https://img.shields.io/badge/-class-darkgreen) `CalendarTimeline`


A calendar-based timeline

### ![instance method](https://img.shields.io/badge/-instance method-orange) `__init__`

```python
__init__(self: neworder.CalendarTimeline, start: datetime.datetime, end: datetime.datetime, step: int, unit: str, n_checkpoints: int) -> None
```


Constructs a calendar-based timeline, given start and end dates, an increment specified as a multiple of days, months or years, and the number
of checkpoints required. Checkpoints are spread evenly over the timeline and always include the final time point


### ![instance method](https://img.shields.io/badge/-instance method-orange) `at_checkpoint`

```python
at_checkpoint(self: neworder.CalendarTimeline) -> bool
```


Returns True if the current step is a checkpoint


### ![instance method](https://img.shields.io/badge/-instance method-orange) `at_end`

```python
at_end(self: neworder.CalendarTimeline) -> bool
```


Returns True if the current step is the end of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `dt`

```python
dt(self: neworder.CalendarTimeline) -> float
```


Returns the step size size of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `end`

```python
end(self: neworder.CalendarTimeline) -> object
```


Returns the time of the end of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `index`

```python
index(self: neworder.CalendarTimeline) -> int
```


Returns the index of the current step in the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `nsteps`

```python
nsteps(self: neworder.CalendarTimeline) -> int
```


Returns the number of steps in the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `start`

```python
start(self: neworder.CalendarTimeline) -> object
```


Returns the time of the start of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `time`

```python
time(self: neworder.CalendarTimeline) -> object
```


Returns the time of the current step in the timeline


---

## ![class](https://img.shields.io/badge/-class-darkgreen) `LinearTimeline`


An equally-spaced non-calendar timeline .

### ![instance method](https://img.shields.io/badge/-instance method-orange) `__init__`

```python
__init__(self: neworder.LinearTimeline, start: float, end: float, checkpoints: List[int]) -> None
```


Constructs a timeline from start to end, with the checkpoints given by a non-empty list of ascending integers.
The total number of steps and the step size is determined by the final checkpoint value


### ![instance method](https://img.shields.io/badge/-instance method-orange) `at_checkpoint`

```python
at_checkpoint(self: neworder.LinearTimeline) -> bool
```


Returns True if the current step is a checkpoint


### ![instance method](https://img.shields.io/badge/-instance method-orange) `at_end`

```python
at_end(self: neworder.LinearTimeline) -> bool
```


Returns True if the current step is the end of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `dt`

```python
dt(self: neworder.LinearTimeline) -> float
```


Returns the step size size of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `end`

```python
end(self: neworder.LinearTimeline) -> object
```


Returns the time of the end of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `index`

```python
index(self: neworder.LinearTimeline) -> int
```


Returns the index of the current step in the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `nsteps`

```python
nsteps(self: neworder.LinearTimeline) -> int
```


Returns the number of steps in the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `start`

```python
start(self: neworder.LinearTimeline) -> object
```


Returns the time of the start of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `time`

```python
time(self: neworder.LinearTimeline) -> object
```


Returns the time of the current step in the timeline


---

## ![class](https://img.shields.io/badge/-class-darkgreen) `Model`


The base model class from which all neworder models should be subclassed

### ![instance method](https://img.shields.io/badge/-instance method-orange) `__init__`

```python
__init__(*args, **kwargs)
```
Overloaded function.

```python
 __init__(self: neworder.Model, timeline: neworder.NoTimeline, seeder: function) -> None
```


Constructs a model object with an empty timeline and a seeder function, for continuous-time models


```python
 __init__(self: neworder.Model, timeline: neworder.LinearTimeline, seeder: function) -> None
```


Constructs a model object from a linear timeline and a seeder function, providing equally spaced timesteps


```python
 __init__(self: neworder.Model, timeline: neworder.NumericTimeline, seeder: function) -> None
```


Constructs a model object from a numeric timeline and a seeder function, allowing user defined timesteps


```python
 __init__(self: neworder.Model, timeline: neworder.CalendarTimeline, seeder: function) -> None
```


Constructs a model object from a calendar timeline and a seeder function, with date-based timesteps


### ![instance method](https://img.shields.io/badge/-instance method-orange) `check`

```python
check(self: neworder.Model) -> bool
```


User-overridable method used to check internal state at each timestep.
Default behaviour is to simply return True.
Returning False will halt the model run.
This function should not be called directly, it is used by the Model.run() function

Returns:
True if checks are ok, False otherwise.


### ![instance method](https://img.shields.io/badge/-instance method-orange) `checkpoint`

```python
checkpoint(self: neworder.Model) -> None
```


User-overridable for custom processing at certain points in the model run (at a minimum the final timestep).
Default behaviour raises NotImplementedError.
This function should not be called directly, it is used by the Model.run() function


### ![instance method](https://img.shields.io/badge/-instance method-orange) `halt`

```python
halt(self: neworder.Model) -> None
```


Signal to the model to stop execution gracefully at the end of the current timestep, e.g. if some convergence criterion has been met,
or input is required from an upstream model. The model can be subsequently resumed by calling the run() function.
For trapping exceptional/error conditions, prefer to raise an exception, or return False from the Model.check() function


### ![instance method](https://img.shields.io/badge/-instance method-orange) `mc`

```python
mc(self: neworder.Model) -> no::MonteCarlo
```


Returns the models Monte-Carlo engine


### ![instance method](https://img.shields.io/badge/-instance method-orange) `modify`

```python
modify(self: neworder.Model, r: int) -> None
```


User-overridable method used to modify state in a per-process basis for multiprocess model runs.
Default behaviour is to do nothing.
This function should not be called directly, it is used by the Model.run() function


### ![instance method](https://img.shields.io/badge/-instance method-orange) `step`

```python
step(self: neworder.Model) -> None
```


User-implemented method used to advance state of a model.
Default behaviour raises NotImplementedError.
This function should not be called directly, it is used by the Model.run() function


### ![instance method](https://img.shields.io/badge/-instance method-orange) `timeline`

```python
timeline(self: neworder.Model) -> no::Timeline
```


Returns the model's timeline object


---

## ![class](https://img.shields.io/badge/-class-darkgreen) `MonteCarlo`


The model's Monte-Carlo engine with configurable options for parallel execution

### ![instance method](https://img.shields.io/badge/-instance method-orange) `arrivals`

```python
arrivals(self: neworder.MonteCarlo, lambda: numpy.ndarray[numpy.float64], dt: float, n: int, mingap: float) -> numpy.ndarray[numpy.float64]
```


Returns an array of n arrays of multiple arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with a minimum separation between events of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm
The final value of lambda must be zero, and thus arrivals don't always occur, indicated by a value of neworder.time.never()
The inner dimension of the returned 2d array is governed by the the maximum number of arrivals sampled, and will thus vary


### ![function](https://img.shields.io/badge/-function-red) `deterministic_identical_stream`

```python
deterministic_identical_stream(r: int) -> int
```


Returns a deterministic seed (19937). Input argument is ignored


### ![function](https://img.shields.io/badge/-function-red) `deterministic_independent_stream`

```python
deterministic_independent_stream(r: int) -> int
```


Returns a deterministic seed that is a function of the input (19937+r).
The model uses the MPI rank as the input argument, allowing for differently seeded streams in each process


### ![instance method](https://img.shields.io/badge/-instance method-orange) `first_arrival`

```python
first_arrival(*args, **kwargs)
```
Overloaded function.

```python
 first_arrival(self: neworder.MonteCarlo, lambda: numpy.ndarray[numpy.float64], dt: float, n: int, minval: float) -> numpy.ndarray[numpy.float64]
```


Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with a minimum start time of minval. Sampling uses the Lewis-Shedler "thinning" algorithm
If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()


```python
 first_arrival(self: neworder.MonteCarlo, lambda: numpy.ndarray[numpy.float64], dt: float, n: int) -> numpy.ndarray[numpy.float64]
```


Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with no minimum start time. Sampling uses the Lewis-Shedler "thinning" algorithm
If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()


### ![instance method](https://img.shields.io/badge/-instance method-orange) `hazard`

```python
hazard(*args, **kwargs)
```
Overloaded function.

```python
 hazard(self: neworder.MonteCarlo, p: float, n: int) -> numpy.ndarray[numpy.float64]
```


Returns an array of ones (with hazard rate lambda) or zeros of length n


```python
 hazard(self: neworder.MonteCarlo, p: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]
```


Returns an array of ones (with hazard rate lambda[i]) or zeros for each element in p


### ![instance method](https://img.shields.io/badge/-instance method-orange) `next_arrival`

```python
next_arrival(*args, **kwargs)
```
Overloaded function.

```python
 next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[numpy.float64], lambda: numpy.ndarray[numpy.float64], dt: float, relative: bool, minsep: float) -> numpy.ndarray[numpy.float64]
```


Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with start times given by startingpoints with a minimum offset of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm.
If the relative flag is True, then lambda[0] corresponds to start time + mingap, not to absolute time
If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()


```python
 next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[numpy.float64], lambda: numpy.ndarray[numpy.float64], dt: float, relative: bool) -> numpy.ndarray[numpy.float64]
```


Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with start times given by startingpoints. Sampling uses the Lewis-Shedler "thinning" algorithm.
If the relative flag is True, then lambda[0] corresponds to start time, not to absolute time
If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()


```python
 next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[numpy.float64], lambda: numpy.ndarray[numpy.float64], dt: float) -> numpy.ndarray[numpy.float64]
```


Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with start times given by startingpoints. Sampling uses the Lewis-Shedler "thinning" algorithm.
If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()


### ![function](https://img.shields.io/badge/-function-red) `nondeterministic_stream`

```python
nondeterministic_stream(r: int) -> int
```


Returns a random seed from the platform's random_device. Input argument is ignored


### ![instance method](https://img.shields.io/badge/-instance method-orange) `raw`

```python
raw(self: neworder.MonteCarlo) -> int
```


Returns a random 64-bit unsigned integer. Useful for seeding other generators.


### ![instance method](https://img.shields.io/badge/-instance method-orange) `reset`

```python
reset(self: neworder.MonteCarlo) -> None
```


Resets the generator using the original seed.
Use with care, esp in multi-process models with identical streams


### ![instance method](https://img.shields.io/badge/-instance method-orange) `sample`

```python
sample(self: neworder.MonteCarlo, n: int, cat_weights: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.int64]
```


Returns an array of length n containing randomly sampled categorical values, weighted according to cat_weights


### ![instance method](https://img.shields.io/badge/-instance method-orange) `seed`

```python
seed(self: neworder.MonteCarlo) -> int
```


Returns the seed used to initialise the random stream


### ![instance method](https://img.shields.io/badge/-instance method-orange) `state`

```python
state(self: neworder.MonteCarlo) -> int
```


Returns a hash of the internal state of the generator. Avoids the extra complexity of tranmitting variable-length strings over MPI.


### ![instance method](https://img.shields.io/badge/-instance method-orange) `stopping`

```python
stopping(*args, **kwargs)
```
Overloaded function.

```python
 stopping(self: neworder.MonteCarlo, lambda: float, n: int) -> numpy.ndarray[numpy.float64]
```


Returns an array of stopping times (with hazard rate lambda) of length n


```python
 stopping(self: neworder.MonteCarlo, lambda: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]
```


Returns an array of stopping times (with hazard rate lambda[i]) for each element in lambda


### ![instance method](https://img.shields.io/badge/-instance method-orange) `ustream`

```python
ustream(self: neworder.MonteCarlo, n: int) -> numpy.ndarray[numpy.float64]
```


Returns an array of uniform random [0,1) variates of length n


---

## ![class](https://img.shields.io/badge/-class-darkgreen) `NoTimeline`


An arbitrary one step timeline, for continuous-time models with no explicit (discrete) timeline

### ![instance method](https://img.shields.io/badge/-instance method-orange) `__init__`

```python
__init__(self: neworder.NoTimeline) -> None
```


Constructs an arbitrary one step timeline, where the start and end times are undefined and there is a single step and a single checkpoint


### ![instance method](https://img.shields.io/badge/-instance method-orange) `at_checkpoint`

```python
at_checkpoint(self: neworder.NoTimeline) -> bool
```


Returns True if the current step is a checkpoint


### ![instance method](https://img.shields.io/badge/-instance method-orange) `at_end`

```python
at_end(self: neworder.NoTimeline) -> bool
```


Returns True if the current step is the end of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `dt`

```python
dt(self: neworder.NoTimeline) -> float
```


Returns the step size size of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `end`

```python
end(self: neworder.NoTimeline) -> object
```


Returns the time of the end of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `index`

```python
index(self: neworder.NoTimeline) -> int
```


Returns the index of the current step in the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `nsteps`

```python
nsteps(self: neworder.NoTimeline) -> int
```


Returns the number of steps in the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `start`

```python
start(self: neworder.NoTimeline) -> object
```


Returns the time of the start of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `time`

```python
time(self: neworder.NoTimeline) -> object
```


Returns the time of the current step in the timeline


---

## ![class](https://img.shields.io/badge/-class-darkgreen) `NumericTimeline`


An custom non-claendar timeline

### ![instance method](https://img.shields.io/badge/-instance method-orange) `__init__`

```python
__init__(self: neworder.NumericTimeline, times: List[float], checkpoints: List[int]) -> None
```


Constructs a timeline from an array of time points and a subset of indices that are checkpoints.
The checkpoint array must contain at least the index of the final point on the timeline.


### ![instance method](https://img.shields.io/badge/-instance method-orange) `at_checkpoint`

```python
at_checkpoint(self: neworder.NumericTimeline) -> bool
```


Returns True if the current step is a checkpoint


### ![instance method](https://img.shields.io/badge/-instance method-orange) `at_end`

```python
at_end(self: neworder.NumericTimeline) -> bool
```


Returns True if the current step is the end of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `dt`

```python
dt(self: neworder.NumericTimeline) -> float
```


Returns the step size size of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `end`

```python
end(self: neworder.NumericTimeline) -> object
```


Returns the time of the end of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `index`

```python
index(self: neworder.NumericTimeline) -> int
```


Returns the index of the current step in the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `nsteps`

```python
nsteps(self: neworder.NumericTimeline) -> int
```


Returns the number of steps in the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `start`

```python
start(self: neworder.NumericTimeline) -> object
```


Returns the time of the start of the timeline


### ![instance method](https://img.shields.io/badge/-instance method-orange) `time`

```python
time(self: neworder.NumericTimeline) -> object
```


Returns the time of the current step in the timeline


---

## ![function](https://img.shields.io/badge/-function-red) `checked`

```python
checked(checked: bool = True) -> None
```


Sets the checked flag, which determines whether the model runs checks during execution


---

## ![module](https://img.shields.io/badge/-module-blue) `neworder.df`


Submodule for operations involving direct manipulation of pandas dataframes

### ![function](https://img.shields.io/badge/-function-red) `testfunc`

```python
testfunc(model: neworder.Model, df: object, colname: str) -> None
```


Test function for direct dataframe manipulation. Results may vary. Do not use.


### ![function](https://img.shields.io/badge/-function-red) `transition`

```python
transition(model: neworder.Model, categories: numpy.ndarray[numpy.int64], transition_matrix: numpy.ndarray[numpy.float64], df: object, colname: str) -> None
```


Randomly changes categorical data in a dataframe, according to supplied transition probabilities.
Args:
model: The model (for access to the MonteCarlo engine).
categories: The set of possible categories
transition_matrix: The probabilities of transitions between categories
df: The dataframe, which is modified in-place
colname: The name of the column in the dataframe


### ![function](https://img.shields.io/badge/-function-red) `unique_index`

```python
unique_index(n: int) -> numpy.ndarray[numpy.int64]
```


Generates an array of n unique values, even across multiple processes, that can be used to unambiguously index multiple dataframes.


---

## ![function](https://img.shields.io/badge/-function-red) `log`

```python
log(obj: object) -> None
```


The logging function. Prints obj to the console, annotated with process information


---

## ![module](https://img.shields.io/badge/-module-blue) `neworder.mpi`


Submodule for basic MPI environment discovery

### ![function](https://img.shields.io/badge/-function-red) `rank`

```python
rank() -> int
```


Returns the MPI rank of the process


### ![function](https://img.shields.io/badge/-function-red) `size`

```python
size() -> int
```


Returns the MPI size (no. of processes) of the run


---

## ![function](https://img.shields.io/badge/-function-red) `run`

```python
run(model: object) -> bool
```


Runs the model. If the model has previously run it will resume from the point at which it was given the "halt" instruction. This is useful
for external processing of model data, and/or feedback from external sources. If the model has already reached the end of the timeline, this
function will have no effect. To re-run the model from the start, you must construct a new model object.
Returns:
True if model succeeded, False otherwise


---

## ![module](https://img.shields.io/badge/-module-blue) `neworder.stats`


Submodule for statistical functions

### ![function](https://img.shields.io/badge/-function-red) `logistic`

```python
logistic(*args, **kwargs)
```
Overloaded function.

```python
 logistic(x: numpy.ndarray[numpy.float64], x0: float, k: float) -> numpy.ndarray[numpy.float64]
```


Computes the logistic function on the supplied values.
Args:
x: The input values.
k: The growth rate
x0: the midpoint location
Returns:
The function values


```python
 logistic(x: numpy.ndarray[numpy.float64], k: float) -> numpy.ndarray[numpy.float64]
```


Computes the logistic function with x0=0 on the supplied values.
Args:
x: The input values.
k: The growth rate
Returns:
The function values


```python
 logistic(x: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]
```


Computes the logistic function with k=1 and x0=0 on the supplied values.
Args:
x: The input values.
Returns:
The function values


### ![function](https://img.shields.io/badge/-function-red) `logit`

```python
logit(x: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]
```


Computes the logit function on the supplied values.
Args:
x: The input probability values in (0,1).
Returns:
The function values (log-odds)


---

## ![module](https://img.shields.io/badge/-module-blue) `neworder.time`


Temporal values and comparison

### ![function](https://img.shields.io/badge/-function-red) `distant_past`

```python
distant_past() -> float
```


Returns a value that compares less than any other value but itself and "never"


### ![function](https://img.shields.io/badge/-function-red) `far_future`

```python
far_future() -> float
```


Returns a value that compares greater than any other value but itself and "never"


### ![function](https://img.shields.io/badge/-function-red) `isnever`

```python
isnever(*args, **kwargs)
```
Overloaded function.

```python
 isnever(t: float) -> bool
```


Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN,
direct comparison will always fail, since NaN != NaN.


```python
 isnever(t: numpy.ndarray[numpy.float64]) -> numpy.ndarray[bool]
```


Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is
implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN.


### ![function](https://img.shields.io/badge/-function-red) `never`

```python
never() -> float
```


Returns a value that compares unequal to any value, including but itself.


---

## ![function](https://img.shields.io/badge/-function-red) `verbose`

```python
verbose(verbose: bool = True) -> None
```


Sets the verbose flag, which toggles detailed runtime logs


---

## ![function](https://img.shields.io/badge/-function-red) `version`

```python
version() -> str
```


Gets the module version


