# ![module](https://img.shields.io/badge/-module-blue) `neworder`
---

## ![class](https://img.shields.io/badge/-class-darkgreen) `CalendarTimeline`


A calendar-based timeline

### ![instance method](https://img.shields.io/badge/-instance method-orange) `__init__`

```python
__init__(*args, **kwargs)
```
Overloaded function.

```python
 __init__(self: neworder.CalendarTimeline, start: datetime.datetime, end: datetime.datetime, step: int, unit: str) -> None
```


Constructs a calendar-based timeline, given start and end dates, an increment specified as a multiple of days, months or years


```python
 __init__(self: neworder.CalendarTimeline, start: datetime.datetime, step: int, unit: str) -> None
```


Constructs an open-ended calendar-based timeline, given a start date and an increment specified as a multiple of days, months or years.
NB the model will run until the Model.halt() method is explicitly called (from inside the step() method). Note also that nsteps() will
return -1 for timelines constructed this way


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


Returns the number of steps in the timeline (or -1 if open-ended)


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

## ![class](https://img.shields.io/badge/-class-darkgreen) `Domain`


Base class for spatial domains.

---

## ![class](https://img.shields.io/badge/-class-darkgreen) `LinearTimeline`


An equally-spaced non-calendar timeline .

### ![instance method](https://img.shields.io/badge/-instance method-orange) `__init__`

```python
__init__(*args, **kwargs)
```
Overloaded function.

```python
 __init__(self: neworder.LinearTimeline, start: float, end: float, nsteps: int) -> None
```


Constructs a timeline from start to end, with the given number of steps.


```python
 __init__(self: neworder.LinearTimeline, start: float, step: float) -> None
```


Constructs an open-ended timeline give a start value and a step size. NB the model will run until the Model.halt() method is explicitly called
(from inside the step() method). Note also that nsteps() will return -1 for timelines constructed this way


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


Returns the number of steps in the timeline (or -1 if open-ended)


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
__init__(self: neworder.Model, timeline: neworder.Timeline, seeder: function) -> None
```


Constructs a model object with a timeline and a seeder function


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


### ![instance method](https://img.shields.io/badge/-instance method-orange) `finalise`

```python
finalise(self: neworder.Model) -> None
```


User-overridable function for custom processing after the final step in the model run.
Default behaviour does nothing. This function does not need to be called directly, it is called by the Model.run() function


### ![instance method](https://img.shields.io/badge/-instance method-orange) `halt`

```python
halt(self: neworder.Model) -> None
```


Signal to the model to stop execution gracefully at the end of the current timestep, e.g. if some convergence criterion has been met,
or input is required from an upstream model. The model can be subsequently resumed by calling the run() function.
For trapping exceptional/error conditions, prefer to raise an exception, or return False from the Model.check() function


---

### ![property](https://img.shields.io/badge/-property-lightgreen) `mc`


The model's Monte-Carlo engine

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


---

### ![property](https://img.shields.io/badge/-property-lightgreen) `timeline`


The model's timeline object

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


### ![instance method](https://img.shields.io/badge/-instance method-orange) `counts`

```python
counts(self: neworder.MonteCarlo, lambda: numpy.ndarray[numpy.float64], dt: float) -> numpy.ndarray[numpy.int64]
```


Returns an array of simulated arrival counts (within time dt) for each intensity in lambda


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


Constructs an arbitrary one step timeline, where the start and end times are undefined and there is a single step of size zero. Useful for continuous-time models


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


Returns the number of steps in the timeline (or -1 if open-ended)


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


An custom non-calendar timeline where the user explicitly specifies the time points, which must be monotonically increasing.

### ![instance method](https://img.shields.io/badge/-instance method-orange) `__init__`

```python
__init__(self: neworder.NumericTimeline, times: List[float]) -> None
```


Constructs a timeline from an array of time points.


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


Returns the number of steps in the timeline (or -1 if open-ended)


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

## ![class](https://img.shields.io/badge/-class-darkgreen) `PositionalGrid`


Discrete rectangular n-dimensional domain

---

## ![class](https://img.shields.io/badge/-class-darkgreen) `Space`


Continuous rectangular n-dimensional finite or infinite domain.
If finite, positioning and/or movement near the domain boundary is
dictated by the `wrap` attribute.

---

## ![class](https://img.shields.io/badge/-class-darkgreen) `Timeline`


`__doc__` empty

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

## ![module](https://img.shields.io/badge/-module-blue) `neworder.domain`


Spatial structures for positioning and moving entities and computing distances

---

### ![class](https://img.shields.io/badge/-class-darkgreen) `Domain`


Base class for spatial domains.

---

### ![class](https://img.shields.io/badge/-class-darkgreen) `PositionalGrid`


Discrete rectangular n-dimensional domain

---

### ![class](https://img.shields.io/badge/-class-darkgreen) `Space`


Continuous rectangular n-dimensional finite or infinite domain.
If finite, positioning and/or movement near the domain boundary is
dictated by the `wrap` attribute.

