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

__doc__ empty
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

