
## `neworder.Model`

!!! note "class"

The base model class from which all neworder models should be subclassed

### `neworder.Model.__init__`

!!! note "instance method"

```python
__init__(self: neworder.Model, timeline: neworder.Timeline, seeder: function) -> None
```


Constructs a model object from a timeline and a seeder function



### `neworder.Model.check`

!!! note "instance method"

```python
check(self: neworder.Model) -> bool
```


User-overridable method used to check internal state at each timestep.
Default behaviour is to simply return True. 
Returning False will halt the model run.
This function should not be called directly, it is used by the Model.run() function 

Returns:
True if checks are ok, False otherwise.



### `neworder.Model.checkpoint`

!!! note "instance method"

```python
checkpoint(self: neworder.Model) -> None
```


User-overridable for custom processing at certain points in the model run (at a minimum the final timestep).
Default behaviour raises NotImplementedError. 
This function should not be called directly, it is used by the Model.run() function 



### `neworder.Model.mc`

!!! note "instance method"

```python
mc(self: neworder.Model) -> no::MonteCarlo
```


Returns the models Monte-Carlo engine



### `neworder.Model.modify`

!!! note "instance method"

```python
modify(self: neworder.Model, r: int) -> None
```


User-overridable method used to modify state in a per-process basis for multiprocess model runs.
Default behaviour is to do nothing. 
This function should not be called directly, it is used by the Model.run() function 



### `neworder.Model.step`

!!! note "instance method"

```python
step(self: neworder.Model) -> None
```


User-implemented method used to advance state of a model.
Default behaviour raises NotImplementedError. 
This function should not be called directly, it is used by the Model.run() function 



### `neworder.Model.timeline`

!!! note "instance method"

```python
timeline(self: neworder.Model) -> neworder.Timeline
```


Returns the model's timeline object



## `neworder.MonteCarlo`

!!! note "class"

The model's Monte-Carlo engine

### `neworder.MonteCarlo.arrivals`

!!! note "instance method"

```python
arrivals(self: neworder.MonteCarlo, lambda: numpy.ndarray[float64], dt: float, mingap: float, n: int) -> numpy.ndarray[float64]
```


Returns an array of n arrays of multiple arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with a minimum separation between events of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm
The final value of lambda must be zero, and thus arrivals don't always occur, indicated by a value of neworder.time.never()
The inner dimension of the returned 2d array is governed by the the maximum number of arrivals sampled, and will thus vary 



### `neworder.MonteCarlo.deterministic_identical_stream`

!!! note "function"

```python
deterministic_identical_stream(r: int) -> int
```


Returns a deterministic seed (19937). Input argument is ignored



### `neworder.MonteCarlo.deterministic_independent_stream`

!!! note "function"

```python
deterministic_independent_stream(r: int) -> int
```


Returns a deterministic seed that is a function of the input (19937+r).
The model uses the MPI rank as the input argument, allowing for differently seeded streams in each process



### `neworder.MonteCarlo.first_arrival`

!!! note "instance method"

```python
first_arrival(*args, **kwargs)
```
Overloaded function.

```python
 first_arrival(self: neworder.MonteCarlo, lambda: numpy.ndarray[float64], dt: float, n: int, minval: float) -> numpy.ndarray[float64]
```


Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with a minimum start time of minval. Sampling uses the Lewis-Shedler "thinning" algorithm
If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()


```python
 first_arrival(self: neworder.MonteCarlo, lambda: numpy.ndarray[float64], dt: float, n: int) -> numpy.ndarray[float64]
```


Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with no minimum start time. Sampling uses the Lewis-Shedler "thinning" algorithm
If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()



### `neworder.MonteCarlo.hazard`

!!! note "instance method"

```python
hazard(*args, **kwargs)
```
Overloaded function.

```python
 hazard(self: neworder.MonteCarlo, p: float, n: int) -> numpy.ndarray[float64]
```


Returns an array of ones (with hazard rate lambda) or zeros of length n


```python
 hazard(self: neworder.MonteCarlo, p: numpy.ndarray[float64]) -> numpy.ndarray[float64]
```


Returns an array of ones (with hazard rate lambda[i]) or zeros for each element in p



### `neworder.MonteCarlo.next_arrival`

!!! note "instance method"

```python
next_arrival(*args, **kwargs)
```
Overloaded function.

```python
 next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[float64], lambda: numpy.ndarray[float64], dt: float, relative: bool, minsep: float) -> numpy.ndarray[float64]
```


Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with start times given by startingpoints with a minimum offset of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm.
If the relative flag is True, then lambda[0] corresponds to start time + mingap, not to absolute time
If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()


```python
 next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[float64], lambda: numpy.ndarray[float64], dt: float, relative: bool) -> numpy.ndarray[float64]
```


Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with start times given by startingpoints. Sampling uses the Lewis-Shedler "thinning" algorithm.
If the relative flag is True, then lambda[0] corresponds to start time, not to absolute time
If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()


```python
 next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[float64], lambda: numpy.ndarray[float64], dt: float) -> numpy.ndarray[float64]
```


Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
with start times given by startingpoints. Sampling uses the Lewis-Shedler "thinning" algorithm.
If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()



### `neworder.MonteCarlo.nondeterministic_stream`

!!! note "function"

```python
nondeterministic_stream(r: int) -> int
```


Returns a random seed from the platform's random_device. Input argument is ignored



### `neworder.MonteCarlo.reset`

!!! note "instance method"

```python
reset(self: neworder.MonteCarlo) -> None
```


Resets the generator using the original seed.
Use with care, esp in multi-process models with identical streams



### `neworder.MonteCarlo.seed`

!!! note "instance method"

```python
seed(self: neworder.MonteCarlo) -> int
```


Returns the seed used to initialise the random stream



### `neworder.MonteCarlo.stopping`

!!! note "instance method"

```python
stopping(*args, **kwargs)
```
Overloaded function.

```python
 stopping(self: neworder.MonteCarlo, lambda: float, n: int) -> numpy.ndarray[float64]
```


Returns an array of stopping times (with hazard rate lambda) of length n


```python
 stopping(self: neworder.MonteCarlo, lambda: numpy.ndarray[float64]) -> numpy.ndarray[float64]
```


Returns an array of stopping times (with hazard rate lambda[i]) for each element in lambda



### `neworder.MonteCarlo.ustream`

!!! note "instance method"

```python
ustream(self: neworder.MonteCarlo, n: int) -> numpy.ndarray[float64]
```


Returns an array of uniform random [0,1) variates of length n



## `neworder.Timeline`

!!! note "class"

Timestepping functionality

### `neworder.Timeline.__init__`

!!! note "instance method"

```python
__init__(self: neworder.Timeline, start: float, end: float, checkpoints: List[int]) -> None
```


Constructs a timeline from start to end, with the checkpoints given by a non-empty list of ascending integers.
The total number of steps and the step size is determined by the final checkpoint value



### `neworder.Timeline.at_checkpoint`

!!! note "instance method"

```python
at_checkpoint(self: neworder.Timeline) -> bool
```


Returns True if the current step is a checkpoint



### `neworder.Timeline.at_end`

!!! note "instance method"

```python
at_end(self: neworder.Timeline) -> bool
```


Returns True if the current step is the end of the timeline



### `neworder.Timeline.dt`

!!! note "instance method"

```python
dt(self: neworder.Timeline) -> float
```


Returns the step size size of the timeline



### `neworder.Timeline.end`

!!! note "instance method"

```python
end(self: neworder.Timeline) -> float
```


Returns the time of the end of the timeline



### `neworder.Timeline.index`

!!! note "instance method"

```python
index(self: neworder.Timeline) -> int
```


Returns the index of the current step in the timeline



### `neworder.Timeline.nsteps`

!!! note "instance method"

```python
nsteps(self: neworder.Timeline) -> int
```


Returns the number of steps in the timeline



### `neworder.Timeline.null`

!!! note "function"

```python
null() -> neworder.Timeline
```


Returns a "null" timeline, where the start and end times are zero and there is a single step and checkpoint
Useful for continuous-time models with no explicit (discrete) timeline



### `neworder.Timeline.start`

!!! note "instance method"

```python
start(self: neworder.Timeline) -> float
```


Returns the time of the start of the timeline



### `neworder.Timeline.time`

!!! note "instance method"

```python
time(self: neworder.Timeline) -> float
```


Returns the time of the current step in the timeline



## `neworder.checked`

!!! note "function"

```python
checked(checked: bool = True) -> None
```


Sets the checked flag, which determines whether the model runs checks during execution



## `neworder.df`

!!! note "module"

Direct manipulations of dataframes

### `neworder.df.testfunc`

!!! note "function"

```python
testfunc(model: neworder.Model, df: object, colname: str) -> None
```


Test function for direct dataframe manipulation. Results may vary. Do not use.



### `neworder.df.transition`

!!! note "function"

```python
transition(model: neworder.Model, categories: numpy.ndarray[int64], transition_matrix: numpy.ndarray[float64], df: object, colname: str) -> None
```


Randomly changes categorical data in a dataframe, according to supplied transition probabilities. 
Args:
model: The model (for access to the MonteCarlo engine).
categories: The set of possible categories
transition_matrix: The probabilities of transitions between categories
df: The dataframe, which is modified in-place
colname: The name of the column in the dataframe



### `neworder.df.unique_index`

!!! note "function"

```python
unique_index(n: int) -> numpy.ndarray[int64]
```


Generates an array of unique values, even across multiple processes, that can be used to uniquely index multiple dataframes.
Args:
n: The number of required index values.
Returns:
The unique index values



## `neworder.log`

!!! note "function"

```python
log(obj: object) -> None
```


The logging function. Prints obj to the console, annotated with process information



## `neworder.mpi`

!!! note "module"

Basic MPI environment discovery

### `neworder.mpi.rank`

!!! note "function"

```python
rank() -> int
```


Returns the MPI rank of the process



### `neworder.mpi.size`

!!! note "function"

```python
size() -> int
```


Returns the MPI size (no. of processes) of the run



## `neworder.run`

!!! note "function"

```python
run(model: object) -> bool
```


Runs the model
Returns:
True if model succeeded, False otherwise



## `neworder.stats`

!!! note "module"

statistical functions

### `neworder.stats.logistic`

!!! note "function"

```python
logistic(*args, **kwargs)
```
Overloaded function.

```python
 logistic(x: numpy.ndarray[float64], x0: float, k: float) -> numpy.ndarray[float64]
```


Computes the logistic function on the supplied values. 
Args:
x: The input values.
k: The growth rate
x0: the midpoint location
Returns:
The function values


```python
 logistic(x: numpy.ndarray[float64], k: float) -> numpy.ndarray[float64]
```


Computes the logistic function with x0=0 on the supplied values. 
Args:
x: The input values.
k: The growth rate
Returns:
The function values


```python
 logistic(x: numpy.ndarray[float64]) -> numpy.ndarray[float64]
```


Computes the logistic function with k=1 and x0=0 on the supplied values. 
Args:
x: The input values.
Returns:
The function values



### `neworder.stats.logit`

!!! note "function"

```python
logit(x: numpy.ndarray[float64]) -> numpy.ndarray[float64]
```


Computes the logit function on the supplied values. 
Args:
x: The input probability values in (0,1).
Returns:
The function values (log-odds)



## `neworder.time`

!!! note "module"

__doc__ empty

### `neworder.time.distant_past`

!!! note "function"

```python
distant_past() -> float
```


Returns a value that compares less than any other value but itself and "never"
Returns:
-inf



### `neworder.time.far_future`

!!! note "function"

```python
far_future() -> float
```


Returns a value that compares greater than any other value but itself and "never"
Returns:
+inf



### `neworder.time.isnever`

!!! note "function"

```python
isnever(*args, **kwargs)
```
Overloaded function.

```python
 isnever(t: float) -> bool
```


Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN, 
direct comparison will always fail, since NaN != NaN. 
Args:
t: The time.
Returns:
True if t is never, False otherwise


```python
 isnever(y: numpy.ndarray[float64]) -> numpy.ndarray[bool]
```


Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is 
implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN. 
Args:
t: The times.
Returns:
Booleans, True where corresponding input value is never, False otherwise



### `neworder.time.never`

!!! note "function"

```python
never() -> float
```


Returns a value that compares unequal to any value, including but itself.
Returns:
nan



## `neworder.verbose`

!!! note "function"

```python
verbose(verbose: bool = True) -> None
```


Sets the verbose flag, which toggles detailed runtime logs



## `neworder.version`

!!! note "function"

```python
version() -> str
```


Gets the module version


