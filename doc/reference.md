# Reference

## Reserved Variables

### User defined

The neworder python module defines, or requires the model to define, the following variables withing the neworder namespace:

name       | type        | default | description
-----------|-------------|---------|--------------
`timestep` | float       |         | the size of the timestep 
`timeline` | float array |         | the timeline of the simulation 
`log_level`| int         |         | currently unused. control verbosity of output 
`do_checks`| bool        |         | run functions specified in `checks` after each timestep 
`initialisations`| dict  |         | initialisation function(s)/constructors  
`modifiers`| list        | []      | expressions to modify/perturb input data for each process   
`transitions`| dict      |         | model evolution functions (per-timestep)  
`checks`   | dict        | {}      | specify functions to run after each timestep 
`checkpoints`| dict      |         | perform specified code at each checkpoint 

### Runtime

Additionally, the module creates the following runtime variable:

name       | type        | default | description
-----------|-------------|---------|--------------
`time`     | float       |         | time of current timestep
`timeindex`| int         |         | index of current timestep

## Functions
The `neworder` module exposes the following functions to python:

### General, Utility and Diagnostics

name                | description
--------------------|------------------------------------
`name()`            | returns the module name (`neworder`)
`version()`         | returns the module version
`python()`          | returns the embedded python version
`log(msg)`          | prints `msg`
`shell()`           | starts an interactive shell (serial mode only)
`lazy_exec(expr)`   | creates a python expression `expr` for later execution via the `()` operator
`lazy_eval(expr)`   | creates a python expression `expr` for later evaluation via the `()` operator
`reseed()`          | resets the random number stream for the current process

### Time-related

name                | description
--------------------|------------------------------------
`distant_past()`    | returns a floating-point number that compares less than any other floating point number (i.e. always before)
`far_future()`      | returns a floating-point number that compares greater than any other floating point number (i.e. always after)


### Monte-Carlo

name                | description
--------------------|------------------------------------
`ustream(n)`        | returns a numpy 1d array of `n` uniform [0,1) random variables.
`hazard(r, n)`      | returns a numpy 1d array of `n` Bernoulli trials given a hazard rate `r`.
`hazard_v()`        | return a numpy 1d array of Bernoulli trials given a hazard rate `r`.
`stopping(r, n)`    | returns a numpy array of `n` stopping times given a fixed hazard rate `r`. 
`stopping_v()`      | returns a vector of stopping times given a fixed hazard rate and a length     |  
`stopping_nhpp()`   | returns stopping times from a non-homogeneous Poisson process 

Each process has its own random number stream (Mersenne Twister), which by default is seeded independently. In most cases this is the preferred configuration. However, for sensitivity analysis, e.g. to gauge the impact perturbing the dynamics of the system in multiple runs, it makes more sense for each run to re-use the same sequence in order to eliminate noise from the Monte-Carlo simulation.  

### Data Frames
name                           | description
-------------------------------|------------------------------------
`transition(c, t, df, colname)`| Modifies the values in `df.colname` according to a set of possible states `c` and a matrix `t` that specifies the transition probabilities between the states.

### Parallel Execution
name                | description
--------------------|------------------------------------
`rank()`            | identifies process for parallel runs (0 if serial)
`size()`            | total number of processes in simulation. (1 if serial)
`indep()`           | returns `True` if each process is using an independent same random stream, `False` otherwise
`send(x, n)`        | send object `x` to process `n` 
`receive(n)`        | accept object from process `n`
`send_csv(df, n)`   | send pandas DataFrame in csv format to process `n`
`receive_csv(n)`    | accept pandas DataFrame transmitted in csv format from process `n`
`broadcast(x, n)`   | send object `x` from process `n` to all other processes
`gather(x, n)`      | aggregate (numeric) `x` from the current process (i) to the i'th element of a numpy array in process `n`
`scatter(a, n)`     | disperse (numeric) `a` elements from the n'th process to current process (rank is index). 
`allgather(a)`      | assemble `a` using elements from each process (rank is index). 
`sync()`            | suspend execution of the current process until all processes have reached this point

## Implementation Detail

As python and C++ have very different memory models, it's not possible to directly share data, i.e. safely have a python object and a C++ object both referencing (and potentially modifying) the same memory location. However, there is a crucial exception to this: the numpy ndarray type. This is fundamental to the operation of the framework, as it enables the C++ module to directly access (and modify) pandas data frames (which do not have a native C API).
