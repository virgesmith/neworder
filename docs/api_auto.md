# API Reference

TODO reformat raw docstrings

# `neworder` module

```
Help on module neworder:

NAME
    neworder

CLASSES
    pybind11_builtins.pybind11_object(builtins.object)
        Model
        MonteCarlo
        Timeline
    
    class Model(pybind11_builtins.pybind11_object)
     |  Method resolution order:
     |      Model
     |      pybind11_builtins.pybind11_object
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(...)
     |      __init__(self: neworder.Model, arg0: neworder.Timeline, arg1: function) -> None
     |  
     |  check(...)
     |      check(self: neworder.Model) -> bool
     |  
     |  checkpoint(...)
     |      checkpoint(self: neworder.Model) -> None
     |  
     |  mc(...)
     |      mc(self: neworder.Model) -> no::MonteCarlo
     |  
     |  modify(...)
     |      modify(self: neworder.Model, arg0: int) -> None
     |  
     |  step(...)
     |      step(self: neworder.Model) -> None
     |  
     |  timeline(...)
     |      timeline(self: neworder.Model) -> neworder.Timeline
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from pybind11_builtins.pybind11_object:
     |  
     |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
     |      Create and return a new object.  See help(type) for accurate signature.
    
    class MonteCarlo(pybind11_builtins.pybind11_object)
     |  Method resolution order:
     |      MonteCarlo
     |      pybind11_builtins.pybind11_object
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, /, *args, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __repr__(...)
     |      __repr__(self: neworder.MonteCarlo) -> str
     |  
     |  arrivals(...)
     |      arrivals(self: neworder.MonteCarlo, arg0: array, arg1: float, arg2: float, arg3: int) -> array
     |  
     |  first_arrival(...)
     |      first_arrival(*args, **kwargs)
     |      Overloaded function.
     |      
     |      1. first_arrival(self: neworder.MonteCarlo, arg0: array, arg1: float, arg2: int, arg3: float) -> array
     |      
     |      2. first_arrival(self: neworder.MonteCarlo, arg0: array, arg1: float, arg2: int) -> array
     |  
     |  hazard(...)
     |      hazard(*args, **kwargs)
     |      Overloaded function.
     |      
     |      1. hazard(self: neworder.MonteCarlo, arg0: float, arg1: int) -> array
     |      
     |      simulate outcomes from a flat hazard rate
     |      
     |      2. hazard(self: neworder.MonteCarlo, arg0: array) -> array
     |      
     |      simulate outcomes from hazard rates
     |  
     |  next_arrival(...)
     |      next_arrival(*args, **kwargs)
     |      Overloaded function.
     |      
     |      1. next_arrival(self: neworder.MonteCarlo, arg0: array, arg1: array, arg2: float, arg3: bool, arg4: float) -> array
     |      
     |      2. next_arrival(self: neworder.MonteCarlo, arg0: array, arg1: array, arg2: float, arg3: bool) -> array
     |      
     |      3. next_arrival(self: neworder.MonteCarlo, arg0: array, arg1: array, arg2: float) -> array
     |  
     |  reset(...)
     |      reset(self: neworder.MonteCarlo) -> None
     |  
     |  seed(...)
     |      seed(self: neworder.MonteCarlo) -> int
     |  
     |  stopping(...)
     |      stopping(*args, **kwargs)
     |      Overloaded function.
     |      
     |      1. stopping(self: neworder.MonteCarlo, arg0: float, arg1: int) -> array
     |      
     |      simulate stopping times from a flat hazard rate
     |      
     |      2. stopping(self: neworder.MonteCarlo, arg0: array) -> array
     |      
     |      simulate stopping times from hazard rates
     |  
     |  ustream(...)
     |      ustream(self: neworder.MonteCarlo, arg0: int) -> array
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  deterministic_identical_stream(...) from builtins.PyCapsule
     |      deterministic_identical_stream(arg0: int) -> int
     |  
     |  deterministic_independent_stream(...) from builtins.PyCapsule
     |      deterministic_independent_stream(arg0: int) -> int
     |  
     |  nondeterministic_stream(...) from builtins.PyCapsule
     |      nondeterministic_stream(arg0: int) -> int
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from pybind11_builtins.pybind11_object:
     |  
     |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
     |      Create and return a new object.  See help(type) for accurate signature.
    
    class Timeline(pybind11_builtins.pybind11_object)
     |  Method resolution order:
     |      Timeline
     |      pybind11_builtins.pybind11_object
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(...)
     |      __init__(*args, **kwargs)
     |      Overloaded function.
     |      
     |      1. __init__(self: neworder.Timeline, arg0: float, arg1: float, arg2: List[int]) -> None
     |      
     |      2. __init__(self: neworder.Timeline) -> None
     |  
     |  __repr__(...)
     |      __repr__(self: neworder.Timeline) -> str
     |  
     |  at_checkpoint(...)
     |      at_checkpoint(self: neworder.Timeline) -> bool
     |  
     |  at_end(...)
     |      at_end(self: neworder.Timeline) -> bool
     |  
     |  dt(...)
     |      dt(self: neworder.Timeline) -> float
     |  
     |  index(...)
     |      index(self: neworder.Timeline) -> int
     |  
     |  nsteps(...)
     |      nsteps(self: neworder.Timeline) -> int
     |  
     |  time(...)
     |      time(self: neworder.Timeline) -> float
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  null(...) from builtins.PyCapsule
     |      null() -> neworder.Timeline
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from pybind11_builtins.pybind11_object:
     |  
     |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
     |      Create and return a new object.  See help(type) for accurate signature.

FUNCTIONS
    checked(...) method of builtins.PyCapsule instance
        checked(checked: bool = True) -> None
    
    log(...) method of builtins.PyCapsule instance
        log(arg0: object) -> None
        
        
        The logging function. Prints x to the console, annotated with process information
        Args:
        x: object
        Returns:
        None
    
    python(...) method of builtins.PyCapsule instance
        python() -> str
    
    run(...) method of builtins.PyCapsule instance
        run(arg0: object) -> bool
    
    verbose(...) method of builtins.PyCapsule instance
        verbose(verbose: bool = True) -> None
    
    version(...) method of builtins.PyCapsule instance
        version() -> str
        
        
        The version function
        Args:
        Returns:
        str: the module version

FILE
    /mnt/data/home/az/dev/neworder/.venv-focal/lib/python3.8/site-packages/neworder-1.0.0-py3.8-linux-x86_64.egg/neworder.cpython-38-x86_64-linux-gnu.so


```

# `neworder.mpi` module

```
Help on module mpi in neworder:

NAME
    mpi - multiprocess communication

FUNCTIONS
    rank(...) method of builtins.PyCapsule instance
        rank() -> int
        
        
        Returns the MPI rank of the process
        Args:
        Returns:
        int: the MPI rank
    
    size(...) method of builtins.PyCapsule instance
        size() -> int
        
        
        Returns the MPI size (no. of processes) of the run
        Args:
        Returns:
        int: the MPI size

FILE
    (built-in)


```

# `neworder.time` module

```
Help on built-in module time in neworder:

NAME
    time

FUNCTIONS
    distant_past(...) method of builtins.PyCapsule instance
        distant_past() -> float
    
    far_future(...) method of builtins.PyCapsule instance
        far_future() -> float
    
    isnever(...) method of builtins.PyCapsule instance
        isnever(*args, **kwargs)
        Overloaded function.
        
        1. isnever(arg0: float) -> bool
        
        2. isnever(arg0: array) -> array
    
    never(...) method of builtins.PyCapsule instance
        never() -> float

FILE
    (built-in)


```

# `neworder.stats` module

```
Help on module stats in neworder:

NAME
    stats - statistical functions

FUNCTIONS
    logistic(...) method of builtins.PyCapsule instance
        logistic(*args, **kwargs)
        Overloaded function.
        
        1. logistic(arg0: array, arg1: float, arg2: float) -> array
        
        2. logistic(arg0: array, arg1: float) -> array
        
        3. logistic(arg0: array) -> array
    
    logit(...) method of builtins.PyCapsule instance
        logit(arg0: array) -> array

FILE
    (built-in)


```

# `neworder.dataframe` module

```
Help on module dataframe in neworder:

NAME
    dataframe - direct manipulations of dataframes

FUNCTIONS
    directmod(...) method of builtins.PyCapsule instance
        directmod(arg0: neworder.Model, arg1: object, arg2: str) -> None
    
    transition(...) method of builtins.PyCapsule instance
        transition(arg0: neworder.Model, arg1: numpy.ndarray[int64], arg2: numpy.ndarray[float64], arg3: object, arg4: str) -> None

FILE
    (built-in)


```
