# API Reference

TODO reformat raw docstrings

# `neworder` module

```text
Help on module neworder:

NAME
    neworder

CLASSES
    pybind11_builtins.pybind11_object(builtins.object)
        Model
        MonteCarlo
        Timeline
    
    class Model(pybind11_builtins.pybind11_object)
     |  The base model class from which all neworder models should be subclassed
     |  
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
     |      modify(self: neworder.Model, r: int) -> None
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
     |  The model's Monte-Carlo engine
     |  
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
     |      arrivals(self: neworder.MonteCarlo, lambda: numpy.ndarray[float64], dt: float, mingap: float, n: int) -> numpy.ndarray[float64]
     |  
     |  first_arrival(...)
     |      first_arrival(*args, **kwargs)
     |      Overloaded function.
     |      
     |      1. first_arrival(self: neworder.MonteCarlo, lambda: numpy.ndarray[float64], dt: float, n: int, minval: float) -> numpy.ndarray[float64]
     |      
     |      
     |      
     |      
     |      2. first_arrival(self: neworder.MonteCarlo, lambda: numpy.ndarray[float64], dt: float, n: int) -> numpy.ndarray[float64]
     |  
     |  hazard(...)
     |      hazard(*args, **kwargs)
     |      Overloaded function.
     |      
     |      1. hazard(self: neworder.MonteCarlo, p: float, n: int) -> numpy.ndarray[float64]
     |      
     |      
     |      
     |      
     |      2. hazard(self: neworder.MonteCarlo, p: numpy.ndarray[float64]) -> numpy.ndarray[float64]
     |  
     |  next_arrival(...)
     |      next_arrival(*args, **kwargs)
     |      Overloaded function.
     |      
     |      1. next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[float64], lambda: numpy.ndarray[float64], dt: float, relative: bool, minsep: float) -> numpy.ndarray[float64]
     |      
     |      
     |      
     |      
     |      2. next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[float64], lambda: numpy.ndarray[float64], dt: float, relative: bool) -> numpy.ndarray[float64]
     |      
     |      
     |      
     |      
     |      3. next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[float64], lambda: numpy.ndarray[float64], dt: float) -> numpy.ndarray[float64]
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
     |      1. stopping(self: neworder.MonteCarlo, p: float, n: int) -> numpy.ndarray[float64]
     |      
     |      
     |      
     |      
     |      2. stopping(self: neworder.MonteCarlo, p: numpy.ndarray[float64]) -> numpy.ndarray[float64]
     |  
     |  ustream(...)
     |      ustream(self: neworder.MonteCarlo, n: int) -> numpy.ndarray[float64]
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  deterministic_identical_stream(...) from builtins.PyCapsule
     |      deterministic_identical_stream(r: int) -> int
     |  
     |  deterministic_independent_stream(...) from builtins.PyCapsule
     |      deterministic_independent_stream(r: int) -> int
     |  
     |  nondeterministic_stream(...) from builtins.PyCapsule
     |      nondeterministic_stream(r: int) -> int
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from pybind11_builtins.pybind11_object:
     |  
     |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
     |      Create and return a new object.  See help(type) for accurate signature.
    
    class Timeline(pybind11_builtins.pybind11_object)
     |  Timestepping functionality
     |  
     |  Method resolution order:
     |      Timeline
     |      pybind11_builtins.pybind11_object
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(...)
     |      __init__(self: neworder.Timeline, start: float, end: float, checkpoints: List[int]) -> None
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
     |  end(...)
     |      end(self: neworder.Timeline) -> float
     |  
     |  index(...)
     |      index(self: neworder.Timeline) -> int
     |  
     |  nsteps(...)
     |      nsteps(self: neworder.Timeline) -> int
     |  
     |  start(...)
     |      start(self: neworder.Timeline) -> float
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
        
        
        Sets the checked flag, which determines whether the model runs checks during execution
    
    log(...) method of builtins.PyCapsule instance
        log(obj: object) -> None
        
        
        The logging function. Prints obj to the console, annotated with process information
    
    python(...) method of builtins.PyCapsule instance
        python() -> None
    
    run(...) method of builtins.PyCapsule instance
        run(model: object) -> bool
        
        
        Runs the model
        Returns:
            True if model succeeded, False otherwise
    
    verbose(...) method of builtins.PyCapsule instance
        verbose(verbose: bool = True) -> None
        
        
        Sets the verbose flag, which toggles detailed runtime logs
    
    version(...) method of builtins.PyCapsule instance
        version() -> str
        
        
        Gets the module version

FILE
    /mnt/data/home/az/dev/neworder/.venv-focal/lib/python3.8/site-packages/neworder-1.0.0-py3.8-linux-x86_64.egg/neworder.cpython-38-x86_64-linux-gnu.so


```

# `neworder.mpi` module

```text
Help on module mpi in neworder:

NAME
    mpi - Basic MPI environment discovery

FUNCTIONS
    rank(...) method of builtins.PyCapsule instance
        rank() -> int
        
        
        Returns the MPI rank of the process
    
    size(...) method of builtins.PyCapsule instance
        size() -> int
        
        
        Returns the MPI size (no. of processes) of the run

FILE
    (built-in)


```

# `neworder.time` module

```text
Help on built-in module time in neworder:

NAME
    time

FUNCTIONS
    distant_past(...) method of builtins.PyCapsule instance
        distant_past() -> float
        
        
        Returns a value that compares less than any other value but itself and "never"
        Returns:
            -inf
    
    far_future(...) method of builtins.PyCapsule instance
        far_future() -> float
        
        
        Returns a value that compares greater than any other value but itself and "never"
        Returns:
            +inf
    
    isnever(...) method of builtins.PyCapsule instance
        isnever(*args, **kwargs)
        Overloaded function.
        
        1. isnever(t: float) -> bool
        
        
            Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN, 
            direct comparison will always fail, since NaN != NaN. 
            Args:
                t: The time.
            Returns:
                True if t is never, False otherwise
        
        
        2. isnever(y: numpy.ndarray[float64]) -> numpy.ndarray[bool]
        
        
            Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is 
            implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN. 
            Args:
                t: The times.
            Returns:
                Booleans, True where corresponding input value is never, False otherwise
    
    never(...) method of builtins.PyCapsule instance
        never() -> float
        
        
        Returns a value that compares unequal to any value, including but itself.
        Returns:
            nan

FILE
    (built-in)


```

# `neworder.stats` module

```text
Help on module stats in neworder:

NAME
    stats - statistical functions

FUNCTIONS
    logistic(...) method of builtins.PyCapsule instance
        logistic(*args, **kwargs)
        Overloaded function.
        
        1. logistic(x: numpy.ndarray[float64], x0: float, k: float) -> numpy.ndarray[float64]
        
        
            Computes the logistic function on the supplied values. 
            Args:
                x: The input values.
                k: The growth rate
                x0: the midpoint location
            Returns:
                The function values
        
        
        2. logistic(x: numpy.ndarray[float64], k: float) -> numpy.ndarray[float64]
        
        
            Computes the logistic function with x0=0 on the supplied values. 
            Args:
                x: The input values.
                k: The growth rate
            Returns:
                The function values
        
        
        3. logistic(x: numpy.ndarray[float64]) -> numpy.ndarray[float64]
        
        
            Computes the logistic function with k=1 and x0=0 on the supplied values. 
            Args:
                x: The input values.
            Returns:
                The function values
    
    logit(...) method of builtins.PyCapsule instance
        logit(x: numpy.ndarray[float64]) -> numpy.ndarray[float64]
        
        
        Computes the logit function on the supplied values. 
        Args:
            x: The input probability values in (0,1).
        Returns:
            The function values (log-odds)

FILE
    (built-in)


```

# `neworder.df` module

```text
Help on module df in neworder:

NAME
    df - Direct manipulations of dataframes

FUNCTIONS
    directmod(...) method of builtins.PyCapsule instance
        directmod(model: neworder.Model, df: object, colname: str) -> None
    
    transition(...) method of builtins.PyCapsule instance
        transition(model: neworder.Model, categories: numpy.ndarray[int64], transition_matrix: numpy.ndarray[float64], df: object, colname: str) -> None

FILE
    (built-in)


```
