# API Reference

!!! note "API documentation"
    We have yet to get an API doc plugin working with the pybind11 module. In the meantime, the raw docstrings are reproduced verbatim below.


## `neworder` module

```text
Help on module neworder:

NAME
    neworder - A dynamic microsimulation framework

CLASSES
    pybind11_builtins.pybind11_object(builtins.object)
        Model
        MonteCarlo
        Timeline

    class Model(pybind11_builtins.pybind11_object)
        The base model class from which all neworder models should be subclassed

        Method resolution order:
            Model
            pybind11_builtins.pybind11_object
            builtins.object

        Methods defined here:

        __init__(...)
            __init__(self: neworder.Model, timeline: neworder.Timeline, seeder: function) -> None


            Constructs a model object from a timeline and a seeder function

        check(...)
            check(self: neworder.Model) -> bool


            User-overridable method used to check internal state at each timestep.
            Default behaviour is to simply return True.
            Returning False will halt the model run.
            This function should not be called directly, it is used by the Model.run() function

            Returns:
                True if checks are ok, False otherwise.

        checkpoint(...)
            checkpoint(self: neworder.Model) -> None


            User-overridable for custom processing at certain points in the model run (at a minimum the final timestep).
            Default behaviour raises NotImplementedError.
            This function should not be called directly, it is used by the Model.run() function

        mc(...)
            mc(self: neworder.Model) -> no::MonteCarlo


            Returns the models Monte-Carlo engine

        modify(...)
            modify(self: neworder.Model, r: int) -> None


            User-overridable method used to modify state in a per-process basis for multiprocess model runs.
            Default behaviour is to do nothing.
            This function should not be called directly, it is used by the Model.run() function

        step(...)
            step(self: neworder.Model) -> None


            User-implemented method used to advance state of a model.
            Default behaviour raises NotImplementedError.
            This function should not be called directly, it is used by the Model.run() function

        timeline(...)
            timeline(self: neworder.Model) -> neworder.Timeline


            Returns the model's timeline object

        ----------------------------------------------------------------------
        Static methods inherited from pybind11_builtins.pybind11_object:

        __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
            Create and return a new object.  See help(type) for accurate signature.

    class MonteCarlo(pybind11_builtins.pybind11_object)
        The model's Monte-Carlo engine

        Method resolution order:
            MonteCarlo
            pybind11_builtins.pybind11_object
            builtins.object

        Methods defined here:

        __init__(self, /, *args, **kwargs)
            Initialize self.  See help(type(self)) for accurate signature.

        __repr__(...)
            __repr__(self: neworder.MonteCarlo) -> str


            Prints a human-readable representation of the MonteCarlo engine

        arrivals(...)
            arrivals(self: neworder.MonteCarlo, lambda: numpy.ndarray[float64], dt: float, mingap: float, n: int) -> numpy.ndarray[float64]


            Returns an array of n arrays of multiple arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
            with a minimum separation between events of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm
            The final value of lambda must be zero, and thus arrivals don't always occur, indicated by a value of neworder.time.never()
            The inner dimension of the returned 2d array is governed by the the maximum number of arrivals sampled, and will thus vary

        first_arrival(...)
            first_arrival(*args, **kwargs)
            Overloaded function.

            1. first_arrival(self: neworder.MonteCarlo, lambda: numpy.ndarray[float64], dt: float, n: int, minval: float) -> numpy.ndarray[float64]


                Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
                with a minimum start time of minval. Sampling uses the Lewis-Shedler "thinning" algorithm
                If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()


            2. first_arrival(self: neworder.MonteCarlo, lambda: numpy.ndarray[float64], dt: float, n: int) -> numpy.ndarray[float64]


                Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
                with no minimum start time. Sampling uses the Lewis-Shedler "thinning" algorithm
                If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()

        hazard(...)
            hazard(*args, **kwargs)
            Overloaded function.

            1. hazard(self: neworder.MonteCarlo, p: float, n: int) -> numpy.ndarray[float64]


                Returns an array of ones (with hazard rate lambda) or zeros of length n


            2. hazard(self: neworder.MonteCarlo, p: numpy.ndarray[float64]) -> numpy.ndarray[float64]


                Returns an array of ones (with hazard rate lambda[i]) or zeros for each element in p

        next_arrival(...)
            next_arrival(*args, **kwargs)
            Overloaded function.

            1. next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[float64], lambda: numpy.ndarray[float64], dt: float, relative: bool, minsep: float) -> numpy.ndarray[float64]


                Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
                with start times given by startingpoints with a minimum offset of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm.
                If the relative flag is True, then lambda[0] corresponds to start time + mingap, not to absolute time
                If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()


            2. next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[float64], lambda: numpy.ndarray[float64], dt: float, relative: bool) -> numpy.ndarray[float64]


                Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
                with start times given by startingpoints. Sampling uses the Lewis-Shedler "thinning" algorithm.
                If the relative flag is True, then lambda[0] corresponds to start time, not to absolute time
                If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()


            3. next_arrival(self: neworder.MonteCarlo, startingpoints: numpy.ndarray[float64], lambda: numpy.ndarray[float64], dt: float) -> numpy.ndarray[float64]


                Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
                with start times given by startingpoints. Sampling uses the Lewis-Shedler "thinning" algorithm.
                If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()

        reset(...)
            reset(self: neworder.MonteCarlo) -> None


            Resets the generator using the original seed.
            Use with care, esp in multi-process models with identical streams

        seed(...)
            seed(self: neworder.MonteCarlo) -> int


            Returns the seed used to initialise the random stream

        stopping(...)
            stopping(*args, **kwargs)
            Overloaded function.

            1. stopping(self: neworder.MonteCarlo, lambda: float, n: int) -> numpy.ndarray[float64]


                Returns an array of stopping times (with hazard rate lambda) of length n


            2. stopping(self: neworder.MonteCarlo, lambda: numpy.ndarray[float64]) -> numpy.ndarray[float64]


                Returns an array of stopping times (with hazard rate lambda[i]) for each element in lambda

        ustream(...)
            ustream(self: neworder.MonteCarlo, n: int) -> numpy.ndarray[float64]


            Returns an array of uniform random [0,1) variates of length n

        ----------------------------------------------------------------------
        Static methods defined here:

        deterministic_identical_stream(...) from builtins.PyCapsule
            deterministic_identical_stream(r: int) -> int


            Returns a deterministic seed (19937). Input argument is ignored

        deterministic_independent_stream(...) from builtins.PyCapsule
            deterministic_independent_stream(r: int) -> int


            Returns a deterministic seed that is a function of the input (19937+r).
            The model uses the MPI rank as the input argument, allowing for differently seeded streams in each process

        nondeterministic_stream(...) from builtins.PyCapsule
            nondeterministic_stream(r: int) -> int


            Returns a random seed from the platform's random_device. Input argument is ignored

        ----------------------------------------------------------------------
        Static methods inherited from pybind11_builtins.pybind11_object:

        __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
            Create and return a new object.  See help(type) for accurate signature.

    class Timeline(pybind11_builtins.pybind11_object)
        Timestepping functionality

        Method resolution order:
            Timeline
            pybind11_builtins.pybind11_object
            builtins.object

        Methods defined here:

        __init__(...)
            __init__(self: neworder.Timeline, start: float, end: float, checkpoints: List[int]) -> None


            Constructs a timeline from start to end, with the checkpoints given by a non-empty list of ascending integers.
            The total number of steps and the step size is determined by the final checkpoint value

        __repr__(...)
            __repr__(self: neworder.Timeline) -> str


            Prints a human-readable representation of the timeline

        at_checkpoint(...)
            at_checkpoint(self: neworder.Timeline) -> bool


            Returns True if the current step is a checkpoint

        at_end(...)
            at_end(self: neworder.Timeline) -> bool


            Returns True if the current step is the end of the timeline

        dt(...)
            dt(self: neworder.Timeline) -> float


            Returns the step size size of the timeline

        end(...)
            end(self: neworder.Timeline) -> float


            Returns the time of the end of the timeline

        index(...)
            index(self: neworder.Timeline) -> int


            Returns the index of the current step in the timeline

        nsteps(...)
            nsteps(self: neworder.Timeline) -> int


            Returns the number of steps in the timeline

        start(...)
            start(self: neworder.Timeline) -> float


            Returns the time of the start of the timeline

        time(...)
            time(self: neworder.Timeline) -> float


            Returns the time of the current step in the timeline

        ----------------------------------------------------------------------
        Static methods defined here:

        null(...) from builtins.PyCapsule
            null() -> neworder.Timeline


            Returns a "null" timeline, where the start and end times are zero and there is a single step and checkpoint
            Useful for continuous-time models with no explicit (discrete) timeline

        ----------------------------------------------------------------------
        Static methods inherited from pybind11_builtins.pybind11_object:

        __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
            Create and return a new object.  See help(type) for accurate signature.

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
    /mnt/data/home/az/dev/neworder/.venv-focal/lib/python3.8/site-packages/neworder-0.0.6-py3.8-linux-x86_64.egg/neworder.cpython-38-x86_64-linux-gnu.so


```

## `neworder.mpi` module

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

## `neworder.time` module

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

## `neworder.stats` module

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

## `neworder.df` module

```text
Help on module df in neworder:

NAME
    df - Direct manipulations of dataframes

FUNCTIONS
    testfunc(...) method of builtins.PyCapsule instance
        testfunc(model: neworder.Model, df: object, colname: str) -> None


        Test function for direct dataframe manipulation. Results may vary. Do not use.

    transition(...) method of builtins.PyCapsule instance
        transition(model: neworder.Model, categories: numpy.ndarray[int64], transition_matrix: numpy.ndarray[float64], df: object, colname: str) -> None


        Randomly changes categorical data in a dataframe, according to supplied transition probabilities.
        Args:
            model: The model (for access to the MonteCarlo engine).
            categories: The set of possible categories
            transition_matrix: The probabilities of transitions between categories
            df: The dataframe, which is modified in-place
            colname: The name of the column in the dataframe

    unique_index(...) method of builtins.PyCapsule instance
        unique_index(n: int) -> numpy.ndarray[int64]


        Generates an array of unique values, even across multiple processes, that can be used to uniquely index multiple dataframes.
        Args:
            n: The number of required index values.
        Returns:
            The unique index values

FILE
    (built-in)


```
