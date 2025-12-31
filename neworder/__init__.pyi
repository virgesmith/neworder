"""

A dynamic microsimulation framework";
"""

from __future__ import annotations

import collections.abc
import datetime
import types
import typing

import numpy
import numpy.typing

from . import df, mpi, stats, time
from .domain import Domain, Edge, Space, StateGrid
from .mc import as_np

__all__: list[str] = [
    "CalendarTimeline",
    "LinearTimeline",
    "Model",
    "MonteCarlo",
    "NoTimeline",
    "NumericTimeline",
    "Timeline",
    "checked",
    "df",
    "freethreaded",
    "log",
    "mpi",
    "run",
    "stats",
    "thread_id",
    "time",
    "verbose",
    "as_np",
    "Edge",
    "Domain",
    "Space",
    "StateGrid",
]

class CalendarTimeline(Timeline):
    """

    A calendar-based timeline
    """
    @typing.overload
    def __init__(self, start: datetime.datetime, end: datetime.datetime, step: typing.SupportsInt, unit: str) -> None:
        """
        Constructs a calendar-based timeline, given start and end dates, an increment specified as a multiple of days, months or years
        """
    @typing.overload
    def __init__(self, start: datetime.datetime, step: typing.SupportsInt, unit: str) -> None:
        """
        Constructs an open-ended calendar-based timeline, given a start date and an increment specified as a multiple of days, months or years.
         NB the model will run until the Model.halt() method is explicitly called (from inside the step() method). Note also that nsteps() will
         return -1 for timelines constructed this way
        """

class LinearTimeline(Timeline):
    """

    An equally-spaced non-calendar timeline .
    """
    @typing.overload
    def __init__(self, start: typing.SupportsFloat, end: typing.SupportsFloat, nsteps: typing.SupportsInt) -> None:
        """
        Constructs a timeline from start to end, with the given number of steps.
        """
    @typing.overload
    def __init__(self, start: typing.SupportsFloat, step: typing.SupportsFloat) -> None:
        """
        Constructs an open-ended timeline give a start value and a step size. NB the model will run until the Model.halt() method is explicitly called
        (from inside the step() method). Note also that nsteps() will return -1 for timelines constructed this way
        """

class Model:
    """

    The base model class from which all neworder models should be subclassed
    """
    class RunState:
        """
        Members:

          NOT_STARTED

          RUNNING

          HALTED

          COMPLETED
        """

        COMPLETED: typing.ClassVar[Model.RunState]  # value = <RunState.COMPLETED: 3>
        HALTED: typing.ClassVar[Model.RunState]  # value = <RunState.HALTED: 2>
        NOT_STARTED: typing.ClassVar[Model.RunState]  # value = <RunState.NOT_STARTED: 0>
        RUNNING: typing.ClassVar[Model.RunState]  # value = <RunState.RUNNING: 1>
        __members__: typing.ClassVar[
            dict[str, Model.RunState]
        ]  # value = {'NOT_STARTED': <RunState.NOT_STARTED: 0>, 'RUNNING': <RunState.RUNNING: 1>, 'HALTED': <RunState.HALTED: 2>, 'COMPLETED': <RunState.COMPLETED: 3>}
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        def __init__(self, value: typing.SupportsInt) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: typing.SupportsInt) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    COMPLETED: typing.ClassVar[Model.RunState]  # value = <RunState.COMPLETED: 3>
    HALTED: typing.ClassVar[Model.RunState]  # value = <RunState.HALTED: 2>
    NOT_STARTED: typing.ClassVar[Model.RunState]  # value = <RunState.NOT_STARTED: 0>
    RUNNING: typing.ClassVar[Model.RunState]  # value = <RunState.RUNNING: 1>
    def __init__(self, timeline: Timeline, seeder: collections.abc.Callable = ...) -> None:
        """
        Constructs a model object with a timeline and (optionally) a seeder function for the random stream(s)
        """
    def check(self) -> bool:
        """
        User-overridable method used to check internal state at each timestep.
        Default behaviour is to simply return True.
        Returning False will halt the model run.
        This function should not be called directly, it is used by the Model.run() function

        Returns:
            True if checks are ok, False otherwise.
        """
    def finalise(self) -> None:
        """
        User-overridable function for custom processing after the final step in the model run.
        Default behaviour does nothing. This function does not need to be called directly, it is called by the Model.run() function
        """
    def halt(self) -> None:
        """
        Signal to the model to stop execution gracefully at the end of the current timestep, e.g. if some convergence criterion has been met,
        or input is required from an upstream model. The model can be subsequently resumed by calling the run() function.
        For trapping exceptional/error conditions, prefer to raise an exception, or return False from the Model.check() function
        """
    def modify(self) -> None:
        """
        User-overridable method used to modify state in a per-process basis for multiprocess model runs.
        Default behaviour is to do nothing.
        This function should not be called directly, it is used by the Model.run() function
        """
    def step(self) -> None:
        """
        User-implemented method used to advance state of a model.
        Default behaviour raises NotImplementedError.
        This function should not be called directly, it is used by the Model.run() function
        """
    @property
    def mc(self) -> MonteCarlo:
        """
        The model's Monte-Carlo engine
        """
    @property
    def run_state(self) -> RunState:
        """
        The model's current state - one of:
            NOT_STARTED: model has not been run
            RUNNING: model is in progress
            HALTED: model has been explicitly halted by calling its halt() method
            COMPLETED: model has run to the end of its timeline
        """
    @property
    def timeline(self) -> Timeline:
        """
        The model's timeline object
        """

class MonteCarlo:
    """

    The model's Monte-Carlo engine with configurable options for parallel execution
    """
    @staticmethod
    def deterministic_identical_stream() -> int:
        """
        Returns a deterministic seed (19937).
        """
    @staticmethod
    def deterministic_independent_stream() -> int:
        """
        Returns a deterministic seed that is a function of the process rank (19937+r).
        Each process will have independent streams. Threads within the process will have identical streams.
        """
    @staticmethod
    def nondeterministic_stream() -> int:
        """
        Returns a random seed from the platform's random_device.
        """
    def __repr__(self) -> str:
        """
        Prints a human-readable representation of the MonteCarlo engine
        """
    def arrivals(
        self,
        lambda_: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
        dt: typing.SupportsFloat,
        n: typing.SupportsInt,
        mingap: typing.SupportsFloat,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """
        Returns an array of n arrays of multiple arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
        with a minimum separation between events of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm
        The final value of lambda must be zero, and thus arrivals don't always occur, indicated by a value of neworder.time.never()
        The inner dimension of the returned 2d array is governed by the the maximum number of arrivals sampled, and will thus vary
        """
    def counts(
        self, lambda_: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], dt: typing.SupportsFloat
    ) -> numpy.typing.NDArray[numpy.int64]:
        """
        Returns an array of simulated arrival counts (within time dt) for each intensity in lambda
        """
    def first_arrival(
        self,
        lambda_: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
        dt: typing.SupportsFloat,
        n: typing.SupportsInt,
        minval: typing.SupportsFloat = 0.0,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """
        Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
        with an optional minimum start time of minval. Sampling uses the Lewis-Shedler "thinning" algorithm
        If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
        """
    @typing.overload
    def hazard(self, p: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """
        Returns an array of ones (with hazard rate lambda) or zeros of length n
        """
    @typing.overload
    def hazard(self, p: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """
        Returns an array of ones (with hazard rate lambda[i]) or zeros for each element in p
        """
    def init_bitgen(self, arg0: types.CapsuleType) -> None:  # type: ignore[name-defined]
        """
        internal helper function used by as_np
        """
    def next_arrival(
        self,
        startingpoints: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
        lambda_: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
        dt: typing.SupportsFloat,
        relative: bool = False,
        minsep: typing.SupportsFloat = 0.0,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """
        Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
        with start times given by startingpoints with a minimum offset of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm.
        If the relative flag is True, then lambda[0] corresponds to start time + mingap, not to absolute time
        If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
        """
    def raw(self) -> int:
        """
        Returns a random 64-bit unsigned integer. Useful for seeding other generators.
        """
    def reset(self) -> None:
        """
        Resets the generator using the original seed.
        Use with care, esp in multi-process models with identical streams
        """
    def sample(
        self, n: typing.SupportsInt, cat_weights: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    ) -> numpy.typing.NDArray[numpy.int64]:
        """
        Returns an array of length n containing randomly sampled categorical values, weighted according to cat_weights
        """
    def seed(self) -> int:
        """
        Returns the seed used to initialise the random stream
        """
    def state(self) -> int:
        """
        Returns a hash of the internal state of the generator. Avoids the extra complexity of tranmitting variable-length strings over MPI.
        """
    @typing.overload
    def stopping(self, lambda_: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """
        Returns an array of stopping times (with hazard rate lambda) of length n
        """
    @typing.overload
    def stopping(
        self, lambda_: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """
        Returns an array of stopping times (with hazard rate lambda[i]) for each element in lambda
        """
    def ustream(self, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """
        Returns an array of uniform random [0,1) variates of length n
        """

class NoTimeline(Timeline):
    """

    An arbitrary one step timeline, for continuous-time models with no explicit (discrete) timeline
    """
    def __init__(self) -> None:
        """
        Constructs an arbitrary one step timeline, where the start and end times are undefined and there is a single step of size zero. Useful for continuous-time models
        """

class NumericTimeline(Timeline):
    """

    An custom non-calendar timeline where the user explicitly specifies the time points, which must be monotonically increasing.
    """
    def __init__(self, times: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Constructs a timeline from an array of time points.
        """

class Timeline:
    def __init__(self) -> None: ...
    def __repr__(self) -> str:
        """
        Prints a human-readable representation of the timeline object
        """
    @property
    def at_end(self) -> bool:
        """
        Returns True if the current step is the end of the timeline
        """
    @property
    def dt(self) -> float:
        """
        Returns the step size size of the timeline
        """
    @property
    def end(self) -> typing.Any:
        """
        Returns the time of the end of the timeline
        """
    @property
    def index(self) -> int:
        """
        Returns the index of the current step in the timeline
        """
    @property
    def nsteps(self) -> int:
        """
        Returns the number of steps in the timeline (or -1 if open-ended)
        """
    @property
    def start(self) -> typing.Any:
        """
        Returns the time of the start of the timeline
        """
    @property
    def time(self) -> typing.Any:
        """
        Returns the time of the current step in the timeline
        """

def checked(checked: bool = True) -> None:
    """
    Sets the checked flag, which determines whether the model runs checks during execution
    """

def freethreaded() -> bool:
    """
    Returns whether neworder was *built* with free-threading support (i.e. no GIL).

    Note: Other packages (e.g. pandas) may re-enable the GIL at runtime. To check, use `sys._is_gil_enabled()`. To force
    free-threading (at your own risk), use PYTHON_GIL=0 or -Xgil=0.
    """

def log(*args: typing.Any) -> None:
    """
    The logging function. Prints *args to the console, annotated with process/thread information
    """

def run(model: Model) -> bool:
    """
    Runs the model. If the model has previously run it will resume from the point at which it was given the "halt" instruction. This is useful
    for external processing of model data, and/or feedback from external sources. If the model has already reached the end of the timeline, this
    function will have no effect. To re-run the model from the start, you must construct a new model object.
    Returns:
        True if model succeeded, False otherwise
    """

def thread_id() -> int:
    """
    Returns a unique thread id - equivalent to threading.get_native_id(). Use for control flow with extreme caution -
    order of thread initialisation cannot be guaranteed, and the python runtime may reuse completed threads.
    """

def verbose(verbose: bool = True) -> None:
    """
    Sets the verbose flag, which toggles detailed runtime logs
    """
