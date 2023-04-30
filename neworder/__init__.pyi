"""
A dynamic microsimulation framework";
"""
from __future__ import annotations
import typing
import datetime
import numpy as np
import numpy.typing as npt

import df  # type: ignore
import mpi # type: ignore
from . import time
import stats # type: ignore
from .domain import *

FloatArray1d = NPFloatArray | list[float]
NPIntArray = npt.NDArray[np.int64]
IntArray1d = NPIntArray | list[int]


__all__ = [
    "CalendarTimeline",
    "LinearTimeline",
    "Model",
    "MonteCarlo",
    "NoTimeline",
    "NumericTimeline",
    "Timeline",
    "checked",
    "df",
    "log",
    "mpi",
    "run",
    "stats",
    "time",
    "verbose"
]


class Timeline():
    def at_end(self) -> bool:
        """
        Returns True if the current step is the end of the timeline
        """
    def dt(self) -> float:
        """
        Returns the step size size of the timeline
        """
    def end(self) -> Any:
        """
        Returns the time of the end of the timeline
        """
    def index(self) -> int:
        """
        Returns the index of the current step in the timeline
        """
    def nsteps(self) -> int:
        """
        Returns the number of steps in the timeline (or -1 if open-ended)
        """
    def start(self) -> Any:
        """
        Returns the time of the start of the timeline
        """
    def time(self) -> Any:
        """
        Returns the time of the current step in the timeline
        """
class LinearTimeline(Timeline):
    """
    An equally-spaced non-calendar timeline .
    """
    @typing.overload
    def __init__(self, start: float, end: float, nsteps: int) -> None:
        """
        Constructs a timeline from start to end, with the given number of steps.
        """
    @typing.overload
    def __init__(self, start: float, step: float) -> None:
        """
        Constructs an open-ended timeline give a start value and a step size. NB the model will run until the Model.halt() method is explicitly called
        (from inside the step() method). Note also that nsteps() will return -1 for timelines constructed this way
        """
    def __repr__(self) -> str:
        """
        Prints a human-readable representation of the timeline
        """
    def at_end(self) -> bool:
        """
        Returns True if the current step is the end of the timeline
        """
    def dt(self) -> float:
        """
        Returns the step size size of the timeline
        """
    def end(self) -> object:
        """
        Returns the time of the end of the timeline
        """
    def index(self) -> int:
        """
        Returns the index of the current step in the timeline
        """
    def nsteps(self) -> int:
        """
        Returns the number of steps in the timeline (or -1 if open-ended)
        """
    def start(self) -> object:
        """
        Returns the time of the start of the timeline
        """
    def time(self) -> object:
        """
        Returns the time of the current step in the timeline
        """
    pass
class Model():
    """
    The base model class from which all neworder models should be subclassed
    """
    def __init__(self, timeline: Timeline, seeder: function) -> None:
        """
        Constructs a model object with a timeline and a seeder function
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
    def modify(self, r: int) -> None:
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

        :type: no::MonteCarlo
        """
    @property
    def timeline(self) -> Timeline:
        """
            The model's timeline object

        :type: Timeline
        """
    pass
class MonteCarlo():
    """
    The model's Monte-Carlo engine with configurable options for parallel execution
    """
    def __repr__(self) -> str:
        """
        Prints a human-readable representation of the MonteCarlo engine
        """
    def arrivals(self, lambda_: FloatArray1d, dt: float, n: int, mingap: float) -> NPFloatArray:
        """
        Returns an array of n arrays of multiple arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
        with a minimum separation between events of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm
        The final value of lambda must be zero, and thus arrivals don't always occur, indicated by a value of neworder.time.never()
        The inner dimension of the returned 2d array is governed by the the maximum number of arrivals sampled, and will thus vary
        """
    def counts(self, lambda_: FloatArray1d, dt: float) -> NPIntArray:
        """
        Returns an array of simulated arrival counts (within time dt) for each intensity in lambda
        """
    @staticmethod
    def deterministic_identical_stream(r: int) -> int:
        """
        Returns a deterministic seed (19937). Input argument is ignored
        """
    @staticmethod
    def deterministic_independent_stream(r: int) -> int:
        """
        Returns a deterministic seed that is a function of the input (19937+r).
        The model uses the MPI rank as the input argument, allowing for differently seeded streams in each process
        """
    @typing.overload
    def first_arrival(self, lambda_: FloatArray1d, dt: float, n: int) -> FloatArray1d:
        """
        Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
        with a minimum start time of minval. Sampling uses the Lewis-Shedler "thinning" algorithm
        If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
        """
    @typing.overload
    def first_arrival(self, lambda_: FloatArray1d, dt: float, n: int, minval: float) -> FloatArray1d:
        """
        Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
        with no minimum start time. Sampling uses the Lewis-Shedler "thinning" algorithm
        If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
        """
    @typing.overload
    def hazard(self, p: float, n: int) -> NPIntArray:
        """
        Returns an array of ones (with hazard rate lambda) or zeros of length n
        """
    @typing.overload
    def hazard(self, p: FloatArray1d) -> NPIntArray:
        """
        Returns an array of ones (with hazard rate lambda[i]) or zeros for each element in p
        """
    @typing.overload
    def next_arrival(self, startingpoints: FloatArray1d, lambda_: FloatArray1d, dt: float) -> FloatArray1d:
        """
        Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
        with start times given by startingpoints with a minimum offset of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm.
        If the relative flag is True, then lambda[0] corresponds to start time + mingap, not to absolute time
        If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
        """
    @typing.overload
    def next_arrival(self, startingpoints: FloatArray1d, lambda_: FloatArray1d, dt: float, relative: bool) -> FloatArray1d:
        """
        Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
        with start times given by startingpoints. Sampling uses the Lewis-Shedler "thinning" algorithm.
        If the relative flag is True, then lambda[0] corresponds to start time, not to absolute time
        If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
        """
    @typing.overload
    def next_arrival(self, startingpoints: FloatArray1d, lambda_: FloatArray1d, dt: float, relative: bool, minsep: float) -> FloatArray1d:
        """
        Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
        with start times given by startingpoints. Sampling uses the Lewis-Shedler "thinning" algorithm.
        If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
        """
    @staticmethod
    def nondeterministic_stream(r: int) -> int:
        """
        Returns a random seed from the platform's random_device. Input argument is ignored
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
    def sample(self, n: int, cat_weights: NPFloatArray) -> NPIntArray:
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
    def stopping(self, lambda_: float, n: int) -> NPFloatArray:
        """
        Returns an array of stopping times (with hazard rate lambda) of length n
        """
    @typing.overload
    def stopping(self, lambda_: NPFloatArray) -> NPFloatArray:
        """
        Returns an array of stopping times (with hazard rate lambda[i]) for each element in lambda
        """
    def ustream(self, n: int) -> NPFloatArray:
        """
        Returns an array of uniform random [0,1) variates of length n
        """
    pass
class NoTimeline(Timeline):
    """
    An arbitrary one step timeline, for continuous-time models with no explicit (discrete) timeline
    """
    def __init__(self) -> None:
        """
        Constructs an arbitrary one step timeline, where the start and end times are undefined and there is a single step of size zero. Useful for continuous-time models
        """
    def __repr__(self) -> str:
        """
        Prints a human-readable representation of the timeline
        """
    def at_end(self) -> bool:
        """
        Returns True if the current step is the end of the timeline
        """
    def dt(self) -> float:
        """
        Returns the step size size of the timeline
        """
    def end(self) -> object:
        """
        Returns the time of the end of the timeline
        """
    def index(self) -> int:
        """
        Returns the index of the current step in the timeline
        """
    def nsteps(self) -> int:
        """
        Returns the number of steps in the timeline (or -1 if open-ended)
        """
    def start(self) -> object:
        """
        Returns the time of the start of the timeline
        """
    def time(self) -> object:
        """
        Returns the time of the current step in the timeline
        """
    pass
class NumericTimeline(Timeline):
    """
    An custom non-calendar timeline where the user explicitly specifies the time points, which must be monotonically increasing.
    """
    def __init__(self, times: typing.List[float]) -> None:
        """
        Constructs a timeline from an array of time points.
        """
    def __repr__(self) -> str:
        """
        Prints a human-readable representation of the timeline
        """
    def at_end(self) -> bool:
        """
        Returns True if the current step is the end of the timeline
        """
    def dt(self) -> float:
        """
        Returns the step size size of the timeline
        """
    def end(self) -> object:
        """
        Returns the time of the end of the timeline
        """
    def index(self) -> int:
        """
        Returns the index of the current step in the timeline
        """
    def nsteps(self) -> int:
        """
        Returns the number of steps in the timeline (or -1 if open-ended)
        """
    def start(self) -> object:
        """
        Returns the time of the start of the timeline
        """
    def time(self) -> object:
        """
        Returns the time of the current step in the timeline
        """
    pass
class CalendarTimeline(Timeline):
    """
    A calendar-based timeline
    """
    @typing.overload
    def __init__(self, start: datetime.date | datetime.datetime, end: datetime.date | datetime.datetime, step: int, unit: str) -> None:
        """
        Constructs a calendar-based timeline, given start and end dates, an increment specified as a multiple of days, months or years
        """
    @typing.overload
    def __init__(self, start: datetime.date | datetime.datetime, step: int, unit: str) -> None:
        """
        Constructs an open-ended calendar-based timeline, given a start date and an increment specified as a multiple of days, months or years.
         NB the model will run until the Model.halt() method is explicitly called (from inside the step() method). Note also that nsteps() will
         return -1 for timelines constructed this way
        """
    def __repr__(self) -> str:
        """
        Prints a human-readable representation of the timeline
        """
    def at_end(self) -> bool:
        """
        Returns True if the current step is the end of the timeline
        """
    def dt(self) -> float:
        """
        Returns the step size size of the timeline
        """
    def end(self) -> object:
        """
        Returns the time of the end of the timeline
        """
    def index(self) -> int:
        """
        Returns the index of the current step in the timeline
        """
    def nsteps(self) -> int:
        """
        Returns the number of steps in the timeline (or -1 if open-ended)
        """
    def start(self) -> object:
        """
        Returns the time of the start of the timeline
        """
    def time(self) -> object:
        """
        Returns the time of the current step in the timeline
        """
    pass
def checked(checked: bool = True) -> None:
    """
    Sets the checked flag, which determines whether the model runs checks during execution
    """
def log(obj: object) -> None:
    """
    The logging function. Prints obj to the console, annotated with process information
    """
def run(model: object) -> bool:
    """
    Runs the model. If the model has previously run it will resume from the point at which it was given the "halt" instruction. This is useful
    for external processing of model data, and/or feedback from external sources. If the model has already reached the end of the timeline, this
    function will have no effect. To re-run the model from the start, you must construct a new model object.
    Returns:
        True if model succeeded, False otherwise
    """
def verbose(verbose: bool = True) -> None:
    """
    Sets the verbose flag, which toggles detailed runtime logs
    """

def as_np(mc: MonteCarlo) -> np.random.Generator:
  """
  Returns an adapter enabling the MonteCarlo object to be used with numpy random functionality
  """
