"""
Timelines. A timeline is an iterator of timesteps, which a model run takes in turn. The factories here return
generators, which end for a fixed timeline and run forever for an open-ended one. Any other iterable of timesteps, e.g.
a list or a generator function, works too.
"""

from collections.abc import Callable, Iterable, Iterator
from datetime import date
from itertools import count, pairwise
from math import isclose
from numbers import Integral
from typing import Any, NamedTuple

import numpy as np
from dateutil.relativedelta import relativedelta

import neworder as no

__all__ = ["Timestep", "calendar", "linear", "null", "numeric", "sequence"]


class Timestep(NamedTuple):
    """
    A step of a timeline. Timelines are iterables of these, and `Model.timestep` is the current one.
    """

    index: int
    """The position of the step in the timeline, counting from 1."""
    start: Any
    """The start of the step, in the timeline's native type (e.g. int, float or date)."""
    end: Any
    """The end of the step, in the timeline's native type."""
    dt: float
    """The length of the step as a float (e.g. in years for calendar), for computations such as hazard rates."""


def null() -> Iterator[Timestep]:
    """
    A single timestep with undefined start, end and length, for continuous-time models with no explicit (discrete)
    timeline.
    """
    yield Timestep(1, None, None, no.time.NEVER)


def _difference(start: Any, end: Any) -> Any:
    return end - start


def sequence(points: Iterable[Any], *, dt: Callable[[Any, Any], float] = _difference) -> Iterator[Timestep]:
    """
    Steps between consecutive points of any type, e.g. unequally-spaced times or dates, so n points give n-1 steps. The
    points can be any iterable, including an open-ended iterator. dt is a function computing the length of a step from
    its start and end, which defaults to their difference and so must be supplied for points that aren't numbers, e.g.
    `lambda start, end: (end - start).days / 365` for dates. Raises a ValueError, when the step is reached, if its
    length isn't positive.
    """
    for i, (t0, t1) in enumerate(pairwise(points), start=1):
        length = float(dt(t0, t1))
        # negative timesteps are disallowed as MC functions will misbehave with dt<0. Also rejects NaN
        if not length > 0.0:
            raise ValueError(f"invalid timeline: step {i} from {t0!r} to {t1!r} has non-positive length ({length})")
        yield Timestep(i, t0, t1, length)


# the factories below validate their arguments up front, then are a sequence of lazily generated points


def linear(start: int | np.integer = 0, *, end: int | np.integer | None = None) -> Iterator[Timestep]:
    """
    Integer unit steps from start to end, or open-ended if end is not specified. Use `numeric` for non-integer times or
    step sizes.
    """
    for name, value in [("start", start), ("end", end)]:
        if value is not None and (not isinstance(value, Integral) or isinstance(value, bool | np.bool_)):
            raise TypeError(f"linear requires an integer {name}, got {value!r}. Use numeric for non-integer times")
    if end is not None and end <= start:
        raise ValueError(f"end time ({end}) must be after the start time ({start})")
    t0 = int(start)
    return sequence(count(t0) if end is None else range(t0, int(end) + 1), dt=lambda _start, _end: 1.0)


def _equal_steps(start: float, step: float, end: float | None) -> int | None:
    if step <= 0.0:
        raise ValueError(f"timeline must have a positive step size, got {step}")
    if end is None:
        return None
    if end <= start:
        raise ValueError(f"end time ({end}) must be after the start time ({start})")
    # fix the number of steps up front, as start + n * step can fall just short of end
    nsteps = round((end - start) / step)
    # absolute tolerance is needed when end is zero, where a relative one is meaningless
    if not isclose(start + nsteps * step, end, abs_tol=1e-9 * step):
        raise ValueError(f"end - start ({end - start}) is not a whole number of steps ({step})")
    return nsteps


def numeric(start: float = 0.0, *, step: float, end: float | None = None) -> Iterator[Timestep]:
    """
    Equally-spaced non-integer steps from start with the given step size, to end (which must be a whole number of steps
    away) or open-ended if end is not specified. For unequally-spaced steps, use `sequence`.
    """
    t0, dt = float(start), float(step)
    nsteps = _equal_steps(t0, dt, end)

    def point(i: int) -> float:
        # computed from the start rather than accumulating steps, to avoid rounding errors
        return float(end) if i == nsteps and end is not None else t0 + i * dt

    # dt is the step size itself, as the difference of the points is subject to rounding
    return sequence(map(point, count() if nsteps is None else range(nsteps + 1)), dt=lambda _start, _end: dt)


def calendar(start: date, *, step: relativedelta, end: date | None = None) -> Iterator[Timestep]:
    """
    Calendar dates, in steps of a `relativedelta` (e.g. days, months or years) from start to end, or open-ended if end
    is not specified. Monthly steps preserve the day of the month where possible. If end isn't a whole number of steps
    from the start, the final step is shortened to end at it. dt is the length of the step in years, on an ACT/365
    basis (so a step spanning a leap day is 366/365 years long).
    """
    if end is not None and end <= start:
        raise ValueError("end date must be after start date")
    if start + step <= start:
        raise ValueError("step must be forward in time")

    def points() -> Iterator[date]:
        for i in count():
            # computed from the start rather than accumulating steps, which would lose the day of month
            t = start + i * step
            if end is not None and t >= end:
                yield end
                return
            yield t

    return sequence(points(), dt=lambda start, end: (end - start).days / 365)
