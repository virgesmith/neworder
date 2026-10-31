"""
Timelines. A timeline is an iterator of timesteps, which a model run takes in turn. The factories here return
generators, which end for a fixed timeline and run forever for an open-ended one. Any other iterable of timesteps, e.g.
a list or a generator function, works too.
"""

from collections.abc import Iterator, Sequence
from datetime import date
from itertools import count, pairwise
from math import isclose
from numbers import Integral
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from dateutil.relativedelta import relativedelta

import neworder as no


class Timestep(NamedTuple):
    """
    A step of a timeline: its index (from 1), its start and end in the timeline's native type (e.g. int, float or
    date), and its length as a float, for computations such as hazard rates.
    """

    index: int
    start: Any
    end: Any
    dt: float


def null() -> Iterator[Timestep]:
    """
    A single timestep with undefined start, end and length, for continuous-time models with no explicit (discrete)
    timeline.
    """
    yield Timestep(1, None, None, no.time.NEVER)


def linear(start: int | np.integer = 0, *, end: int | np.integer | None = None) -> Iterator[Timestep]:
    """
    Integer unit steps from start to end, or open-ended if end is not specified. Use `numeric` for non-integer times or
    step sizes.
    """
    for name, value in [("start", start), ("end", end)]:
        if value is not None and (not isinstance(value, Integral) or isinstance(value, bool | np.bool_)):
            raise TypeError(f"linear requires an integer {name}, got {value!r}. Use numeric for non-integer times")
    # negative timesteps are disallowed as MC functions will misbehave with dt<0
    if end is not None and end <= start:
        raise ValueError(f"end time ({end}) must be after the start time ({start})")
    t0 = int(start)
    return (Timestep(i, t0 + i - 1, t0 + i, 1.0) for i in (count(1) if end is None else range(1, int(end) - t0 + 1)))


def _strictly_increasing(times: Sequence[float] | npt.NDArray[np.floating]) -> list[float]:
    points = [float(t) for t in times]
    if len(points) < 2:
        raise ValueError("timeline must have at least 2 points")
    for i, (t0, t1) in enumerate(pairwise(points), start=1):
        if not t1 > t0:  # also rejects NaN
            raise ValueError(f"invalid timeline: time at index {i} ({t1}) is not strictly greater than previous ({t0})")
    return points


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


def numeric(
    times: Sequence[float] | npt.NDArray[np.floating] | None = None,
    *,
    start: float | None = None,
    step: float | None = None,
    end: float | None = None,
) -> Iterator[Timestep]:
    """
    Non-integer steps, either between explicitly specified time points, which must be strictly increasing and allow
    for unequally-spaced steps, or equally-spaced from start with the given step size, to end (which must be a whole
    number of steps away) or open-ended if end is not specified.
    """
    if times is not None:
        if start is not None or step is not None or end is not None:
            raise TypeError("specify either times, or start and step (and optionally end), but not both")
        points = _strictly_increasing(times)
        return (Timestep(i, t0, t1, t1 - t0) for i, (t0, t1) in enumerate(pairwise(points), start=1))
    if start is None or step is None:
        raise TypeError("specify either times, or start and step (and optionally end)")
    t0, dt = float(start), float(step)
    nsteps = _equal_steps(t0, dt, end)

    def point(i: int) -> float:
        # computed from the start rather than accumulating steps, to avoid rounding errors
        return float(end) if i == nsteps and end is not None else t0 + i * dt

    return (Timestep(i, point(i - 1), point(i), dt) for i in (count(1) if nsteps is None else range(1, nsteps + 1)))


def calendar(*, start: date, step: relativedelta, end: date | None = None) -> Iterator[Timestep]:
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

    def timesteps() -> Iterator[Timestep]:
        for i in count(1):
            # computed from the start rather than accumulating steps, which would lose the day of month
            t0, t1 = start + (i - 1) * step, start + i * step
            if end is not None:
                t1 = min(t1, end)
            yield Timestep(i, t0, t1, (t1 - t0).days / 365)
            if t1 == end:
                return

    return timesteps()
