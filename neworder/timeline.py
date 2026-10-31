from collections.abc import Sequence
from datetime import date
from itertools import pairwise
from math import isclose
from numbers import Integral
from typing import overload

import numpy as np
import numpy.typing as npt
from dateutil.relativedelta import relativedelta

import neworder as no


class _SteppedTimeline[T](no.Timeline):
    """
    Common iteration logic for timelines defined by a sequence of time points. Before iteration the timeline is at
    point 0, its start, and each call to `__next__` advances to the next point, returning the step just taken as a
    (start, end) tuple of native values. Subclasses implement `_point`, which returns the native value of a point, and
    (unless the native type is float) `_numeric`, which returns it as a float.
    """

    def __init__(self, nsteps: int | None) -> None:
        super().__init__()
        # None means open-ended
        self._nsteps = nsteps
        self._index = 0

    def _point(self, i: int) -> T:
        raise NotImplementedError

    def _numeric(self, i: int) -> float:
        return float(self._point(i))  # ty:ignore[invalid-argument-type]

    @property
    def index(self) -> int:
        return self._index

    @property
    def time(self) -> float:
        return self._numeric(self._index)

    @property
    def dt(self) -> float:
        # undefined before the first step, unless the subclass has a fixed step size
        if self._index == 0:
            return no.time.NEVER
        return self._numeric(self._index) - self._numeric(self._index - 1)

    def __next__(self) -> tuple[T, T]:
        if self._nsteps is not None and self._index >= self._nsteps:
            raise StopIteration()
        self._index += 1
        return self._point(self._index - 1), self._point(self._index)


class NoTimeline(_SteppedTimeline[None]):
    """
    An arbitrary one step timeline, for continuous-time models with no explicit (discrete) timeline. The start and end
    of the step are undefined: None natively, and `no.time.NEVER` as time. The step has zero length.
    """

    def __init__(self) -> None:
        super().__init__(1)

    def _point(self, i: int) -> None:  # noqa: ARG002
        return None

    def _numeric(self, i: int) -> float:  # noqa: ARG002
        return no.time.NEVER

    @property
    def dt(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class LinearTimeline(_SteppedTimeline[int]):
    """
    An integer timeline with unit steps, from start to end, or open-ended if end is not specified. Use NumericTimeline
    for non-integer times or step sizes. NB an open-ended model will run until the Model.halt() method is explicitly
    called (from inside the step() method).
    """

    def __init__(self, start: int | np.integer = 0, end: int | np.integer | None = None) -> None:
        for name, value in [("start", start), ("end", end)]:
            if value is not None and not isinstance(value, Integral):
                raise TypeError(
                    f"LinearTimeline requires an integer {name}, got {value!r}. Use NumericTimeline for non-integer times"
                )
        # negative timesteps are disallowed as MC functions will misbehave with dt<0
        if end is not None and end <= start:
            raise ValueError(f"end time ({end}) must be after the start time ({start})")
        super().__init__(None if end is None else int(end - start))
        self._start = int(start)
        self._end = None if end is None else int(end)

    def _point(self, i: int) -> int:
        return self._start + i

    @property
    def dt(self) -> float:
        return 1.0

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} start={self._start} end={self._end}>"


def _strictly_increasing(times: Sequence[float] | npt.NDArray[np.floating]) -> list[float]:
    points = [float(t) for t in times]
    if len(points) < 2:
        raise ValueError("timeline must have at least 2 points")
    for i, (t0, t1) in enumerate(pairwise(points), start=1):
        if t1 <= t0:
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
    if not isclose(start + nsteps * step, end):
        raise ValueError(f"end - start ({end - start}) is not a whole number of steps ({step})")
    return nsteps


class NumericTimeline(_SteppedTimeline[float]):
    """
    A non-integer timeline, either equally-spaced from start with the given step size, to end or open-ended if end is
    not specified, or from explicitly specified time points, which must be strictly increasing and allow for
    unequally-spaced steps (the number of steps is one fewer than the number of points). NB an open-ended model will
    run until the Model.halt() method is explicitly called (from inside the step() method).
    """

    @overload
    def __init__(self, *, start: float, step: float, end: float | None = None) -> None: ...

    @overload
    def __init__(self, times: Sequence[float] | npt.NDArray[np.floating]) -> None: ...

    def __init__(
        self,
        times: Sequence[float] | npt.NDArray[np.floating] | None = None,
        *,
        start: float | None = None,
        step: float | None = None,
        end: float | None = None,
    ) -> None:
        if times is not None:
            if start is not None or step is not None or end is not None:
                raise TypeError("specify either times, or start and step (and optionally end), but not both")
            points = _strictly_increasing(times)
            super().__init__(len(points) - 1)
            self._times: list[float] | None = points
            return
        if start is None or step is None:
            raise TypeError("specify either times, or start and step (and optionally end)")
        super().__init__(_equal_steps(start, step, end))
        self._times = None
        self._start = float(start)
        self._step = float(step)
        self._end = None if end is None else float(end)

    def _point(self, i: int) -> float:
        if self._times is not None:
            return self._times[i]
        if i == self._nsteps and self._end is not None:
            return self._end
        # computed from the start rather than accumulating steps, to avoid rounding errors
        return self._start + i * self._step

    @property
    def dt(self) -> float:
        return super().dt if self._times is not None else self._step

    def __repr__(self) -> str:
        if self._times is not None:
            return f"<{self.__class__.__name__} times={self._times}>"
        return f"<{self.__class__.__name__} start={self._start} end={self._end} step={self._step}>"


class CalendarTimeline(_SteppedTimeline[date]):
    """
    A timeline representing calendar days. For monthly timesteps, preserves day of month. If end is not on the
    timeline, the final step ends after it. As floats, time is in days since the start, and dt in days.
    To preserve day of month for month-based timesteps when day >= 28, explicitly pass day=... to `relativedelta`
    """

    def __init__(self, *, start: date, step: relativedelta, end: date | None = None) -> None:
        if end and end <= start:
            raise ValueError("end date must be after start date")
        if start + step <= start:
            raise ValueError("step must be forward in time")

        nsteps = None
        if end:
            nsteps = 1
            while start + nsteps * step < end:
                nsteps += 1
        super().__init__(nsteps)
        self._start = start
        self._end = end
        self._step = step

    def _point(self, i: int) -> date:
        # computed from the start rather than accumulating steps, which would lose the day of month
        return self._start + i * self._step

    def _numeric(self, i: int) -> float:
        return float((self._point(i) - self._start).days)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} start={self._start} end={self._end} step={self._step}>"
