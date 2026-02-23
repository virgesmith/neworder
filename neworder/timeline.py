from datetime import date

from dateutil.relativedelta import relativedelta

import neworder as no


class CalendarTimeline(no.Timeline):
    """
    A timeline representing calendar days. At any given step, `time` returns the date at the *start* of the step
    For monthly timesteps, preserves day of month.
    Numeric step size is computed using an ACT/365 basis and may vary depending on choice of step
    """

    def __init__(self, start: date, step: relativedelta, *, end: date | None = None) -> None:
        super().__init__()

        if end and end <= start:
            raise ValueError("end date must be after start date")
        if start + step <= start:
            raise ValueError("step must be forward in time")

        self._start = start
        self._end = end
        self._current = start
        self._step = step

    def _next(self) -> date:
        if self._end and self._current >= self._end:
            raise StopIteration()
        # steps must be relative to start date, not current date in order to preserve day of month when >= 28
        self._current = self._start + self.index * self._step
        return self._current

    @property
    def start(self) -> date:
        return self._start

    @property
    def end(self) -> date | float:
        return self._end or no.time.FAR_FUTURE

    @property
    def time(self) -> date:
        return self._current

    @property
    def dt(self) -> float:
        """Returns year fraction on ACT/365 basis, or 0 if the timeline has ended"""
        if self.at_end:
            return 0.0
        next: date = self._start + (self.index + 1) * self._step
        return (next - self._current).days / 365

    @property
    def at_end(self) -> bool:
        return self._end is not None and self._current >= self._end
