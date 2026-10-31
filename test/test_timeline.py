from collections.abc import Iterator
from datetime import date
from itertools import islice

import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

import neworder as no


class CustomTimeline(no.Timeline):
    """Halves the remaining distance to 1.0 at each step, never ending"""

    def __init__(self) -> None:
        super().__init__()
        self._index = 0

    @property
    def index(self) -> int:
        return self._index

    @property
    def time(self) -> float:
        return 1.0 - 0.5**self._index

    @property
    def dt(self) -> float:
        return 0.5**self._index

    def __next__(self) -> tuple[float, float]:
        t = self.time
        self._index += 1
        return t, self.time


def states(t: no.Timeline, n: int | None = None) -> list[tuple[int, float, float]]:
    """Takes (at most n) steps, returning (index, time, dt) after each one"""
    return [(t.index, t.time, t.dt) for _ in islice(t, n)]


class RecordingModel(no.Model):
    def __init__(self, timeline: no.Timeline, halt_at: int | None = None) -> None:
        super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)
        self.halt_at = halt_at
        self.modified: tuple[int, float, float] | None = None
        self.steps: list[tuple[int, float, float]] = []
        self.checks: list[tuple[int, float, float]] = []
        self.finalised: tuple[int, float, float] | None = None

    def _state(self) -> tuple[int, float, float]:
        return self.timeline.index, self.timeline.time, self.timeline.dt

    def modify(self) -> None:
        self.modified = self._state()

    def step(self) -> None:
        self.steps.append(self._state())
        if self.timeline.index == self.halt_at:
            self.halt()

    def check(self) -> bool:
        self.checks.append(self._state())
        return True

    def finalise(self) -> None:
        self.finalised = self._state()


def test_abstract_base() -> None:
    class Incomplete(no.Timeline):
        @property
        def index(self) -> int:
            return 0

    t = Incomplete()  # ty:ignore[call-non-callable]
    with pytest.raises(TypeError, match="must implement the abstract member 'time'"):
        _ = t.time
    with pytest.raises(TypeError, match="must implement the abstract member 'dt'"):
        _ = t.dt
    with pytest.raises(TypeError, match="must implement the abstract member '__next__'"):
        next(t)
    with pytest.raises(TypeError, match="must implement the abstract member"):
        no.run(no.Model(t))


def test_custom_timeline() -> None:
    ct = CustomTimeline()
    assert isinstance(ct, Iterator)
    assert iter(ct) is ct
    # default __repr__
    assert str(ct) == "<CustomTimeline index=0>"
    with pytest.raises(AttributeError):
        ct.index = 3  # ty:ignore[invalid-assignment]

    assert states(ct, 3) == [(1, 0.5, 0.5), (2, 0.75, 0.25), (3, 0.875, 0.125)]
    assert str(ct) == "<CustomTimeline index=3>"

    # the C++ base class dispatches to the python implementation
    m = RecordingModel(CustomTimeline(), halt_at=2)
    assert no.run(m)
    assert m.steps == [(1, 0.5, 0.5), (2, 0.75, 0.25)]


def test_iteration() -> None:
    t = no.LinearTimeline(0, 3)
    # at the start before iteration
    assert t.index == 0
    assert t.time == 0.0
    # iteration yields each step as native (start, end) values
    assert list(t) == [(0, 1), (1, 2), (2, 3)]
    assert t.index == 3
    assert t.time == 3.0
    # stays exhausted
    with pytest.raises(StopIteration):
        next(t)
    assert t.index == 3


def test_time() -> None:
    t = -1e10
    assert t > no.time.DISTANT_PAST
    assert t < no.time.FAR_FUTURE
    t = 1e10
    assert t > no.time.DISTANT_PAST
    assert t < no.time.FAR_FUTURE

    # dreams never end
    assert no.time.NEVER != no.time.NEVER
    assert t != no.time.NEVER
    assert not t > no.time.NEVER
    assert not t == no.time.NEVER  # noqa: SIM201
    assert not t <= no.time.NEVER
    # no nay never
    assert not no.time.isnever(t)
    # no nay never no more
    assert no.time.isnever(no.time.NEVER)


def test_time_is_float() -> None:
    timelines: list[no.Timeline] = [
        no.NoTimeline(),
        no.LinearTimeline(0, 2),
        no.NumericTimeline([0, 1, 3]),
        no.NumericTimeline(start=0.0, step=0.5, end=1.0),
        no.CalendarTimeline(start=date(2020, 1, 1), step=relativedelta(months=1), end=date(2020, 3, 1)),
    ]
    for t in timelines:
        for _ in t:
            assert type(t.time) is float
            assert type(t.dt) is float
            # usable with the time constants regardless of the native type
            assert no.time.isnever(t.time) or no.time.DISTANT_PAST < t.time < no.time.FAR_FUTURE


def test_null_timeline() -> None:
    t = no.NoTimeline()
    assert t.index == 0
    assert no.time.isnever(t.time)
    assert t.dt == 0.0
    assert str(t) == "<NoTimeline>"
    with pytest.raises(AttributeError):
        t.index = 3  # ty:ignore[invalid-assignment]
    assert next(t) == (None, None)
    assert t.index == 1
    assert no.time.isnever(t.time)
    assert t.dt == 0.0
    with pytest.raises(StopIteration):
        next(t)


def test_timeline_validation() -> None:
    with pytest.raises(TypeError, match="Use NumericTimeline"):
        no.LinearTimeline(0.0, 1)  # ty:ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="integer end"):
        no.LinearTimeline(0, 0.5)  # ty:ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        no.LinearTimeline(2020, 2021, 1)  # ty:ignore[too-many-positional-arguments]
    with pytest.raises(ValueError):
        no.LinearTimeline(2020, 2020)
    with pytest.raises(ValueError):
        no.LinearTimeline(2020, 2019)
    with pytest.raises(ValueError):
        no.NumericTimeline([2021, 2020])
    with pytest.raises(ValueError, match="index 2"):
        no.NumericTimeline([2019, 2020, 2020])
    with pytest.raises(ValueError):
        no.NumericTimeline([2020])
    with pytest.raises(TypeError):
        no.NumericTimeline([2020, 2021], start=2020, step=1)  # ty:ignore[no-matching-overload]
    with pytest.raises(TypeError):
        no.NumericTimeline(start=2020.0)  # ty:ignore[no-matching-overload]
    with pytest.raises(ValueError):
        no.NumericTimeline(start=2020.0, step=0.0)
    with pytest.raises(ValueError):
        no.NumericTimeline(start=2020.0, step=0.5, end=2020.0)
    with pytest.raises(ValueError, match="whole number of steps"):
        no.NumericTimeline(start=0.0, step=0.3, end=1.0)
    with pytest.raises(ValueError):
        no.CalendarTimeline(start=date(2021, 1, 1), step=relativedelta(years=0))
    with pytest.raises(ValueError):
        no.CalendarTimeline(start=date(2021, 1, 1), step=relativedelta(days=-1))
    with pytest.raises(ValueError):
        no.CalendarTimeline(start=date(2021, 1, 1), end=date(2020, 1, 1), step=relativedelta(months=1))
    with pytest.raises(TypeError):
        no.CalendarTimeline(date(2021, 1, 1), relativedelta(months=1))  # ty:ignore[missing-argument, too-many-positional-arguments]


def test_model_lifecycle() -> None:
    no.checked()
    # 40 years annual steps
    m = RecordingModel(no.LinearTimeline(2011, 2051))
    no.run(m)
    # modify sees the start
    assert m.modified == (0, 2011.0, 1.0)
    assert m.steps == [(i, 2011.0 + i, 1.0) for i in range(1, 41)]
    # check sees the same step as step
    assert m.checks == m.steps
    assert m.finalised == (40, 2051.0, 1.0)
    assert m.run_state == no.Model.COMPLETED


def test_linear_timeline_is_integer() -> None:
    t = no.LinearTimeline(-5, 5)
    steps = list(t)
    assert steps == [(i, i + 1) for i in range(-5, 5)]
    assert all(type(x) is int for step in steps for x in step)
    assert str(t) == "<LinearTimeline start=-5 end=5>"
    # end of zero isn't treated as open-ended
    assert len(list(no.LinearTimeline(-3, 0))) == 3
    # start defaults to zero
    assert list(no.LinearTimeline(end=2)) == [(0, 1), (1, 2)]

    # numpy integers are accepted
    assert list(no.LinearTimeline(np.int64(0), np.int32(2))) == [(0, 1), (1, 2)]


def test_numeric_timeline() -> None:
    t = no.NumericTimeline(np.linspace(0.0, 1.0, 17))
    assert t.time == 0.0
    # step size varies so undefined before the first step
    assert no.time.isnever(t.dt)
    assert states(t) == [(i, i / 16, 1 / 16) for i in range(1, 17)]

    # uneven
    t = no.NumericTimeline([0, 1, 3, 7])
    assert [(step, t.time, t.dt) for step in t] == [
        ((0.0, 1.0), 1.0, 1.0),
        ((1.0, 3.0), 3.0, 2.0),
        ((3.0, 7.0), 7.0, 4.0),
    ]
    assert str(t) == "<NumericTimeline times=[0.0, 1.0, 3.0, 7.0]>"


def test_numeric_timeline_equal_steps() -> None:
    t = no.NumericTimeline(start=2025.0, end=2026.0, step=0.25)
    assert t.dt == 0.25
    assert list(t) == [(2025.0, 2025.25), (2025.25, 2025.5), (2025.5, 2025.75), (2025.75, 2026.0)]
    assert str(t) == "<NumericTimeline start=2025.0 end=2026.0 step=0.25>"

    # start + n * step falls just short of end, which mustn't produce an extra step
    assert 3 * 0.3 < 0.9
    t = no.NumericTimeline(start=0.0, end=0.9, step=0.3)
    assert list(t) == [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9)]
    assert t.time == 0.9

    # end of zero isn't treated as open-ended
    assert len(list(no.NumericTimeline(start=-1.0, end=0.0, step=0.5))) == 2

    # open-ended, computed from the start so no accumulated rounding error
    t = no.NumericTimeline(start=0.0, step=0.1)
    for _ in range(1000):
        step = next(t)
    assert step == (0.1 * 999, 0.1 * 1000)
    assert t.time == 0.1 * 1000
    assert str(t) == "<NumericTimeline start=0.0 end=None step=0.1>"


def test_calendar_timeline() -> None:
    # monthly timesteps checking we don't overshoot in shorter months
    dim = [31, 29, 31, 30, 31, 30, 31]

    for d in range(21, 32):
        start = date(2020, 1, d)
        t = no.CalendarTimeline(start=start, end=date(2020, 7, d), step=relativedelta(months=1, day=d))
        assert t.time == 0.0
        assert no.time.isnever(t.dt)
        for i, (t0, t1) in enumerate(t, start=1):
            assert t.index == i
            # check day of month adjusted only in months with fewer days
            assert t0.day == min(dim[i - 1], d)
            assert t1.day == min(dim[i], d)
            # float time is days since the start, dt days in the step
            assert t.time == (t1 - start).days
            assert t.dt == (t1 - t0).days
        assert t.index == 6


def test_calendar_timeline_preserves_day_of_month() -> None:
    t = no.CalendarTimeline(start=date(2020, 1, 31), end=date(2020, 5, 31), step=relativedelta(months=1))
    assert [(t1, t.dt) for _, t1 in t] == [
        (date(2020, 2, 29), 29.0),
        (date(2020, 3, 31), 31.0),
        (date(2020, 4, 30), 30.0),
        (date(2020, 5, 31), 31.0),
    ]


def test_calendar_timeline_end_off_grid() -> None:
    t = no.CalendarTimeline(start=date(2020, 1, 1), step=relativedelta(months=1), end=date(2020, 2, 15))
    assert list(t) == [(date(2020, 1, 1), date(2020, 2, 1)), (date(2020, 2, 1), date(2020, 3, 1))]
    assert str(t) == "<CalendarTimeline start=2020-01-01 end=2020-02-15 step=relativedelta(months=+1)>"


def test_open_ended_timeline() -> None:
    t = no.LinearTimeline()
    assert states(t, 10)[-1] == (10, 10.0, 1.0)
    assert str(t) == "<LinearTimeline start=0 end=None>"

    t = no.NumericTimeline(start=0.0, step=0.5)
    assert states(t, 3)[-1] == (3, 1.5, 0.5)

    t = no.CalendarTimeline(start=date(2020, 12, 17), step=relativedelta(days=1))
    assert states(t, 10)[-1] == (10, 10.0, 1.0)

    t = no.CalendarTimeline(start=date(2020, 12, 17), step=relativedelta(months=1))
    assert [dt for _, _, dt in states(t, 3)] == [31.0, 31.0, 28.0]


# check the timestepping is consistent across the different timeline implementations
def test_consistency() -> None:
    assert states(no.LinearTimeline(2009, 2021))[-1] == (12, 2021.0, 1.0)

    assert states(no.NumericTimeline([2020 + i / 12 for i in range(13)]))[-1][:2] == (12, 2021.0)

    t = no.CalendarTimeline(start=date(2019, 10, 31), end=date(2020, 10, 31), step=relativedelta(months=1, day=31))
    # 366 days, spanning a leap day
    assert states(t)[-1] == (12, 366.0, 31.0)


def test_resume() -> None:
    t0 = 2020
    n = 10

    class HaltEveryStep(no.Model):
        def __init__(self) -> None:
            super().__init__(no.LinearTimeline(t0, t0 + n), no.MonteCarlo.deterministic_identical_stream)

        def step(self) -> None:
            self.halt()

    m = HaltEveryStep()
    for i in range(1, n + 1):
        no.run(m)
        assert m.run_state == no.Model.HALTED
        assert m.timeline.index == i
        assert m.timeline.time == t0 + i

    with pytest.raises(StopIteration):
        no.run(m)


# check that halt/finalise interaction works as expected
def test_halt_finalise() -> None:
    m = RecordingModel(no.LinearTimeline(0, 3))
    no.run(m)
    assert m.finalised == (3, 3.0, 1.0)

    m = RecordingModel(no.LinearTimeline(0, 3), halt_at=1)
    no.run(m)
    assert m.finalised is None
    assert m.run_state == no.Model.HALTED
    assert m.timeline.index == 1
    # resume
    m.halt_at = 2
    no.run(m)
    assert m.finalised is None
    assert m.timeline.index == 2
    m.halt_at = None
    no.run(m)
    assert m.finalised == (3, 3.0, 1.0)
    assert m.run_state == no.Model.COMPLETED
    assert [s[0] for s in m.steps] == [1, 2, 3]

    with pytest.raises(StopIteration):
        no.run(m)


def test_check_failure() -> None:
    no.checked()

    class FailingCheck(RecordingModel):
        def check(self) -> bool:
            return self.timeline.index < 2

    m = FailingCheck(no.LinearTimeline(0, 3))
    with pytest.warns(RuntimeWarning, match="check\\(\\) FAILED"):
        assert not no.run(m)
    assert [s[0] for s in m.steps] == [1, 2]
    assert m.finalised is None
    assert m.timeline.index == 2
