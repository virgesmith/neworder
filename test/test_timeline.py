from collections.abc import Iterable, Iterator
from datetime import date
from itertools import islice
from types import GeneratorType

import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

import neworder as no
from neworder.timeline import Timestep


class RecordingModel(no.Model):
    def __init__(self, timeline: Iterable[Timestep], halt_at: int | None = None) -> None:
        super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)
        self.halt_at = halt_at
        self.initial = self.timestep
        self.modified: list[Timestep | None] = []
        self.steps: list[Timestep] = []
        self.checks: list[Timestep] = []
        self.finalised: Timestep | None = None

    def modify(self) -> None:
        self.modified.append(self.timestep)

    def step(self) -> None:
        self.steps.append(self.timestep)
        if self.timestep.index == self.halt_at:
            self.halt()

    def check(self) -> bool:
        self.checks.append(self.timestep)
        return True

    def finalise(self) -> None:
        self.finalised = self.timestep


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


def test_null() -> None:
    assert isinstance(no.timeline.null(), Iterator)
    (step,) = no.timeline.null()
    assert step[:3] == (1, None, None)
    assert no.time.isnever(step.dt)


def test_linear() -> None:
    steps = list(no.timeline.linear(-2, end=2))
    assert steps == [(1, -2, -1, 1.0), (2, -1, 0, 1.0), (3, 0, 1, 1.0), (4, 1, 2, 1.0)]
    assert all(type(s.start) is int and type(s.end) is int and type(s.dt) is float for s in steps)
    # start defaults to zero
    assert list(no.timeline.linear(end=2)) == [(1, 0, 1, 1.0), (2, 1, 2, 1.0)]
    # numpy integers are accepted, and give python ints
    steps = list(no.timeline.linear(np.int64(0), end=np.int32(2)))
    assert steps == [(1, 0, 1, 1.0), (2, 1, 2, 1.0)]
    assert type(steps[0].start) is int

    # open-ended
    open_ended = no.timeline.linear(2020)
    assert isinstance(open_ended, GeneratorType)
    assert list(islice(open_ended, 3)) == [(1, 2020, 2021, 1.0), (2, 2021, 2022, 1.0), (3, 2022, 2023, 1.0)]


def test_numeric() -> None:
    assert list(no.timeline.numeric([0, 1, 3, 7])) == [(1, 0.0, 1.0, 1.0), (2, 1.0, 3.0, 2.0), (3, 3.0, 7.0, 4.0)]
    steps = list(no.timeline.numeric(np.linspace(0.0, 1.0, 17)))
    assert [s.end for s in steps] == [i / 16 for i in range(1, 17)]
    assert all(s.dt == 1 / 16 for s in steps)


def test_numeric_equal_steps() -> None:
    steps = list(no.timeline.numeric(start=2025.0, step=0.25, end=2026.0))
    assert [(s.start, s.end) for s in steps] == [
        (2025.0, 2025.25),
        (2025.25, 2025.5),
        (2025.5, 2025.75),
        (2025.75, 2026.0),
    ]
    assert all(s.dt == 0.25 for s in steps)

    # start + n * step falls just short of end, which mustn't produce an extra step
    assert 3 * 0.3 < 0.9
    steps = list(no.timeline.numeric(start=0.0, step=0.3, end=0.9))
    assert [(s.start, s.end) for s in steps] == [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9)]

    # end of zero isn't treated as open-ended
    assert len(list(no.timeline.numeric(start=-1.0, step=0.5, end=0.0))) == 2
    # nor rejected when start + n * step only approximates zero
    assert -0.3 + 3 * 0.1 != 0.0
    steps = list(no.timeline.numeric(start=-0.3, step=0.1, end=0.0))
    assert len(steps) == 3
    assert steps[-1].end == 0.0

    # open-ended, computed from the start so no accumulated rounding error
    last = list(islice(no.timeline.numeric(start=0.0, step=0.1), 1000))[-1]
    assert last == (1000, 0.1 * 999, 0.1 * 1000, 0.1)


def test_calendar() -> None:
    # monthly steps, checking the day of the month is only adjusted in months with fewer days
    dim = [31, 29, 31, 30, 31, 30, 31]
    for d in range(21, 32):
        start = date(2020, 1, d)
        steps = list(no.timeline.calendar(start=start, step=relativedelta(months=1, day=d), end=date(2020, 7, d)))
        assert [s.index for s in steps] == list(range(1, 7))
        for i, (_, t0, t1, dt) in enumerate(steps, start=1):
            assert t0.day == min(dim[i - 1], d)
            assert t1.day == min(dim[i], d)
            # dt is the length of the step in years, ACT/365
            assert dt == (t1 - t0).days / 365

    steps = list(no.timeline.calendar(start=date(2020, 1, 31), step=relativedelta(months=1), end=date(2020, 5, 31)))
    assert [(s.end, s.dt) for s in steps] == [
        (date(2020, 2, 29), 29 / 365),
        (date(2020, 3, 31), 31 / 365),
        (date(2020, 4, 30), 30 / 365),
        (date(2020, 5, 31), 31 / 365),
    ]

    # a year spanning a leap day
    (step,) = no.timeline.calendar(start=date(2020, 1, 1), step=relativedelta(years=1), end=date(2021, 1, 1))
    assert step.dt == 366 / 365


def test_calendar_end_off_grid() -> None:
    # the final step is shortened to end at the end date
    steps = list(no.timeline.calendar(start=date(2020, 1, 1), step=relativedelta(months=1), end=date(2020, 2, 15)))
    assert steps == [
        (1, date(2020, 1, 1), date(2020, 2, 1), 31 / 365),
        (2, date(2020, 2, 1), date(2020, 2, 15), 14 / 365),
    ]
    # a long daily timeline
    steps = list(no.timeline.calendar(start=date(1900, 1, 1), step=relativedelta(days=1), end=date(2100, 1, 1)))
    assert len(steps) == (date(2100, 1, 1) - date(1900, 1, 1)).days
    assert steps[-1].end == date(2100, 1, 1)


def test_calendar_open_ended() -> None:
    steps = no.timeline.calendar(start=date(2020, 12, 17), step=relativedelta(months=1))
    assert isinstance(steps, GeneratorType)
    assert [s.dt for s in islice(steps, 3)] == [31 / 365, 31 / 365, 28 / 365]


def test_validation() -> None:
    with pytest.raises(TypeError, match="Use numeric"):
        no.timeline.linear(0.0, end=1)  # ty:ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="integer end"):
        no.timeline.linear(0, end=0.5)  # ty:ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="integer start"):
        no.timeline.linear(True, end=3)
    # end is keyword-only, as in neworder 1.x LinearTimeline(start, step) was open-ended
    with pytest.raises(TypeError):
        no.timeline.linear(0, 1)  # ty:ignore[too-many-positional-arguments]
    with pytest.raises(ValueError):
        no.timeline.linear(2020, end=2020)
    with pytest.raises(ValueError):
        no.timeline.linear(2020, end=2019)
    with pytest.raises(ValueError):
        no.timeline.numeric([2021, 2020])
    with pytest.raises(ValueError, match="index 2"):
        no.timeline.numeric([2019, 2020, 2020])
    with pytest.raises(ValueError):
        no.timeline.numeric([2020])
    with pytest.raises(ValueError, match="index 1"):
        no.timeline.numeric([0.0, float("nan"), 2.0])
    with pytest.raises(TypeError):
        no.timeline.numeric([2020, 2021], start=2020, step=1)
    with pytest.raises(TypeError):
        no.timeline.numeric(start=2020.0)
    with pytest.raises(ValueError):
        no.timeline.numeric(start=2020.0, step=0.0)
    with pytest.raises(ValueError):
        no.timeline.numeric(start=2020.0, step=0.5, end=2020.0)
    with pytest.raises(ValueError, match="whole number of steps"):
        no.timeline.numeric(start=0.0, step=0.3, end=1.0)
    with pytest.raises(ValueError):
        no.timeline.calendar(start=date(2021, 1, 1), step=relativedelta(years=0))
    with pytest.raises(ValueError):
        no.timeline.calendar(start=date(2021, 1, 1), step=relativedelta(days=-1))
    with pytest.raises(ValueError):
        no.timeline.calendar(start=date(2021, 1, 1), step=relativedelta(months=1), end=date(2020, 1, 1))
    with pytest.raises(TypeError):
        no.timeline.calendar(date(2021, 1, 1), relativedelta(months=1))  # ty:ignore[missing-argument, too-many-positional-arguments]


def test_model_lifecycle() -> None:
    no.checked()
    m = RecordingModel(no.timeline.linear(2011, end=2051))
    assert m.timestep is None
    assert no.run(m)
    # no step has been taken in the constructor or modify
    assert m.initial is None
    assert m.modified == [None]
    assert m.steps == list(no.timeline.linear(2011, end=2051))
    # check sees the same step as step
    assert m.checks == m.steps
    # finalise sees the last step
    assert m.finalised == (40, 2050, 2051, 1.0)
    assert m.run_state == no.Model.COMPLETED


def test_custom_timeline() -> None:
    # any iterable of steps
    m = RecordingModel([Timestep(1, "a", "b", 0.5), Timestep(2, "b", "c", 0.5)])
    no.run(m)
    assert [s.end for s in m.steps] == ["b", "c"]

    # e.g. a generator function, open-ended
    def halving() -> Iterator[Timestep]:
        i, t = 1, 0.0
        while True:
            yield Timestep(i, t, t + 0.5**i, 0.5**i)
            i, t = i + 1, t + 0.5**i

    m = RecordingModel(halving(), halt_at=3)
    no.run(m)
    assert m.steps == [(1, 0.0, 0.5, 0.5), (2, 0.5, 0.75, 0.25), (3, 0.75, 0.875, 0.125)]


def test_timeline_reuse() -> None:
    # a list of timesteps can be used by more than one model
    timeline = list(no.timeline.linear(0, end=3))
    m1, m2 = RecordingModel(timeline), RecordingModel(timeline)
    no.run(m1)
    no.run(m2)
    assert m1.steps == m2.steps == timeline
    # but an iterator only once
    timeline = no.timeline.linear(0, end=3)
    m1, m2 = RecordingModel(timeline), RecordingModel(timeline)
    no.run(m1)
    no.run(m2)
    assert len(m1.steps) == 3
    assert m2.steps == []


def test_empty_timeline() -> None:
    m = RecordingModel(())
    assert no.run(m)
    assert m.steps == []
    assert m.finalised is None
    assert m.run_state == no.Model.COMPLETED


def test_resume() -> None:
    class HaltEveryStep(no.Model):
        def __init__(self) -> None:
            super().__init__(no.timeline.linear(2020, end=2030), no.MonteCarlo.deterministic_identical_stream)

        def step(self) -> None:
            self.halt()

    m = HaltEveryStep()
    for i in range(1, 11):
        no.run(m)
        assert m.run_state == no.Model.HALTED
        assert m.timestep.index == i
        assert m.timestep.end == 2020 + i

    # halted on the final step, so resuming completes the run without taking a step
    assert no.run(m)
    assert m.run_state == no.Model.COMPLETED
    assert m.timestep.index == 10

    with pytest.raises(StopIteration):
        no.run(m)


# check that halt/finalise interaction works as expected
def test_halt_finalise() -> None:
    m = RecordingModel(no.timeline.linear(0, end=3), halt_at=1)
    no.run(m)
    assert m.finalised is None
    assert m.run_state == no.Model.HALTED
    # resume
    m.halt_at = 2
    no.run(m)
    assert m.finalised is None
    assert m.timestep.index == 2
    m.halt_at = None
    no.run(m)
    assert m.finalised == (3, 2, 3, 1.0)
    assert m.run_state == no.Model.COMPLETED
    assert [s.index for s in m.steps] == [1, 2, 3]
    # modify is only called by the first run, not when resuming
    assert m.modified == [None]

    with pytest.raises(StopIteration):
        no.run(m)


def test_check_failure() -> None:
    no.checked()

    class FailingCheck(RecordingModel):
        def check(self) -> bool:
            return self.timestep.index < 2

    m = FailingCheck(no.timeline.linear(0, end=3))
    with pytest.warns(RuntimeWarning, match="check\\(\\) FAILED"):
        assert not no.run(m)
    assert [s.index for s in m.steps] == [1, 2]
    assert m.finalised is None


def test_halt_in_modify() -> None:
    class HaltInModify(RecordingModel):
        def modify(self) -> None:
            super().modify()
            self.halt()

    m = HaltInModify(no.timeline.linear(0, end=3))
    assert no.run(m)
    assert m.run_state == no.Model.HALTED
    assert m.steps == []
    # no step was taken, so resuming runs every step, without calling modify again
    no.run(m)
    assert [s.index for s in m.steps] == [1, 2, 3]
    assert m.run_state == no.Model.COMPLETED
    assert m.modified == [None]


def test_halt_on_final_step_then_resume() -> None:
    m = RecordingModel(no.timeline.linear(0, end=2), halt_at=2)
    no.run(m)
    assert m.run_state == no.Model.HALTED
    assert m.finalised is None
    # resuming has no more steps to take, so it just finalises
    m.halt_at = None
    assert no.run(m)
    assert m.run_state == no.Model.COMPLETED
    assert m.finalised == (2, 1, 2, 1.0)
    assert [s.index for s in m.steps] == [1, 2]
    assert m.modified == [None]


def test_advance_timeline_manually() -> None:
    class Skipping(RecordingModel):
        def step(self) -> None:
            super().step()
            if self.timestep.index == 2:
                assert isinstance(self.timeline, Iterator)
                assert next(self.timeline, None) == (3, 2, 3, 1.0)
                assert self.timestep.index == 2

    m = Skipping(no.timeline.linear(0, end=4))
    assert no.run(m)
    assert [s.index for s in m.steps] == [1, 2, 4]
    assert m.finalised == (4, 3, 4, 1.0)

    # skipping the remaining steps finalises after the current one
    m = Skipping(no.timeline.linear(0, end=3))
    assert no.run(m)
    assert [s.index for s in m.steps] == [1, 2]
    assert m.run_state == no.Model.COMPLETED
    assert m.finalised == (2, 1, 2, 1.0)


def test_iterating_timeline_consumes_iterator_not_list() -> None:
    class Iterating(RecordingModel):
        def step(self) -> None:
            super().step()
            self.remaining = len(list(self.timeline))

    m = Iterating(list(no.timeline.linear(0, end=3)))
    no.run(m)
    assert [s.index for s in m.steps] == [1, 2, 3]

    m = Iterating(no.timeline.linear(0, end=3))
    no.run(m)
    assert [s.index for s in m.steps] == [1]
    assert m.remaining == 2
    assert m.run_state == no.Model.COMPLETED


def test_next_on_exhausted_timeline_raises() -> None:
    class Overrunning(RecordingModel):
        def step(self) -> None:
            super().step()
            assert isinstance(self.timeline, Iterator)
            next(self.timeline)

    m = Overrunning(no.timeline.linear(0, end=1))
    with pytest.raises(StopIteration):
        no.run(m)
    assert m.finalised is None


def test_partly_consumed_timeline() -> None:
    timeline = no.timeline.linear(0, end=4)
    next(timeline)
    m = RecordingModel(timeline)
    assert no.run(m)
    assert [s.index for s in m.steps] == [2, 3, 4]


def test_rerun_completed_model() -> None:
    m = RecordingModel(no.timeline.linear(0, end=2))
    no.run(m)
    m.modified = []
    with pytest.raises(StopIteration):
        no.run(m)
    # fails before modify is called
    assert m.modified == []
    assert m.run_state == no.Model.COMPLETED


def test_strided_arrays() -> None:
    a = np.array([1.0, no.time.NEVER, 2.0, no.time.NEVER])
    assert no.time.isnever(a[::2]).tolist() == [False, False]
    assert no.time.isnever(a[1::2]).tolist() == [True, True]
    assert no.time.isnever(a[::-1]).tolist() == [True, False, True, False]
