from typing import cast
from datetime import datetime, date
import pytest
import numpy as np
import neworder as no


class _TestModel(no.Model):
  def __init__(self) -> None:
    # 10 steps of 10
    super().__init__(no.LinearTimeline(0,100,10), no.MonteCarlo.deterministic_identical_stream)

    self.step_count = 0
    self.t_end = 100
    self.i_end = 10

  def step(self) -> None:
    self.step_count += 1

  def finalise(self) -> None:
    assert self.timeline.time == self.t_end and self.timeline.index == self.timeline.index


class _TestModel2(no.Model):
  def __init__(self, start: float, end: float, steps: int) -> None:
    super().__init__(no.LinearTimeline(start, end, steps), no.MonteCarlo.deterministic_identical_stream)

    self.i = 0
    self.t = start
    self.steps = steps
    self.end = end

  def step(self) -> None:
    self.i += 1
    self.t += self.timeline.dt

  def check(self) -> bool:
    return self.timeline.index == self.i and self.timeline.time == self.t

  def finalise(self) -> None:
    assert self.timeline.at_end and self.timeline.index == self.steps

class _TestResume(no.Model):
  def __init__(self, t0: float, n: int) -> None:
    super().__init__(no.LinearTimeline(t0, t0 + n, n), no.MonteCarlo.deterministic_identical_stream)

  def step(self) -> None:
    self.halt()


class CustomTimeline(no.Timeline):
  def __init__(self) -> None:
    # NB base class takes care of index
    super().__init__()
    self.t = 1.0

  @property
  def start(self) -> float:
    return 1.0

  @property
  def end(self) -> float:
    return 0.0

  @property
  def nsteps(self) -> int:
    return -1

  @property
  def time(self) -> float:
    return 1.0 - self.t

  @property
  def dt(self) -> float:
    return self.t / 2

  def _next(self) -> None:
    self.t /= 2

  @property
  def at_end(self) -> bool:
    return False


class CustomTimelineModel(no.Model):
  def __init__(self) -> None:
    super().__init__(CustomTimeline(), no.MonteCarlo.deterministic_identical_stream)

  def step(self) -> None:
    self.halt()


def test_timeline_properties() -> None:
  n = no.NoTimeline()
  assert n.index == 0
  assert np.isnan(n.start)  # type: ignore[call-overload]
  assert np.isnan(n.time)  # type: ignore[call-overload]
  assert np.isnan(n.end)  # type: ignore[call-overload]
  assert n.dt == 0.0
  assert n.nsteps == 1


  with pytest.raises(AttributeError):
    n.index = 3  # type: ignore[misc]
  with pytest.raises(AttributeError):
    n.next()  # type: ignore[attr-defined]
  c = CustomTimeline()
  with pytest.raises(AttributeError):
    c.index = 3  # type: ignore[misc]
  # for python implementations next must be exposed
  # with pytest.raises(AttributeError):
  #   c.next()


def test_custom_timeline() -> None:
  ct = CustomTimeline()
  # default __repr__
  assert str(ct) == "<CustomTimeline index=0>"
  m = CustomTimelineModel()
  assert no.run(m)
  assert str(m.timeline) == "<CustomTimeline index=1>"


def test_time() -> None:
  t = -1e10
  assert no.time.DISTANT_PAST < t
  assert no.time.FAR_FUTURE > t
  t = 1e10
  assert no.time.DISTANT_PAST < t
  assert no.time.FAR_FUTURE > t

  # dreams never end
  assert no.time.NEVER != no.time.NEVER
  assert no.time.NEVER != t
  assert not no.time.NEVER < t
  assert not no.time.NEVER == t
  assert not no.time.NEVER >= t
  # no nay never
  assert not no.time.isnever(t)
  # no nay never no more
  assert no.time.isnever(no.time.NEVER)


def test_null_timeline() -> None:
  t0 = no.NoTimeline()
  assert t0.nsteps == 1
  assert t0.dt == 0.0
  assert not t0.at_end
  assert t0.index == 0
  assert no.time.isnever(t0.time)  # type: ignore[call-overload]
  assert no.time.isnever(t0.end)  # type: ignore[call-overload]

  m = _TestModel2(0, 1, 1)
  no.run(m)
  assert m.timeline.at_end
  assert m.timeline.index == 1
  assert m.timeline.time == 1.0


def test_timeline_validation() -> None:

  with pytest.raises(TypeError):
    no.LinearTimeline(2020, 2020, [])  # type: ignore[call-overload]
  with pytest.raises(ValueError):
    no.LinearTimeline(2020, 0.0)
  with pytest.raises(ValueError):
    no.LinearTimeline(2020, -1.0)
  with pytest.raises(ValueError):
    no.LinearTimeline(2020, 2019, 1)
  with pytest.raises(ValueError):
    no.LinearTimeline(2020, 2021, 0)
  with pytest.raises(ValueError):
    no.NumericTimeline([2021, 2020])
  with pytest.raises(ValueError):
    no.NumericTimeline([2020])
  with pytest.raises(ValueError):
    no.CalendarTimeline(date(2021, 1, 1), 0, "y")
  with pytest.raises(ValueError):
    no.CalendarTimeline(date(2021, 1, 1), 12, "n")
  with pytest.raises(ValueError):
    no.CalendarTimeline(date(2021, 1, 1), date(2020, 1, 1), 1, "m")
  with pytest.raises(ValueError):
    no.CalendarTimeline(date(2019, 1, 1), date(2020, 1, 1), 1, "w")
  with pytest.raises(ValueError):
    no.CalendarTimeline(date(2019, 1, 1), date(2020, 1, 1), 1, "q")
  with pytest.raises(ValueError):
    no.CalendarTimeline(date(2019, 1, 1), date(2020, 1, 1), 0, "m")#

  # NOTE: passing a -ve int leads to a *TypeError* (when casting to size_t is attempted)
  with pytest.raises(TypeError):
    no.CalendarTimeline(date(2019, 1, 1), date(2020, 1, 1), -1, "m")


def test_linear_timeline() -> None:
  # 40 years annual steps
  m = _TestModel2(2011, 2051, 40)
  assert m.timeline.time == 2011
  assert m.timeline.dt == 1.0
  assert m.timeline.index == 0
  assert m.timeline.end == 2051

  no.run(m)
  assert m.timeline.index == 40
  assert m.timeline.time == 2051


def test_numeric_timeline() -> None:
  class NumericTimelineModel(no.Model):
    def __init__(self, numerictimeline: no.Timeline) -> None:
      super().__init__(numerictimeline, no.MonteCarlo.deterministic_identical_stream)
    def step(self) -> None:
      assert self.timeline.dt == 1/16
      assert self.timeline.time == self.timeline.index / 16

    def finalise(self) -> None:
      assert self.timeline.time == 1.0
      assert self.timeline.time == self.timeline.end
      assert self.timeline.index == 16
  # 16 steps to avoid rounding errors
  m = NumericTimelineModel(no.NumericTimeline(np.linspace(0.0, 1.0, 17).tolist()))
  assert m.timeline.time == 0.0
  assert m.timeline.index == 0
  no.run(m)


def test_calendar_timeline() -> None:
  # monthly timesteps checking we don't overshoot in shorter months
  dim = [31, 29, 31, 30, 31, 30]

  class CalendarModel(no.Model):
    def __init__(self, calendartimeline: no.Timeline) -> None:
      super().__init__(calendartimeline, no.MonteCarlo.deterministic_identical_stream)

    def step(self) -> None:
      assert cast(date, self.timeline.time).day == min(dim[self.timeline.index], d)

    def finalise(self) -> None:
      assert self.timeline.dt == 0.0
      assert self.timeline.time == self.timeline.end
      assert self.timeline.index == 6

  for d in range(1,32):
    t = no.CalendarTimeline(date(2020, 1, d), date(2020, 7, d), 1, "m")

    m = CalendarModel(t)
    no.run(m)

def test_open_ended_timeline() -> None:

  class OpenEndedModel(no.Model):
    def __init__(self, timeline: no.Timeline) -> None:
      super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)
      self.i = 0

    def step(self) -> None:
      assert self.i == self.timeline.index
      self.i += 1
      if self.i > 10: self.halt()

  m = OpenEndedModel(no.LinearTimeline(0, 1))
  assert m.timeline.end == no.time.FAR_FUTURE
  assert m.timeline.nsteps == -1
  assert m.timeline.dt == 1.0
  no.run(m)
  assert m.i == 11

  m = OpenEndedModel(no.CalendarTimeline(date(2020, 12, 17), 1, "d"))
  assert m.timeline.end == no.time.FAR_FUTURE
  assert m.timeline.nsteps == -1
  assert np.fabs(m.timeline.dt - 1.0/365.2475) < 1e-8
  no.run(m)
  assert m.i == 11

  m = OpenEndedModel(no.CalendarTimeline(date(2020, 12, 17), 1, "m"))
  assert m.timeline.end == no.time.FAR_FUTURE
  assert m.timeline.nsteps == -1
  assert np.fabs(m.timeline.dt - 31.0 / 365.2475) < 1e-8
  no.run(m)
  assert m.i == 11

def test_model() -> None:
  model = _TestModel()
  no.run(model)
  assert model.step_count == 10

# check the timestepping is consistent across the different timeline implementations
def test_consistency() -> None:

  # need to wrap timeline in a model to do the stepping, which isnt directly accessible from python
  class ConsistencyTest(no.Model):
    def __init__(self, timeline: no.Timeline) -> None:
      super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)

    def step(self) -> None:
      pass

  m = ConsistencyTest(no.NoTimeline())
  assert m.timeline.nsteps == 1
  no.run(m)
  assert m.timeline.index == 1

  m = ConsistencyTest(no.LinearTimeline(2020, 2021, 12))

  assert m.timeline.nsteps == 12
  no.run(m)
  assert m.timeline.index == 12
  assert m.timeline.time == 2021

  m = ConsistencyTest(no.NumericTimeline([2020 + i/12 for i in range(13)]))
  assert m.timeline.nsteps == 12
  no.run(m)
  assert m.timeline.index == 12
  assert m.timeline.time == 2021

  s = date(2019, 10, 31)
  e = date(2020, 10, 31)

  m = ConsistencyTest(no.CalendarTimeline(s, e, 1, "m"))
  assert cast(datetime, m.timeline.time).date() == s
  assert m.timeline.nsteps == 12
  no.run(m)
  assert cast(datetime, m.timeline.time).date() == e
  assert m.timeline.index == 12

def test_resume() -> None:
  t0 = 0.1
  n = 10
  m = _TestResume(t0, n) # unit timesteps

  t = t0
  while not m.timeline.at_end:
    no.run(m)
    t += 1
    assert m.timeline.time == t

  assert m.timeline.time == t0 + n

# check that halt/finalise interaction works as expected
def test_halt_finalise() -> None:

  class HCModel(no.Model):
    def __init__(self, timeline: no.Timeline, halt: bool=False) -> None:
      super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)
      self.do_halt = halt
      self.finalise_called = False

    def step(self) -> None:
      if self.do_halt:
        self.halt()

    def finalise(self) -> None:
      self.finalise_called = True

  m = HCModel(no.LinearTimeline(0,3,3))
  no.run(m)
  assert m.finalise_called

  m = HCModel(no.LinearTimeline(0,3,3), True)
  no.run(m)
  assert not m.finalise_called
  assert not m.timeline.at_end
  assert m.timeline.index == 1
  # resume
  no.run(m)
  assert not m.finalise_called
  assert not m.timeline.at_end
  assert m.timeline.index == 2
  m.do_halt = False
  no.run(m)
  assert m.finalise_called
  assert m.timeline.at_end
  assert m.timeline.index == 3

  with pytest.raises(StopIteration):
    no.run(m)
