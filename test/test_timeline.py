import pytest
import numpy as np
import neworder as no

from datetime import date

from utils import assert_throws

#no.verbose()

class _TestModel(no.Model):
  def __init__(self):
    # 10 steps of 10 with checkpoint at 50 and 100
    super().__init__(no.LinearTimeline(0,100,[5,10]), no.MonteCarlo.deterministic_identical_stream)

    self.step_count = 0
    self.checkpoint_count = 0

  def step(self):
    self.step_count += 1

  def checkpoint(self):
    self.checkpoint_count += 1
    assert self.timeline().time() == 50 * self.checkpoint_count

class _TestModel2(no.Model):
  def __init__(self, start, end, checkpoints):
    super().__init__(no.LinearTimeline(start, end, checkpoints), no.MonteCarlo.deterministic_identical_stream)

    self.i = 0
    self.t = start
    self.checkpoints = checkpoints
    self.end = end

  def step(self):
    self.i += 1
    self.t += self.timeline().dt()

  def check(self):
    return self.timeline().index() == self.i and self.timeline().time() == self.t

  def checkpoint(self):
    assert self.timeline().at_checkpoint() and self.timeline().index() in self.checkpoints


def test_time():
  t = -1e10
  assert no.time.distant_past() < t
  assert no.time.far_future() > t
  t = 1e10
  assert no.time.distant_past() < t
  assert no.time.far_future() > t

  # dreams never end
  assert no.time.never() != no.time.never()
  assert no.time.never() != t
  assert not no.time.never() < t
  assert not no.time.never() == t
  assert not no.time.never() >= t
  # no nay never
  assert not no.time.isnever(t)
  # no nay never no more
  assert no.time.isnever(no.time.never())

def test_null_timeline():
  t0 = no.NoTimeline()
  assert t0.nsteps() == 1
  assert t0.dt() == 0.0
  assert not t0.at_end()
  assert t0.index() == 0
  assert no.time.isnever(t0.time())

  m = _TestModel2(0, 1, [1])
  no.run(m)
  assert m.timeline().at_checkpoint()
  assert m.timeline().at_end()
  assert m.timeline().index() == 1
  assert m.timeline().time() == 1.0

def test_timeline_validation():

  assert_throws(ValueError, no.LinearTimeline, 2020, 2020, [])
  assert_throws(ValueError, no.LinearTimeline, 2020, 2019, [1])
  assert_throws(ValueError, no.LinearTimeline, 2020, 2022, [2,1])
  assert_throws(ValueError, no.LinearTimeline, 2020, 2022, [1,1])

  assert_throws(ValueError, no.NumericTimeline, [2021, 2020], [1])

  assert_throws(ValueError, no.CalendarTimeline, date(2021, 1, 1), date(2020, 1, 1), 1, "m", 1)
  assert_throws(ValueError, no.CalendarTimeline, date(2019, 1, 1), date(2020, 1, 1), 1, "w", 1)

  assert_throws(ValueError, no.CalendarTimeline, date(2019, 1, 1), date(2020, 1, 1), 1, "m", 0)
  assert_throws(ValueError, no.CalendarTimeline, date(2019, 1, 1), date(2020, 1, 1), 1, "m", 13)


def test_linear_timeline():
  # 40 years annual steps with 10y checkpoints
  m = _TestModel2(2011, 2051, [10,20,30,40])
  assert m.timeline().time() == 2011
  assert m.timeline().dt() == 1.0
  assert m.timeline().index() == 0

  no.run(m)
  assert m.timeline().index() == 40
  assert m.timeline().time() == 2051

def test_calendar_timeline():
  # monthly timesteps checking we don't overshoot in shorter months
  dim = [31,29,31,30,31,30]

  class CalendarModel(no.Model):
    def __init__(self, calendartimeline):
      super().__init__(calendartimeline, no.MonteCarlo.deterministic_identical_stream)

    def step(self):
      assert t.time().day == min(dim[t.index()], d)

    def checkpoint(self):
      pass

  for d in range(1,32):
    t = no.CalendarTimeline(date(2020, 1, d), date(2020, 7, d), 1, "m", 1)

    m = CalendarModel(t)
    no.run(m)

def test_model():
  model = _TestModel()
  no.run(model)
  assert model.step_count == 10
  assert model.checkpoint_count == 2

# check the timestepping is consistent across the different timeline implementations
def test_consistency():

  # need to wrap timeline in a model to do the stepping, which isnt directly accessible from python
  class ConsistencyTest(no.Model):
    def __init__(self, timeline):
      super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)

    def step(self):
      pass

    def checkpoint(self):
      pass

  m = ConsistencyTest(no.NoTimeline())
  assert m.timeline().nsteps() == 1
  no.run(m)
  assert m.timeline().index() == 1

  m = ConsistencyTest(no.LinearTimeline(2020, 2021, [12]))

  assert m.timeline().nsteps() == 12
  no.run(m)
  assert m.timeline().index() == 12
  assert m.timeline().time() == 2021

  m = ConsistencyTest(no.NumericTimeline([2020 + i/12 for i in range(13)], [12]))
  assert m.timeline().nsteps() == 12
  no.run(m)
  assert m.timeline().index() == 12
  assert m.timeline().time() == 2021

  s = date(2019,10,31)
  e = date(2020,10,31)

  m = ConsistencyTest(no.CalendarTimeline(s, e, 1, "m", 1))
  assert m.timeline().time().date() == s
  assert m.timeline().nsteps() == 12
  no.run(m)
  assert m.timeline().time().date() == e
  assert m.timeline().index() == 12



