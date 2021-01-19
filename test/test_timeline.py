import pytest
import numpy as np
import neworder as no

from datetime import date

from utils import assert_throws

#no.verbose()

class _TestModel(no.Model):
  def __init__(self):
    # 10 steps of 10
    super().__init__(no.LinearTimeline(0,100,10), no.MonteCarlo.deterministic_identical_stream)

    self.step_count = 0
    self.t_end = 100
    self.i_end = 10

  def step(self):
    self.step_count += 1

  def finalise(self):
    assert self.timeline().time() == self.t_end and self.timeline().index() == self.timeline().index()

class _TestModel2(no.Model):
  def __init__(self, start, end, steps):
    super().__init__(no.LinearTimeline(start, end, steps), no.MonteCarlo.deterministic_identical_stream)

    self.i = 0
    self.t = start
    self.steps = steps
    self.end = end

  def step(self):
    self.i += 1
    self.t += self.timeline().dt()

  def check(self):
    return self.timeline().index() == self.i and self.timeline().time() == self.t

  def finalise(self):
    assert self.timeline().at_end() and self.timeline().index() == self.steps

class _TestResume(no.Model):
  def __init__(self, t0, n):
    super().__init__(no.LinearTimeline(t0, t0 + n, n), no.MonteCarlo.deterministic_identical_stream)

  def step(self):
    self.halt()

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

  m = _TestModel2(0, 1, 1)
  no.run(m)
  assert m.timeline().at_end()
  assert m.timeline().index() == 1
  assert m.timeline().time() == 1.0

def test_timeline_validation():

  assert_throws(TypeError, no.LinearTimeline, 2020, 2020, [])
  assert_throws(ValueError, no.LinearTimeline, 2020, 0.0)
  assert_throws(ValueError, no.LinearTimeline, 2020, -1.0)

  assert_throws(ValueError, no.NumericTimeline, [2021, 2020])

  assert_throws(ValueError, no.CalendarTimeline, date(2021, 1, 1), date(2020, 1, 1), 1, "m")
  assert_throws(ValueError, no.CalendarTimeline, date(2019, 1, 1), date(2020, 1, 1), 1, "w")

  assert_throws(ValueError, no.CalendarTimeline, date(2019, 1, 1), date(2020, 1, 1), 1, "q")
  assert_throws(ValueError, no.CalendarTimeline, date(2019, 1, 1), date(2020, 1, 1), 0, "m")#
  # NOTE: passing a -ve int leads to a *TypeError* (when casting to size_t is attempted)
  assert_throws(TypeError, no.CalendarTimeline, date(2019, 1, 1), date(2020, 1, 1), -1, "m")


def test_linear_timeline():
  # 40 years annual steps
  m = _TestModel2(2011, 2051, 40)
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

    def finalise(self):
      assert self.timeline().dt() == 0.0
      assert self.timeline().time() == self.timeline().end()
      assert self.timeline().index() == 6

  for d in range(1,32):
    t = no.CalendarTimeline(date(2020, 1, d), date(2020, 7, d), 1, "m")

    m = CalendarModel(t)
    no.run(m)

def test_open_ended_timeline():

  class OpenEndedModel(no.Model):
    def __init__(self, calendartimeline):
      super().__init__(calendartimeline, no.MonteCarlo.deterministic_identical_stream)
      self.i = 0

    def step(self):
      assert self.i == self.timeline().index()
      self.i += 1
      if self.i > 10: self.halt()

  m = OpenEndedModel(no.LinearTimeline(0, 1))
  assert m.timeline().nsteps() == -1
  assert m.timeline().dt() == 1.0
  no.run(m)
  assert m.i == 11

  m = OpenEndedModel(no.CalendarTimeline(date(2020,12,17), 1, "d"))
  assert m.timeline().nsteps() == -1
  assert np.fabs(m.timeline().dt() - 1.0/365.2475) < 1e-8
  no.run(m)
  assert m.i == 11

  m = OpenEndedModel(no.CalendarTimeline(date(2020,12,17), 1, "m"))
  assert m.timeline().nsteps() == -1
  assert np.fabs(m.timeline().dt() - 31.0/365.2475) < 1e-8
  no.run(m)
  assert m.i == 11

def test_model():
  model = _TestModel()
  no.run(model)
  assert model.step_count == 10

# check the timestepping is consistent across the different timeline implementations
def test_consistency():

  # need to wrap timeline in a model to do the stepping, which isnt directly accessible from python
  class ConsistencyTest(no.Model):
    def __init__(self, timeline):
      super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)

    def step(self):
      pass

  m = ConsistencyTest(no.NoTimeline())
  assert m.timeline().nsteps() == 1
  no.run(m)
  assert m.timeline().index() == 1

  m = ConsistencyTest(no.LinearTimeline(2020, 2021, 12))

  assert m.timeline().nsteps() == 12
  no.run(m)
  assert m.timeline().index() == 12
  assert m.timeline().time() == 2021

  m = ConsistencyTest(no.NumericTimeline([2020 + i/12 for i in range(13)]))
  assert m.timeline().nsteps() == 12
  no.run(m)
  assert m.timeline().index() == 12
  assert m.timeline().time() == 2021

  s = date(2019,10,31)
  e = date(2020,10,31)

  m = ConsistencyTest(no.CalendarTimeline(s, e, 1, "m"))
  assert m.timeline().time().date() == s
  assert m.timeline().nsteps() == 12
  no.run(m)
  assert m.timeline().time().date() == e
  assert m.timeline().index() == 12

def test_resume():
  t0 = 0.1
  n = 10
  m = _TestResume(t0, n) # unit timesteps

  t = t0
  while not m.timeline().at_end():
    no.run(m)
    t += 1
    assert m.timeline().time() == t

  assert m.timeline().time() == t0 + n

# check that halt/finalise interaction works as expected
def test_halt_finalise():

  class HCModel(no.Model):
    def __init__(self, timeline, halt=False):
      super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)
      self.do_halt = halt
      self.finalise_called = False

    def step(self):
      if self.do_halt:
        self.halt()

    def finalise(self):
      self.finalise_called = True

  m = HCModel(no.LinearTimeline(0,3,3))
  no.run(m)
  assert m.finalise_called

  m = HCModel(no.LinearTimeline(0,3,3), True)
  no.run(m)
  assert not m.finalise_called
  assert not m.timeline().at_end()
  assert m.timeline().index() == 1
  no.run(m)
  assert not m.finalise_called
  assert not m.timeline().at_end()
  assert m.timeline().index() == 2
  no.run(m)
  assert m.finalise_called
  assert m.timeline().at_end()
  assert m.timeline().index() == 3

