import pytest
import numpy as np
import neworder as no

#no.verbose()

class _TestModel(no.Model):
  def __init__(self):
    # 10 steps of 10 with checkpoint at 50 and 100
    super().__init__(no.Timeline(0,100,[5,10]), no.MonteCarlo.deterministic_identical_stream)

    self.step_count = 0
    self.checkpoint_count = 0

  def step(self):
    self.step_count += 1

  def checkpoint(self):
    self.checkpoint_count += 1
    assert self.timeline().time() == 50 * self.checkpoint_count

class _TestModel2(no.Model):
  def __init__(self, start, end, checkpoints):
    super().__init__(no.Timeline(start, end, checkpoints), no.MonteCarlo.deterministic_identical_stream)

    self.i = 0
    self.t = start
    self.checkpoints = checkpoints 
    self.end = end

  # if you implement step you MUST call super.step() to increment the timeline
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
  t0 = no.Timeline.null()
  assert t0.nsteps() == 1
  assert t0.dt() == 0.0
  assert not t0.at_end()
  assert t0.index() == 0
  assert t0.time() == 0.0

  m = _TestModel2(0, 0, [1])
  no.run(m)
  assert m.timeline().at_checkpoint() 
  assert m.timeline().at_end()
  assert m.timeline().index() == 1
  assert m.timeline().time() == 0.0

def test_timeline():
  # 40 years annual steps with 10y checkpoints
  m = _TestModel2(2011, 2051, [10,20,30,40])
  assert m.timeline().time() == 2011
  assert m.timeline().dt() == 1.0
  assert m.timeline().index() == 0

  no.run(m)
  assert m.timeline().index() == 40
  assert m.timeline().time() == 2051

def test_model():
  model = _TestModel() 
  no.run(model)
  assert model.step_count == 10
  assert model.checkpoint_count == 2


