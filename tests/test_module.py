import pytest
import numpy as np
import neworder as no

class _TestModel(no.Model):
  def __init__(self):
    # 10 steps of 10 with checkpoint at 50 and 100
    super().__init__(no.Timeline(0,100,[5,10]))

    self.step_count = 0
    self.checkpoint_count = 0

  def transition(self):
    self.step_count += 1

  def checkpoint(self):
    self.checkpoint_count += 1
    assert self.timeline().time() == 50 * self.checkpoint_count

@pytest.fixture
def model():
  return _TestModel()

def test_time(model):
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

def test_null_timeline(model):
  t0 = no.Timeline.null()
  assert t0.nsteps() == 1
  assert t0.dt() == 0.0
  assert not t0.at_end()
  assert t0.index() == 0
  assert t0.time() == 0.0
  t0.next()
  assert t0.at_checkpoint() #not currently exposed to python
  assert t0.at_end()
  assert t0.index() == 1
  assert t0.time() == 0.0

def test_timeline(model):
  # 40 years annual steps with 10y checkpoints
  t = no.Timeline(2011, 2051, [10,20,30,40])
  assert t.time() == 2011
  assert t.dt() == 1.0
  assert t.index() == 0
  while not t.at_end():
    t.next()
    assert t.time() == 2011 + t.index() #* t.dt()=1
    if t.time() % 10 == 1:
      assert t.at_checkpoint()
    else:
      assert not t.at_checkpoint()
  assert t.index() == 40
  assert t.time() == 2051

def test_model(model):
  no.run(model)
  assert model.step_count == 10
  assert model.checkpoint_count == 2

def test_multimodel(model):
  model2 = _TestModel()
  # TODO ensure 2 models can work...

