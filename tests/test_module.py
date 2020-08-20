import pytest
import numpy as np
import neworder as no

#no.module_init(independent=False, verbose=False)

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

#ensure initialised
@pytest.fixture
def model():
  no.module_init(independent=False, verbose=False)
  return _TestModel()



def test_basics(model):
  # just check you can call the functions
  no.name()
  no.version()
  no.python()
  assert no.mpi.indep() == False
  # TODO 
  # assert no.verbose() == False
  no.log("testing")
  assert not no.embedded()

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

def test_mc(model):
  if no.mpi.size() == 1:
    _test_mc_serial(model)
  else:
    _test_mc_parallel(model)

def _test_mc_serial(model):
  mc = model.mc()
  assert mc.seed() == 19937

  n = 10000
  # 10% constant hazard for 10 time units, followed by zero
  dt = 1.0
  p = np.full(11, 0.1)
  p[-1] = 0
  a = mc.first_arrival(p, dt, n)
  assert np.nanmin(a) > 0.0
  assert np.nanmax(a) < 10.0
  no.log("%f - %f" % (np.nanmin(a), np.nanmax(a)))

  # now set a to all 8.0
  a = np.full(n, 8.0)
  # next arrivals (absolute) only in range 8-10, if they happen
  b = mc.next_arrival(a, p, dt)
  assert np.nanmin(b) > 8.0
  assert np.nanmax(b) < 10.0

  # next arrivals with gap dt (absolute) only in range 9-10, if they happen
  b = mc.next_arrival(a, p, dt, False, dt)
  assert np.nanmin(b) > 9.0
  assert np.nanmax(b) < 10.0

  # next arrivals (relative) only in range 8-18, if they happen
  b = mc.next_arrival(a, p, dt, True)
  assert np.nanmin(b) > 8.0
  assert np.nanmax(b) < 18.0

  # next arrivals with gap dt (relative) only in range 9-19, if they happen
  b = mc.next_arrival(a, p, dt, True, dt)
  assert np.nanmin(b) > 9.0
  assert np.nanmax(b) < 19.0

  # now set a back to random arrivals
  a = mc.first_arrival(p, dt, n)
  # next arrivals (absolute) only in range (min(a), 10), if they happen
  b = mc.next_arrival(a, p, dt)
  assert np.nanmin(b) > np.nanmin(a)
  assert np.nanmax(b) < 10.0

  # next arrivals with gap dt (absolute) only in range (min(a)+dt, 10), if they happen
  b = mc.next_arrival(a, p, dt, False, dt)
  assert np.nanmin(b) > np.nanmin(a) + dt
  assert np.nanmax(b) < 10.0

  # next arrivals (relative) only in range (min(a), max(a)+10), if they happen
  b = mc.next_arrival(a, p, dt, True)
  assert np.nanmin(b) > np.nanmin(a)
  assert np.nanmax(b) < np.nanmax(a) + 10.0

  # next arrivals with gap dt (relative) only in range (min(a)+dt, max(a)+dt+10), if they happen
  b = mc.next_arrival(a, p, dt, True, dt)
  assert np.nanmin(b) > np.nanmin(a) + dt
  assert np.nanmax(b) < np.nanmax(a) + dt + 10.0

  mc.reset()
  a = mc.first_arrival(np.array([0.1, 0.2, 0.3]), 1.0, 6, 0.0)
  assert len(a) == 6
  # only works for single-process
  assert a[0] == 3.6177811673165667
  assert a[1] == 0.6896205251312125
  assert a[2] == 3.610216282947799
  assert a[3] == 7.883336832344425
  assert a[4] == 6.461894711350323
  assert a[5] == 2.8566436418145944

def _test_mc_parallel(model):

  mc = model.mc()
  assert mc.seed() == 19937
