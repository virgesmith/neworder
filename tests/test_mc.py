import numpy as np
import neworder as no

def test_mc():

  model = no.Model(no.Timeline.null())

  x = model.mc().ustream(1)
  model.mc().reset()
  assert x == model.mc().ustream(1)

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


  # Exp.value = p +/- 1/sqrt(N)
  h = model.mc().hazard(0.2, 10000)
  assert isinstance(h, np.ndarray)
  assert len(h) == 10000
  assert abs(np.mean(h) - 0.2) < 0.01

  hv = model.mc().hazard(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
  assert isinstance(hv, np.ndarray)
  assert len(hv) == 5

  # Exp.value = 1/p +/- 1/sqrt(N)
  s = model.mc().stopping(0.1, 10000)
  assert isinstance(s, np.ndarray)
  assert len(s) == 10000
  assert abs(np.mean(s)/10 - 1.0) < 0.03

  sv = model.mc().stopping(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
  assert isinstance(sv, np.ndarray)
  assert len(sv) == 5

  # Non-homogeneous Poisson process (time-dependent hazard)
  nhpp = model.mc().first_arrival(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 1.0, 10, 0.0)
  assert isinstance(nhpp, np.ndarray)
  assert len(nhpp) == 10


def _test_mc_parallel(model):

  mc = model.mc()
  assert mc.seed() == 19937



