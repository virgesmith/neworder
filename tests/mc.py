"""
mc.py: Monte-Carlo specific tests

Test framework expects modules with a function called test that
- takes no arguments
- returns True on success and False on failure
"""
import neworder as no
import numpy as np

import test as test_

def test():
  t = test_.Test()

  # base model for MC engine
  model = no.Model(no.Timeline.null())

  n = 10000
  # 10% constant hazard for 10 time units, followed by zero
  dt = 1.0
  p = np.full(11, 0.1)
  p[-1] = 0
  a = model.mc().first_arrival(p, dt, n)
  t.check(np.nanmin(a) > 0.0)
  t.check(np.nanmax(a) < 10.0)
  no.log("%f - %f" % (np.nanmin(a), np.nanmax(a)))

  # now set a to all 8.0
  a = np.full(n, 8.0)
  # next arrivals (absolute) only in range 8-10, if they happen
  b = model.mc().next_arrival(a, p, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > 8.0)
  t.check(np.nanmax(b) < 10.0)

  # next arrivals with gap dt (absolute) only in range 9-10, if they happen
  b = model.mc().next_arrival(a, p, dt, False, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > 9.0)
  t.check(np.nanmax(b) < 10.0)

  # next arrivals (relative) only in range 8-18, if they happen
  b = model.mc().next_arrival(a, p, dt, True)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > 8.0)
  t.check(np.nanmax(b) < 18.0)

  # next arrivals with gap dt (relative) only in range 9-19, if they happen
  b = model.mc().next_arrival(a, p, dt, True, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > 9.0)
  t.check(np.nanmax(b) < 19.0)

  # now set a back to random arrivals
  a = model.mc().first_arrival(p, dt, n)
  # next arrivals (absolute) only in range (min(a), 10), if they happen
  b = model.mc().next_arrival(a, p, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > np.nanmin(a))
  t.check(np.nanmax(b) < 10.0)

  # next arrivals with gap dt (absolute) only in range (min(a)+dt, 10), if they happen
  b = model.mc().next_arrival(a, p, dt, False, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > np.nanmin(a) + dt)
  t.check(np.nanmax(b) < 10.0)

  # next arrivals (relative) only in range (min(a), max(a)+10), if they happen
  b = model.mc().next_arrival(a, p, dt, True)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > np.nanmin(a))
  t.check(np.nanmax(b) < np.nanmax(a) + 10.0)

  # next arrivals with gap dt (relative) only in range (min(a)+dt, max(a)+dt+10), if they happen
  b = model.mc().next_arrival(a, p, dt, True, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > np.nanmin(a) + dt)
  t.check(np.nanmax(b) < np.nanmax(a) + dt + 10.0)

  model.mc().reset()
  a = model.mc().first_arrival(np.array([0.1, 0.2, 0.3]), 1.0, 6, 0.0)
  t.check(len(a) == 6)
  # only works for single-process
  if no.mpi.size() == 1:
    # these are the rust values...
    t.check_eq(a[0], 3.6177811673165667)
    t.check_eq(a[1], 0.6896205251312125)
    t.check_eq(a[2], 3.610216282947799)
    t.check_eq(a[3], 7.883336832344425)
    t.check_eq(a[4], 6.461894711350323)
    t.check_eq(a[5], 2.8566436418145944)

  return not t.any_failed

