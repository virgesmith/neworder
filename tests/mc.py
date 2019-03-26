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

  n = 10000
  # 10% constant hazard for 10 time units, followed by zero
  dt = 1.0
  p = np.full(11, 0.1)
  p[-1] = 0
  a = no.first_arrival(p, dt, n)
  t.check(np.nanmin(a) > 0.0)
  t.check(np.nanmax(a) < 10.0)
  no.log("%f - %f" % (np.nanmin(a), np.nanmax(a)))

  # now set a to all 8.0
  a = np.full(n, 8.0)
  # next arrivals (absolute) only in range 8-10, if they happen
  b = no.next_arrival(a, p, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > 8.0)
  t.check(np.nanmax(b) < 10.0)

  # next arrivals with gap dt (absolute) only in range 9-10, if they happen
  b = no.next_arrival(a, p, dt, False, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > 9.0)
  t.check(np.nanmax(b) < 10.0)

  # next arrivals (relative) only in range 8-18, if they happen
  b = no.next_arrival(a, p, dt, True)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > 8.0)
  t.check(np.nanmax(b) < 18.0)

  # next arrivals with gap dt (relative) only in range 9-19, if they happen
  b = no.next_arrival(a, p, dt, True, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > 9.0)
  t.check(np.nanmax(b) < 19.0)

  # now set a back to random arrivals
  a = no.first_arrival(p, dt, n)
  # next arrivals (absolute) only in range (min(a), 10), if they happen
  b = no.next_arrival(a, p, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > np.nanmin(a))
  t.check(np.nanmax(b) < 10.0)

  # next arrivals with gap dt (absolute) only in range (min(a)+dt, 10), if they happen
  b = no.next_arrival(a, p, dt, False, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > np.nanmin(a) + dt)
  t.check(np.nanmax(b) < 10.0)

  # next arrivals (relative) only in range (min(a), max(a)+10), if they happen
  b = no.next_arrival(a, p, dt, True)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > np.nanmin(a))
  t.check(np.nanmax(b) < np.nanmax(a) + 10.0)

  # next arrivals with gap dt (relative) only in range (min(a)+dt, max(a)+dt+10), if they happen
  b = no.next_arrival(a, p, dt, True, dt)
  #no.log("%f - %f" % (np.nanmin(b), np.nanmax(b)))
  t.check(np.nanmin(b) > np.nanmin(a) + dt)
  t.check(np.nanmax(b) < np.nanmax(a) + dt + 10.0)

  return not t.any_failed

