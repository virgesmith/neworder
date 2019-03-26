# https://docs.python.org/3/extending/embedding.html

import numpy as np
import neworder

import test as test_

def add(a,b):
  return a + b

def sub(a,b):
  return a - b

def mul(a,b):
  return a * b

def div(a,b):
  return a / b

def void(a,b):
  pass

notafunc = 3.1

#neworder.log("callback: %s" % neworder.name())

def test():
  t = test_.Test()

  # Exp.value = p +/- 1/sqrt(N)
  h = neworder.hazard(0.2, 10000)
  t.check(isinstance(h, np.ndarray))
  t.check(len(h) == 10000)
  t.check(abs(np.mean(h) - 0.2) < 0.01)

  hv = neworder.hazard(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
  t.check(isinstance(hv, np.ndarray))
  t.check(len(hv) == 5)

  # Exp.value = 1/p +/- 1/sqrt(N)
  s = neworder.stopping(0.1, 10000)
  t.check(isinstance(s, np.ndarray))
  t.check(len(s) == 10000)
  t.check(abs(np.mean(s)/10 - 1.0) < 0.03)

  sv = neworder.stopping(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
  t.check(isinstance(sv, np.ndarray))
  t.check(len(sv) == 5)

  # Non-homogeneous Poisson process (time-dependent hazard) 
  nhpp = neworder.first_arrival(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 1.0, 10)
  t.check(isinstance(nhpp, np.ndarray))
  t.check(len(nhpp) == 10)

  return not t.any_failed
