# https://docs.python.org/3/extending/embedding.html

import numpy as np
import neworder

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

notafunc = 3


#neworder.log("callback: %s" % neworder.name())

def test():

  # Exp.value = p +/- 1/sqrt(N)
  h = neworder.hazard(0.2, 10000)
  if not isinstance(h, np.ndarray):
    return False
  if not len(h) == 10000:
    return False
  if not abs(np.mean(h) - 0.2) < 0.01:
    return False

  hv = neworder.hazard_v(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
  if not isinstance(hv, np.ndarray):
    return False
  if not len(hv) == 5:
    return False

  # Exp.value = 1/p +/- 1/sqrt(N)
  s = neworder.stopping(0.1, 10000)
  if not isinstance(s, np.ndarray):
    return False
  if not len(s) == 10000:
    return False
  if not abs(np.mean(s)/10 - 1.0) < 0.02:
    return False

  sv = neworder.stopping_v(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
  if not isinstance(sv, np.ndarray):
    return False
  if not len(sv) == 5:
    return False

  # Non-homogeneous Poisson process (time-dependent hazard) 
  nhpp = neworder.stopping_nhpp(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 10)
  if not isinstance(nhpp, np.ndarray):
    return False
  if not len(nhpp) == 10:
    return False

  return True
