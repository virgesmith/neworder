"""
Test framework expects modules with a function called test that
- takes no arguments
- returns True on success and False on failure
"""
import neworder as no
import numpy as np

def test():

  s = no.ustream(10000)
  if not isinstance(s, np.ndarray):
    return False
  if not len(s) == 10000:
    return False
  if not abs(np.mean(s) - 0.5) < 0.01:
    return False

  f = no.lazy_eval("2 + 2")
  if not f() == 4: 
    return False

  # test thinning algorithm for non-homogeneous Poisson process
  h = np.array([0.014] * 10)
  l = no.stopping_v(h)
  le = no.stopping_nhpp(h, 10000)
  no.log(sum(le)/len(le))

  # this also tests a zero hazard rate 
  h = np.array([i/3000 for i in range(100)])
  #no.log(h)
  le = no.stopping_nhpp(h, 10000)
  no.log(sum(le)/len(le))

  # not convinced this is working correctly
  h = np.array([0.999, 0.1])
  le = no.stopping_nhpp(h, 1000)
  no.log(sum(le)/len(le))

  return True
