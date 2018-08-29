"""
Test framework expects modules with a function called test that
- takes no arguments
- returns True on success and False on failure
"""
import neworder as no
import numpy as np

def test():

  dv = no.DVector(10)

  no.log(no.ustream(10))
  no.log(no.ustream(10))

  # test arithmetic
  dv = dv + 0.5
  dv = 0.5 + dv
  #dv = dv + dv
  dv = 0.5 * dv
  dv = dv * 0.5
  # dv = dv / 0.5

  no.log(np.array(no.hazard_v(no.ustream(1000)).tolist()).mean())

  f = no.lazy_eval("2 + 2")

  no.testVec = no.DVector.fromlist([1,2,3,4])

  no.testVec2 = no.SVector.fromlist(["a", "b", "c"])
  no.log(no.testVec)
  no.log(str(no.testVec))
  no.log(repr(no.testVec))
  no.log(no.testVec2)
  no.log(no.testVec[3])

  no.log(f())

  # test thinning algorithm for non-homogeneous Poisson process
  h = no.DVector.fromlist([0.014] * 10)
  le = no.stopping_nhpp(h, 10000).tolist()
  no.log(sum(le)/len(le))

  # this also tests a zero hazard rate 
  h = no.DVector.fromlist([i/3000 for i in range(100)])
  #no.log(h)
  le = no.stopping_nhpp(h, 10000).tolist()
  no.log(sum(le)/len(le))

  # not convinced this is working correctly
  h = no.DVector.fromlist([0.999, 0.1])
  le = no.stopping_nhpp(h, 1000).tolist()
  no.log(sum(le)/len(le))

  return True
