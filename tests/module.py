"""
Test framework expects modules with a function called test that
- takes no arguments
- returns True on success and False on failure
"""
import neworder as no
import numpy as np

def test():

  dv = no.DVector(10)

  rng0 = no.UStream(0)
  #print(rng0.get(10).tolist())
  rng1 = no.UStream(0)
  #print(rng1.get(10).tolist())

  # test arithmetic
  dv = dv + 0.5
  dv = 0.5 + dv
  #dv = dv + dv
  dv = 0.5 * dv
  dv = dv * 0.5
  # dv = dv / 0.w5

  print("[py]", np.array(no.hazard_v(rng1.get(1000)).tolist()).mean())

  f = no.Callback("2 + 2")

  no.testVec = no.DVector.fromlist([1,2,3,4])
  print(no.testVec[3])

  print(f())

  return True
