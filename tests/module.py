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

  print("[py]", np.array(no.hazard_v(rng1.get(1000)).tolist()).mean())

  return True
