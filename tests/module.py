"""
Test framework expects modules with a function called test that
- takes no arguments
- returns True on success and False on failure
"""
import neworder as no

def test():

  dv = no.dvector(10)

  rng0 = no.ustream(0)
  print(rng0.get(10).tolist())
  rng1 = no.ustream(0)
  print(rng1.get(10).tolist())

  return True
