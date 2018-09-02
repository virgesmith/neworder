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

neworder.log("callback: %s" % neworder.name())

v = np.zeros(20, dtype=float)

v[0] = 3.4
neworder.log(v)

# use "object" for strings, str will do char 
v2 = np.empty(10, dtype=object)
for i in range(0,10):
  v2[i] = str(i) + " potato"

neworder.log(v2)

h = neworder.hazard(0.2, 10)
neworder.log(h)
hv = neworder.hazard_v(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
neworder.log(hv)
s = neworder.stopping(0.1, 10)
neworder.log(s)
sv = neworder.stopping_v(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
neworder.log(sv)
nhpp = neworder.stopping_nhpp(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 10)
neworder.log(nhpp)
