# https://docs.python.org/3/extending/embedding.html

import neworder

def printvec(x):
  neworder.log(x)

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

v = neworder.DVector(20)

v[0] = 3.4
printvec(v)
v[1] = v[0]
printvec(v)
v.clear()
printvec(v)

v2 = neworder.SVector(10)
for i in range(0,10):
  v2[i] = str(i) + " potato"

printvec(v2)

h = neworder.hazard(0.2, 10)
printvec(h)
hv = neworder.hazard_v(neworder.DVector.fromlist([0.1, 0.2, 0.3, 0.4, 0.5]))
printvec(hv)
s = neworder.stopping(0.1, 10)
printvec(s)

