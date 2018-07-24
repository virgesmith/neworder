# https://docs.python.org/3/extending/embedding.html

def printvec(v):
  print("[python]", type(v), end=":")
  for i in range(0,len(v)):
    print(v[i], end=" ")
  print()


def add(a,b):
    print("[python]: ", a, "+", b)
    return a + b

def sub(a,b):
    print("[python]: ", a, "-", b)
    return a - b

def mul(a,b):
    print("[python]: ", a, "*", b)
    return a * b

def div(a,b):
    print("[python]: ", a, "/", b)
    return a / b


notafunc = 3

import neworder
print("[python] callback:", neworder.name())

v = neworder.dvector(20)

#print(v.size())
v[0] = 3.4
printvec(v)
v[1] = v[0]
printvec(v)
v.clear()
printvec(v)

v2 = neworder.svector(10)
for i in range(0,10):
  v2[i] = str(i) + " potato"

printvec(v2)

h = neworder.hazard(0.2, 10)
printvec(h)
s = neworder.stopping(0.1, 10)
printvec(s)

