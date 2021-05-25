
import numpy as np
from neworder import domain

import matplotlib.pyplot as plt

space2d = domain.Space(np.array([-1.0, -3.0]), np.array([2.0, 5.0]))

point = np.zeros(2)
delta = np.array([0.6, 0.7])

print(space2d)
print(point)
print(delta)
for _ in range(10):
  point = space2d.move(point, delta)
  print(point)

space2dw = domain.Space(np.array([-1.0, -3.0]), np.array([2.0, 5.0]), True)

point = np.zeros(2)

print(space2dw)
delta = np.array([0.06, 0.07])
x = [point[0]]
y = [point[1]]
for _ in range(100):
  point = space2dw.move(point, delta)
  x.append(point[0])
  y.append(point[1])

# plt.plot(x, y, ".")
# plt.show()

rng = np.random.default_rng(19937)

points = rng.random((5,2))
# print(points)
# #points2 = space2d.move(points, delta)
# #print(points2)
print(space2dw.dists(points))

