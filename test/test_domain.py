
import numpy as np
import neworder as no
import pandas as pd


def test_space2d():

  space2d = no.Space(np.array([-1.0, -3.0]), np.array([2.0, 5.0]))

  point = np.zeros(2)
  delta = np.array([0.6, 0.7])

  print(space2d)
  print(point)
  print(delta)
  for _ in range(10):
    point = space2d.move(point, delta)
    print(point)

  space2dw = no.Space(np.array([-1.0, -3.0]), np.array([2.0, 5.0]), True)

  assert space2dw.dim == 2

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


  points = np.array([[0.,0.], [0., 1.], [1.,1.], [3., 4.]])
  # print(points)
  # #points2 = space2d.move(points, delta)
  # #print(points2)
  print(space2dw.dists(points))


def test_space3d():
  rng = np.random.default_rng(19937)

  N = 5
  bodies = pd.DataFrame(index=no.df.unique_index(N), data={
      "x": rng.random(N) - 0.5,
      "y": rng.random(N) - 0.5,
      "z": rng.random(N) - 0.5,
      "vx": 0.01,
      "vy": 0.01,
      "vz": 0.01
    })

  print(bodies)

  space = no.domain.Space.unbounded(3)

  s = np.column_stack((bodies.x, bodies.y, bodies.z))
  print(space.dists(s))

  print(space.dim)

  dt = 1.0
  bodies.x, bodies.y, bodies.z = space.move((bodies.x, bodies.y, bodies.z), (bodies.vx*dt, bodies.vy*dt, bodies.vz*dt), ungroup=True)
  print(bodies)

  #print(p.shape)
  #print(np.split(p, space.dim, axis=1))


# def test_grid2d():
#   grid2d = no.domain.Grid(np.array([8,8], dtype=np.int64))

#   points = np.array([[0,0], [1,2], [3,4]], dtype=np.int64)

#   to_points = np.array([[0,0]], dtype=np.int64)



#   print(grid2d.dists(points, to_points))
#   #assert False
