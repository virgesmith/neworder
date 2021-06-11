
import numpy as np
import neworder as no

dim = np.array([3,5])

grid = no.Grid(dim)

# p = np.array([0,0,0])
# i = grid.index(p)
# assert i == 0
# i = grid.index(dim-1)
# print(i)
# assert i == np.prod(dim) - 1

# print(grid.strides)
# print(i)
# print(grid.coord(i))

for i in range(grid.ncells()):
  c = grid.coord(i)
  #print(i, c, grid.index(c))
  assert i == grid.index(c)

print(grid.neighbours(np.array([0,0])))