
import numpy as np
import pandas as pd
import neworder

import matplotlib.pyplot as plt
from matplotlib import colors

class Schelling():
  def __init__(self, gridsize, categories, similarity):

    # category 0 is empty cell
    self.ncategories = len(categories)
    # randomly assign agents into 1D array for simplicity
    index = int(0)
    length = np.prod(gridsize)
    self.pop = np.ndarray(length, dtype=int)
    for i in range(self.ncategories):
      end = int(index + categories[i] * length)
      neworder.log("%d:%d" % (index, end))
      self.pop[index:end] = i 
      index = end
    # random shuffle
    np.random.shuffle(self.pop)
    # reshape to 2D
    self.pop.shape = gridsize
    self.cmap = colors.ListedColormap(['white', 'red', 'blue', 'green', 'yellow'][:self.ncategories])
    plt.imshow(self.pop, cmap=self.cmap)

    self.sat = np.full(gridsize, 0, dtype=int)
    self.similarity = similarity

  def step(self):
    self.sat.fill(1)

    # corners
    self.sat[0,  0] = ((np.sum(self.pop[:2,  :2] == self.pop[0,  0]) - 1) / 3 > self.similarity) or (self.pop[0,  0] == 0)
    self.sat[-1, 0] = ((np.sum(self.pop[-2:, :2] == self.pop[-1, 0]) - 1) / 3 > self.similarity) or (self.pop[-1, 0] == 0)
    self.sat[0, -1] = ((np.sum(self.pop[:2, -2:] == self.pop[0, -1]) - 1) / 3 > self.similarity) or (self.pop[0, -1] == 0)
    self.sat[-1,-1] = ((np.sum(self.pop[-2:,-2:] == self.pop[-1,-1]) - 1) / 3 > self.similarity) or (self.pop[-1,-1] == 0)

    imax = self.pop.shape[0] - 1
    jmax = self.pop.shape[1] - 1

    # edges
    for i in range(1, imax):
      self.sat[i, 0] = ((np.sum(self.pop[i-1:i+2,:2] == self.pop[i,0]) - 1) / 5 > self.similarity) or (self.pop[i,0] == 0)
      self.sat[i,-1] = ((np.sum(self.pop[i-1:i+2,-2:] == self.pop[i,-1]) - 1) / 5 > self.similarity) or (self.pop[i,-1] == 0)
    for j in range(1, jmax):
      self.sat[0, j] = ((np.sum(self.pop[:2,j-1:j+1] == self.pop[0,  j]) - 1) / 5 > self.similarity) or (self.pop[0, j] == 0)
      self.sat[-1,j] = ((np.sum(self.pop[-2:j-1:j+2] == self.pop[-1, j]) - 1) / 5 > self.similarity) or (self.pop[-1, j] == 0)

    # interior
    for i in range(1, imax):
      for j in range(1, jmax):
        self.sat[i,j] = ((np.sum(self.pop[i-1:i+2,j-1:j+2] == self.pop[i,j]) - 1) / 8 > self.similarity) or (self.pop[i,j] == 0)


    for i in range(1, imax):
      for j in range(1, jmax):
        self.sat[i,j] = ((np.sum(self.pop[i-1:i+2,j-1:j+2] == self.pop[i,j]) - 1) / 8 > self.similarity) or (self.pop[i,j] == 0)
    #plt.imshow(self.sat, cmap=self.cmap)
    #plt.show()

    # enumerate empty cells
    empty = pd.DataFrame(self.pop).unstack() \
            .to_frame().reset_index() \
            .rename({"level_0": "y", "level_1": "x", 0: "occ"}, axis=1) 
    # sample randomly empty cells only 
    empty = empty[empty['occ'] == 0].sample(frac=1).reset_index(drop=True)
    #print(empty.head())

    # enumerate unsatisfied
    unsat = pd.DataFrame(self.sat).unstack() \
              .to_frame().reset_index() \
              .rename({"level_0": "y", "level_1": "x", 0: "sat"}, axis=1) 
    # sample randomly unstaisfied only
    unsat = unsat[unsat['sat'] == False]
    if len(unsat):
      unsat = unsat.sample(frac=1).reset_index(drop=True)

    #print(unsat.head())
    neworder.log("%d unsatisfied, %d empty" % (len(unsat), len(empty)))

    # move unsatisfied to empty
    for i in range(min(len(unsat), len(empty))):
      p = self.pop[unsat.ix[i,"x"], unsat.ix[i, "y"]]
      #neworder.log("(%d,%d)=%d" % (unsat.ix[i,"x"], unsat.ix[i, "y"], p))
      self.pop[empty.ix[i,"x"], empty.ix[i, "y"]] = p
      self.pop[unsat.ix[i,"x"], unsat.ix[i, "y"]] = 0
      #self.pop[]

    plt.imshow(self.pop, cmap=self.cmap)
    plt.pause(0.1)
    #neworder.log(self.pop)
    #print(sat_df)

  def stats(self):
    pass

    