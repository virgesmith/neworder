
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
    self.cmap = colors.ListedColormap(['white', 'red', 'blue'])
    plt.imshow(self.pop, cmap=self.cmap)
    #plt.show()

    self.sat = np.full(gridsize, 0, dtype=int)
    self.similarity = similarity

  def step(self):
    # TODO edges
    self.sat.fill(1)
    for i in range(1, self.pop.shape[0] - 1):
      for j in range(1, self.pop.shape[1] - 1):
        self.sat[i,j] = ((np.sum(self.pop[i-1:i+2,j-1:j+2] == self.pop[i,j]) - 1) / 8 > self.similarity) or (self.pop[i,j] == 0)
    #plt.imshow(self.sat, cmap=self.cmap)
    #plt.show()

    # enumerate empty cells
    empty = pd.DataFrame(self.pop).unstack() \
            .to_frame().reset_index() \
            .rename({"level_0": "x", "level_1": "y", 0: "occ"}, axis=1) 
    # sample randomly empty cells only 
    empty = empty[empty['occ'] == 0].sample(frac=1).reset_index(drop=True)
    #print(empty.head())

    # enumerate unsatisfied
    unsat = pd.DataFrame(self.sat).unstack() \
              .to_frame().reset_index() \
              .rename({"level_0": "x", "level_1": "y", 0: "sat"}, axis=1) 
    # sample randomly unstaisfied only
    unsat = unsat[unsat['sat'] == False]
    if len(unsat):
      unsat = unsat.sample(frac=1).reset_index(drop=True)

    #print(unsat.head())
    neworder.log("%d unsatisfied, %d empty" % (len(unsat), len(empty)))

    # move unsatisfied to empty
    for i in range(min(len(unsat), len(empty))):
      p = self.pop[unsat.ix[i,"y"], unsat.ix[i, "x"]]
      #neworder.log("(%d,%d)=%d" % (unsat.ix[i,"x"], unsat.ix[i, "y"], p))
      self.pop[empty.ix[i,"y"], empty.ix[i, "x"]] = p
      self.pop[unsat.ix[i,"y"], unsat.ix[i, "x"]] = 0
      #self.pop[]

    plt.imshow(self.pop, cmap=self.cmap)
    plt.pause(0.1)
    #neworder.log(self.pop)
    #print(sat_df)

  def stats(self):
    pass

    