
import numpy as np
import pandas as pd
import neworder

import matplotlib.pyplot as plt
from matplotlib import colors

class Schelling(neworder.Model):
  def __init__(self, timeline, gridsize, categories, similarity):
    # NB missing this line can cause memory corruption
    super().__init__(timeline, neworder.MonteCarlo.deterministic_identical_stream)

    # category 0 is empty cell
    self.ncategories = len(categories)
    # randomly sample initial population according to category weights
    self.pop = self.mc().sample(np.prod(gridsize), categories)
    #reshape to 2D
    self.pop.shape = gridsize
    self.cmap = colors.ListedColormap(['white', 'red', 'blue', 'green', 'yellow'][:self.ncategories])

    self.img = plt.imshow(self.pop, cmap=self.cmap)
    plt.axis('off')
    plt.pause(0.1)

    self.sat = np.full(gridsize, 0, dtype=int)
    self.similarity = similarity

  def step(self):
    self.sat.fill(1)

    pop = self.pop.copy()

    # counting only occupied cells can result in div by zero so rearrange similarity check to avoid division
    # corners
    nocc =  np.sum(pop[:2,  :2] != 0) - 1
    nsame = np.sum(pop[:2,  :2] == pop[0,  0]) - 1
    self.sat[0,  0] = (nsame > nocc * self.similarity) or (pop[0,  0] == 0)
    nocc =  np.sum(pop[-2:, :2] != 0) - 1
    nsame = np.sum(pop[-2:, :2] == pop[-1,  0]) - 1
    self.sat[-1, 0] = (nsame > nocc * self.similarity) or (pop[-1, 0] == 0)
    nocc =  np.sum(pop[:2, -2:] != 0) - 1
    nsame = np.sum(pop[:2, -2:] == pop[0,  -1]) - 1
    self.sat[0, -1] = (nsame > nocc * self.similarity) or (pop[0, -1] == 0)
    nocc =  np.sum(pop[-2:,-2:] != 0) - 1
    nsame = np.sum(pop[-2:,-2:] == pop[-1,  -1]) - 1
    self.sat[-1,-1] = (nsame > nocc * self.similarity) or (pop[-1,-1] == 0)

    imax = pop.shape[0] - 1
    jmax = pop.shape[1] - 1

    # edges
    for i in range(1, imax):
      nocc =  np.sum(pop[i-1:i+2, :2] != 0) - 1
      nsame = np.sum(pop[i-1:i+2, :2] == pop[i, 0]) - 1
      self.sat[i, 0] = (nsame > nocc * self.similarity) or (pop[i, 0] == 0)
      nocc =  np.sum(pop[i-1:i+2,-2:] != 0) - 1
      nsame = np.sum(pop[i-1:i+2,-2:] == pop[i,-1]) - 1
      self.sat[i,-1] = (nsame > nocc * self.similarity) or (pop[i,-1] == 0)

    for j in range(1, jmax):
      nocc =  np.sum(pop[:2, j-1:j+2] != 0) - 1
      nsame = np.sum(pop[:2, j-1:j+2] == pop[0,  j]) - 1
      self.sat[0, j] = (nsame > nocc * self.similarity) or (pop[0, j] == 0)
      nocc =  np.sum(pop[-2:,j-1:j+2] != 0) - 1
      nsame = np.sum(pop[-2:,j-1:j+2] == pop[-1,j]) - 1
      self.sat[-1,j] = (nsame > nocc * self.similarity) or (pop[-1, j] == 0)

    # interior
    for i in range(1, imax):
      for j in range(1, jmax):
        nocc = np.sum(pop[i-1:i+2,j-1:j+2] != 0) - 1
        nsame = np.sum(pop[i-1:i+2,j-1:j+2] == pop[i,j]) - 1
        self.sat[i,j] = (nsame > nocc * self.similarity) or (pop[i,j] == 0)

    # enumerate empty cells
    empty = pd.DataFrame(pop).unstack() \
            .to_frame().reset_index() \
            .rename({"level_0": "y", "level_1": "x", 0: "occ"}, axis=1)
    # !sample!
    # sample randomly empty cells only (seeding with neworder's mc engine to ensure reproduciblity)
    empty = empty[empty['occ'] == 0].sample(frac=1, random_state=self.mc().raw() % 2**32).reset_index(drop=True)
    # !sample!

    # enumerate unsatisfied
    unsat = pd.DataFrame(self.sat).unstack() \
              .to_frame().reset_index() \
              .rename({"level_0": "y", "level_1": "x", 0: "sat"}, axis=1)
    # sample randomly unsatisfied only
    unsat = unsat[unsat['sat'] == False]
    if len(unsat):
      # sample randomly empty cells only (seeding with neworder's mc engine to ensure reproduciblity)
      unsat = unsat.sample(frac=1, random_state=self.mc().raw() % 2**32).reset_index(drop=True)

    neworder.log("step %d %.2f%% unsatisfied" % (self.timeline().index(), 100.0 * len(unsat) / pop.size))

    # move unsatisfied to empty
    for i in range(min(len(unsat), len(empty))):
      p = pop[unsat.loc[i,"x"], unsat.loc[i, "y"]]
      pop[empty.loc[i,"x"], empty.loc[i, "y"]] = p
      pop[unsat.loc[i,"x"], unsat.loc[i, "y"]] = 0

    self.pop = pop

    self.img.set_array(self.pop)
    plt.pause(0.01) # allow the image to update

    # !halt!
    # finish early if everyone satisfied
    if len(unsat) == 0:
      self.halt()
    # !halt!

  def checkpoint(self):
    pass
