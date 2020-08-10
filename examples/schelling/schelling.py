
import numpy as np
import pandas as pd
import neworder

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation

class Schelling(neworder.Model):
  def __init__(self, timeline, gridsize, categories, similarity):

    # NB missing this line ended up with memory corruption, bad_alloc thrown
    super().__init__(timeline)

    # category 0 is empty cell
    self.ncategories = len(categories)
    # assign agents proportionally into 1D array for simplicity, then shuffle and reshape
    index = int(0)
    length = np.prod(gridsize)
    self.pop = []
    self.pop.append(np.zeros(length, dtype=int))
    for i in range(1, self.ncategories):
      end = int(index + categories[i] * length)
      #neworder.log("%d:%d" % (index, end))
      self.pop[0][index:end] = i 
      index = end
    #random shuffle
    np.random.shuffle(self.pop[0])
    #reshape to 2D
    self.pop[0].shape = gridsize
    self.cmap = colors.ListedColormap(['white', 'red', 'blue', 'green', 'yellow'][:self.ncategories])

    self.img = plt.imshow(self.pop[0], cmap=self.cmap)
    #self.img.set_data(self.pop[0])
    plt.axis('off')
    plt.pause(0.1)

    self.sat = np.full(gridsize, 0, dtype=int)
    self.similarity = similarity

  def transition(self):
    self.sat.fill(1)

    pop = self.pop[-1].copy()

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


    # for i in range(1, imax):
    #   for j in range(1, jmax):
    #     self.sat[i,j] = ((np.sum(self.pop[i-1:i+2,j-1:j+2] == self.pop[i,j]) - 1) / 8 > self.similarity) or (self.pop[i,j] == 0)
    #plt.imshow(self.sat, cmap=self.cmap)
    #plt.show()

    # enumerate empty cells
    empty = pd.DataFrame(pop).unstack() \
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
      p = pop[unsat.loc[i,"x"], unsat.loc[i, "y"]]
      pop[empty.loc[i,"x"], empty.loc[i, "y"]] = p
      pop[unsat.loc[i,"x"], unsat.loc[i, "y"]] = 0

    self.pop.append(pop)  

    # plt.imshow(self.pop, cmap=self.cmap)
    # plt.pause(0.01)
    #neworder.log(self.pop)
    self.img.set_array(self.pop[-1])
    plt.pause(0.1)

  def checkpoint(self):
    pass
    # self.animate()

  # def __updatefig(self, frameno):
  #   neworder.log("frame=%d" % frameno)
  #   self.im.set_array(self.pop[frameno])
  #   return self.im,

  # def animate(self):
  #   fig = plt.figure()
  #   plt.axis('off')

  #   self.im = plt.imshow(self.pop[0], cmap=self.cmap, animated=True)

  #   anim = animation.FuncAnimation(fig, self.__updatefig, frames=neworder.timeline.index(), interval=100, repeat=True, repeat_delay=3000)
  #   anim.save("./test.gif", dpi=80, writer='imagemagick') 
  #   #plt.show()  
    