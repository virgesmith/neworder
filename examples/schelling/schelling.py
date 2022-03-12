import numpy as np
import neworder

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.image import AxesImage # type: ignore
from matplotlib import colors  # type: ignore

class Schelling(neworder.Model):
  def __init__(self,
               timeline: neworder.Timeline,
               gridsize: tuple[int, int],
               categories: np.ndarray[np.float64, np.dtype[np.float64]],
               similarity: float) -> None:
    # NB missing this line can cause memory corruption
    super().__init__(timeline, neworder.MonteCarlo.deterministic_identical_stream)

    # category 0 is empty cell
    self.ncategories = len(categories)
    # randomly sample initial population according to category weights
    init_pop = self.mc.sample(np.prod(gridsize), categories).reshape(gridsize)
    self.sat = np.empty(gridsize, dtype=int)
    self.similarity = similarity

    self.domain = neworder.StateGrid(init_pop, neworder.Edge.CONSTRAIN)

    self.fig, self.img = self.__init_visualisation()

  def step(self) -> None:

    # start with empty cells being satisfied
    self.sat = (self.domain.state == 0)

    # !count!
    # count all neighbours, scaling by acceptable similarity ratio
    n_any = self.domain.count_neighbours(lambda x: x>0) * self.similarity

    for c in range(1,self.ncategories):
      # count neighbour with a specific state
      n_cat = self.domain.count_neighbours(lambda x: x==c)
      self.sat = np.logical_or(self.sat, np.logical_and(n_cat > n_any, self.domain.state == c))
    # !count!

    n_unsat = np.sum(~self.sat)

    pop = self.domain.state.copy()

    free = list(zip(*np.where(pop == 0)))
    for src in zip(*np.where(~self.sat)):
      # pick a random destination
      r = self.mc.raw() % len(free)
      dest = free[r]
      pop[dest] = pop[src]
      pop[src] = 0
      free[r] = src

    self.domain.state = pop

    neworder.log("step %d %.4f%% unsatisfied" % (self.timeline.index(), 100.0 * n_unsat / pop.size))

    self.__update_visualisation()

    # !halt!
    # finish early if everyone satisfied
    if n_unsat == 0:
      # set the halt flag in the runtime
      self.halt()
      # since the timeline is open-ended we need to explicitly call finalise
      self.finalise()
    # !halt!

  def finalise(self) -> None:
    plt.pause(5.0)

  def __init_visualisation(self) -> tuple[plt.Figure, AxesImage]:
    plt.ion()

    cmap = colors.ListedColormap(['white', 'red', 'blue', 'green', 'yellow'][:self.ncategories])

    fig = plt.figure(constrained_layout=True, figsize=(8,6))
    img = plt.imshow(self.domain.state.T, cmap=cmap)
    plt.axis('off')
    fig.canvas.mpl_connect('key_press_event', lambda event: self.halt() if event.key == "q" else None)
    fig.canvas.flush_events()

    return fig, img

  def __update_visualisation(self) -> None:
    self.img.set_array(self.domain.state.T)
    # plt.savefig("/tmp/schelling%04d.png" % self.timeline.index(), dpi=80)
    self.fig.canvas.flush_events()
