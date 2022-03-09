import numpy as np
import neworder as no
import matplotlib.pyplot as plt
from matplotlib import colors


class Conway(no.Model):

  __glider = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=int)

  def __init__(self, nx, ny, n, edge=no.Edge.WRAP):
    super().__init__(no.LinearTimeline(0, 1), no.MonteCarlo.nondeterministic_stream)

    # create n automata at random positions
    rng = np.random.default_rng(self.mc.raw())
    s = rng.choice(np.arange(nx * ny), n, replace=False)

    init_state = np.zeros((nx * ny))
    for s in s:
     init_state[s] = 1

    self.domain = no.StateGrid(init_state.reshape(ny, nx), edge=edge)

    self.domain.state[20:23, 20:23] = Conway.__glider

    self.fig, self.g = self.__init_visualisation()

  # !step!
  def step(self):
    n = self.domain.count_neighbours(lambda x: x > 0)

    deaths = np.logical_or(n < 2, n > 3)
    births = n == 3

    self.domain.state = self.domain.state * ~deaths + births

    self.__update_visualisation()
  # !step!

  def check(self):
    # # randomly place a glider (not across edge)
    # if self.timeline.index() == 0:
    #   x = self.mc.raw() % (self.domain.state.shape[0] - 2)
    #   y = self.mc.raw() % (self.domain.state.shape[1] - 2)
    #   self.domain.state[x:x+3, y:y+3] = np.rot90(Conway.__glider, self.mc.raw() % 4)
    return True

  def __init_visualisation(self):
    plt.ion()
    cmap = colors.ListedColormap(['black', 'white', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown'])
    fig = plt.figure(constrained_layout=True, figsize=(8, 6))
    g = plt.imshow(self.domain.state, cmap=cmap, vmax=9)
    plt.axis("off")

    fig.canvas.mpl_connect('key_press_event', lambda event: self.halt() if event.key == "q" else None)
    fig.canvas.flush_events()

    return fig, g

  def __update_visualisation(self):
    self.g.set_data(self.domain.state)
    # plt.savefig("/tmp/conway%04d.png" % self.timeline.index(), dpi=80)
    # if self.timeline.index() > 100:
    #   self.halt()

    self.fig.canvas.flush_events()
