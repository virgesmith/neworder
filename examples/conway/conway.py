import numpy as np
import neworder as no
import matplotlib.pyplot as plt

class Conway(no.Model):

  def __init__(self, nx, ny, n, edge=no.Domain.WRAP):
    super().__init__(no.LinearTimeline(0, 1), no.MonteCarlo.deterministic_identical_stream)

    # create n automata at random positions
    rng = np.random.default_rng(self.mc.raw())
    s = rng.choice(np.arange(nx*ny), n, replace=False)

    init_state = np.zeros((nx*ny))
    for s in s:
     init_state[s] = 1
    #for i in s:

    self.domain = no.Grid(init_state.reshape(ny, nx), edge=edge)

    self.fig, self.g = self.__init_visualisation()

  def step(self):

    self.domain_old = self.domain

    n = self.domain.count_neighbours(lambda x: x > 0)

    deaths = 1 - np.logical_or(n < 2, n > 3)
    births = 0 + (n == 3)

    self.domain.state = self.domain.state * deaths + births

    self.__update_visualisation()

  def check(self):
    return True

  def __init_visualisation(self):

    plt.ion()

    fig = plt.figure(constrained_layout=True, figsize=(12,9))
    g = plt.imshow(self.domain.state, cmap='hot', vmax=8)
    plt.axis("off")

    fig.canvas.mpl_connect('key_press_event', lambda event: self.halt() if event.key == "q" else None)
    fig.canvas.flush_events()

    return fig, g

  def __update_visualisation(self):

    self.g.set_data(self.domain.state)
    self.fig.canvas.flush_events()
