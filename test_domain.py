
from neworder.domain import Domain
import numpy as np
import neworder as no
#import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

class SpaceTest(no.Model):

  def __init__(self, n, edge):
    super().__init__(no.LinearTimeline(0, 100), no.MonteCarlo.deterministic_identical_stream)

    self.space2d = no.Space(np.array([-1.0, -3.0]), np.array([2.0, 5.0]), edge)

    self.positions = self.mc().ustream(n * 2).reshape(n, 2)
    self.velocities = (self.mc().ustream(n * 2) - 0.4).reshape(n, 2) 

    self.fig, self.g = self.__init_visualisation()


  def __init_visualisation(self):

    plt.ion()

    fig = plt.figure(constrained_layout=True, figsize=(8,8))
    g = plt.scatter(self.positions[:,0], self.positions[:,1], s=40)
    plt.xlim(self.space2d.min[0], self.space2d.max[0])
    plt.ylim(self.space2d.min[1], self.space2d.max[1])
    #plt.axis("off")

    fig.canvas.mpl_connect('key_press_event', lambda event: self.halt() if event.key == "q" else None)

    fig.canvas.flush_events()

    return fig, g

  def __update_visualisation(self):

    self.g.set_offsets(np.c_[self.positions[:,0], self.positions[:,1]])
    self.fig.canvas.flush_events()

  def step(self):
    self.positions, self.velocities = self.space2d.move(self.positions, self.velocities, 0.1)

    no.log(self.space2d.in_range(1.0, self.positions, count=True))

    self.__update_visualisation()
    sleep(0.01)
    #self.halt()

  def check(self):
    if self.space2d.edge == Domain.WRAP:
      d = (self.space2d.max - self.space2d.min)/2
      return np.all(self.space2d.dists2(self.positions) < d@d)
    return True


if __name__ == "__main__":
  m = SpaceTest(50, Domain.WRAP)
  no.run(m)
