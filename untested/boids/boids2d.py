import numpy as np
import pandas as pd
import neworder as no
from time import sleep
import matplotlib.pyplot as plt

TEST_MODE=False#True

TWO_PI = 2 * np.pi

class Boids(no.Model):

  def __init__(self, N):
    super().__init__(no.LinearTimeline(0.0, 0.01), no.MonteCarlo.nondeterministic_stream)

    self.N = N
    self.speed = 0.5

    self.range = 1.0

    # continuous wraparound 2d space
    self.domain = no.Space(np.array([0,0]), np.array([self.range, self.range]), edge=no.Domain.WRAP)

    self.vision = 0.2
    self.exclusion = 0.02

    self.sep_max_turn = TWO_PI / 240 # 1.5 degrees
    self.align_max_turn = TWO_PI / 72 # 5 degrees
    self.cohere_max_turn = TWO_PI / 120 # 3 degrees

    if TEST_MODE:
      self.N = 10
      self.boids = pd.DataFrame(
        index = pd.Index(name="id", data=no.df.unique_index(self.N)),
        data={
          "x": np.linspace(0, 0.9, 10),
          "y": np.full(10, 0.5),
          "theta": np.full(10, np.pi/2)
        })
      no.log(self.boids)
    else:
      # initially in [0,1]^dim
      self.boids = pd.DataFrame(
        index = pd.Index(name="id", data=no.df.unique_index(self.N)),
        data={
          "x": self.mc.ustream(N) * self.range,
          "y": self.mc.ustream(N) * self.range,
          "theta": self.mc.ustream(N) * TWO_PI # zero is +ve x axis
        }
      )

    self.fig, self.g = self.__init_visualisation()

  def step(self):
    d2, (dx, dy) = self.domain.dists2((self.boids.x, self.boids.y))
    np.fill_diagonal(d2, np.inf) # no self-influence

    mindists = np.tile(np.amin(d2, axis=0), self.N).reshape((self.N, self.N))
    nearest = np.where(mindists == d2, True, False)

    too_close = np.where(d2 < self.exclusion**2, 1.0, 0.0)
    nearest_in_range = np.logical_and(nearest, too_close).astype(float) #, 1.0, 0.0)

    self.__separate(nearest_in_range, dx, dy, self.sep_max_turn)

    # this masks the align/cohere steps for boids that need to separate
    mask = np.repeat(1.0-np.sum(nearest_in_range, axis=0), self.N).reshape(self.N, self.N)

    in_range = np.logical_and(np.where(d2 < self.vision**2, 1.0, 0.0), mask).astype(float)
    #in_range = np.where(d2 < self.vision**2, 1.0, 0.0).astype(float)
    #self.__cohere(in_range, dx, dy, self.cohere_max_turn)
    self.__align(in_range, self.align_max_turn)

    (self.boids.x, self.boids.y), (vx, vy) = self.domain.move((self.boids.x, self.boids.y),
                                                              (np.cos(self.boids.theta) * self.speed, np.sin(self.boids.theta) * self.speed),
                                                              self.timeline.dt(), ungroup=True)

    #self.__track()
    self.__update_visualisation()
    #sleep(1)
    #self.halt()

  def __align(self, in_range, max_turn):

    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0

    # print(self.boids.theta)
    # print(weights)

    # need to convert to x,y to average angles correctly
    # matrix x vector <piecewise-x> vector = vector
    mean_vx = in_range @ np.cos(self.boids.theta) * weights
    mean_vy = in_range @ np.sin(self.boids.theta) * weights
    #print(mean_vx, mean_vy)
    delta = np.clip(np.arctan2(mean_vy, mean_vx)-self.boids.theta, -max_turn, max_turn)
    self.boids.theta = (self.boids.theta + delta) % TWO_PI

  def __cohere(self, in_range, dx, dy, max_turn):
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0
    # compute vectors to cen  tre of gravity of neighbours (if any), relative to boid location
    # matrix * matrix * vector = vector
    xbar = in_range @ dx @ weights
    ybar = in_range @ dy @ weights
    delta = np.clip(np.arctan2(ybar, xbar) - self.boids.theta, -max_turn, max_turn)
    self.boids.theta = (self.boids.theta + delta) % TWO_PI

  def __separate(self, in_range, dx, dy, max_turn):
    if np.all(in_range == 0): return
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0

    x = in_range @ dx @ weights
    y = in_range @ dy @ weights
    delta = np.clip(np.arctan2(y, x) - self.boids.theta, -max_turn, max_turn)
    self.boids.theta = (self.boids.theta - delta) % TWO_PI

  # def __track(self):
  #   """ adjust origin to CoG of flock, has the effect of camera tracking """
  #   self.boids.x -= self.boids.x.mean() - self.range/2
  #   self.boids.y -= self.boids.y.mean() - self.range/2


  def __init_visualisation(self):

    plt.ion()

    fig = plt.figure(constrained_layout=True, figsize=(8,8))
    g = plt.scatter(self.boids.x, self.boids.y, s=20)
    plt.xlim(0.0, self.range)
    plt.ylim(0.0, self.range)
    plt.axis("off")

    fig.canvas.mpl_connect('key_press_event', lambda event: self.halt() if event.key == "q" else None)

    fig.canvas.flush_events()

    return fig, g

  def __update_visualisation(self):

    self.g.set_offsets(np.c_[self.boids.x, self.boids.y])
    self.fig.canvas.flush_events()

