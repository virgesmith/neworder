import numpy as np
import pandas as pd
import neworder as no

import matplotlib.pyplot as plt

PAUSE=1e-6

TEST_MODE=False#True

TWO_PI = 2 * np.pi


class Boids(no.Model):

  def __init__(self, N):
    super().__init__(no.LinearTimeline(0.0, 0.01), no.MonteCarlo.nondeterministic_stream)

    self.N = N
    self.speed = 0.5

    self.range = 1.0

    self.vision = 0.2
    self.exclusion = 0.01

    self.sep_max_turn = TWO_PI / 240 # 1.5 degrees
    self.align_max_turn = TWO_PI / 72 # 5 degrees
    self.cohere_max_turn = TWO_PI / 120 # 3 degrees

    if TEST_MODE:
      self.N = 2
      self.boids = pd.DataFrame(
        index = pd.Index(name="id", data=no.df.unique_index(self.N)),
        data={
          "x": [0.01,1],
          "y": [0,1],
          "theta": [np.pi/4, 5*np.pi/4]
        })
    else:
      # initially in [0,1]^3
      self.boids = pd.DataFrame(
        index = pd.Index(name="id", data=no.df.unique_index(self.N)),
        data={
          "x": self.mc().ustream(N) * self.range,
          "y": self.mc().ustream(N) * self.range,
          "theta": self.mc().ustream(N) * TWO_PI # zero is +ve x axis
        }
      )

    self.fig, self.g = self.__init_visualisation()

  def step(self):

    dx, dy, d2 = self.__dists()
    mindists = np.tile(np.amin(d2, axis=0), self.N).reshape((self.N, self.N))
    nearest = np.where(mindists == d2, True, False)

    too_close = np.where(d2 < self.exclusion**2, 1.0, 0.0)
    nearest_in_range = np.logical_and(nearest, too_close).astype(float) #, 1.0, 0.0)

    self.__separate(nearest_in_range, dx, dy, self.sep_max_turn)

    # this masks the align/cohere steps for boids that need to separate
    mask = np.repeat(1.0-np.sum(nearest_in_range, axis=0), self.N).reshape(self.N, self.N)

    in_range = np.logical_and(np.where(d2 < self.vision**2, 1.0, 0.0), mask).astype(float)
    self.__cohere(in_range, dx, dy, self.cohere_max_turn)
    self.__align(in_range, self.align_max_turn)

    # # bounce off walls
    # self.boids.loc[self.boids.x < 0.0, "theta"] = np.pi - self.boids.loc[self.boids.x < 0.0, "theta"]
    # self.boids.loc[self.boids.x > self.range, "theta"] = np.pi - self.boids.loc[self.boids.x > self.range, "theta"]
    # self.boids.loc[self.boids.y < 0.0, "theta"] = 2.0 * np.pi - self.boids.loc[self.boids.y < 0.0, "theta"]
    # self.boids.loc[self.boids.y > self.range, "theta"] = 2.0 * np.pi - self.boids.loc[self.boids.y > self.range, "theta"]

    self.boids.x = (self.boids.x + np.cos(self.boids.theta) * self.speed * self.timeline().dt())
    self.boids.y = (self.boids.y + np.sin(self.boids.theta) * self.speed * self.timeline().dt())

    #self.__track()
    self.__update_visualisation()
    #self.halt()


  def __dists(self):
    """ Compute vector distances and squared distance matrix, adjusted diagonal to remove self-influence """
    dx = self.boids.x.values.repeat(self.N).reshape((self.N,self.N))
    dy = self.boids.y.values.repeat(self.N).reshape((self.N,self.N))
    dx -= dx.T
    dy -= dy.T
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, np.inf) # no self-influence
    return (dx, dy, d2)

  def __align(self, in_range, max_turn):

    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0

    # need to convert to x,y to average angles correctly
    # matrix x vector <piecewise-x> vector = vector
    mean_vx = in_range @ np.cos(self.boids.theta) * weights
    mean_vy = in_range @ np.sin(self.boids.theta) * weights
    delta = np.clip(np.arctan2(mean_vy, mean_vx), -max_turn, max_turn)
    #no.log(delta)
    self.boids.theta = (self.boids.theta + delta) % TWO_PI

  def __cohere(self, in_range, dx, dy, max_turn):
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0
    # compute vectors to cen  tre of gravity of neighbours (if any), relative to boid location
    # matrix * matrix * vector = vector
    xbar = in_range @ dx @ weights
    ybar = in_range @ dy @ weights
    heading = np.clip(np.arctan2(ybar, xbar), -max_turn, max_turn)
    self.boids.theta = (self.boids.theta + heading) % TWO_PI

  def __separate(self, in_range, dx, dy, max_turn):
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0

    x = in_range @ dx @ weights
    y = in_range @ dy @ weights
    heading = np.clip(np.arctan2(y, x), -max_turn, max_turn)
    self.boids.theta = (self.boids.theta - heading) % TWO_PI

  def __track(self):
    """ adjust origin to CoG of flock, has the effect of camera tracking """
    self.boids.x -= self.boids.x.mean() - self.range/2
    self.boids.y -= self.boids.y.mean() - self.range/2


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

