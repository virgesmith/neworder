import numpy as np
import pandas as pd
import neworder as no

import matplotlib.pyplot as plt

PAUSE=1e-6

TEST_MODE=False#True

TWO_PI = 2 * np.pi


class Boids(no.Model):

  def __init__(self, N):
    super().__init__(no.LinearTimeline(0.0, 0.01), no.MonteCarlo.deterministic_identical_stream)

    self.N = N
    self.speed = 0.5

    self.range = 1.0

    self.vision = 0.2
    self.exclusion = 0.01

    self.sep_max_turn = TWO_PI / 180
    self.align_max_turn = TWO_PI / 90
    self.cohere_max_turn = TWO_PI / 90

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

    self.ax, self.g = self.__init_visualisation()

  def step(self):

    #no.log(self.boids)
    dx, dy, d2 = self.__dists()
    d2e = d2.copy()
    np.fill_diagonal(d2e, np.inf) # distance excluding self
    mindists = np.tile(np.amin(d2e, axis=0), self.N).reshape((self.N, self.N))
    nearest = np.where(mindists == d2e, True, False)
    #print(np.argmin(d2,axis=0))
    #np.savetxt("d2.csv", d2, delimiter=",")

    #print(nearest)
    too_close = np.where(d2 < self.exclusion**2, 1.0, 0.0)
    #print(too_close)
    nearest_in_range = np.logical_and(nearest, too_close).astype(float) #, 1.0, 0.0)
    #no.log(np.sum(nearest_in_range, axis=0))
    # print(nearest_in_range)
    # print(np.sum(nearest_in_range, axis=0))

    self.__separate(nearest_in_range, dx, dy, self.sep_max_turn)

    # this masks the align/cohere steps for boids that need to separate
    mask = np.repeat(1.0-np.sum(nearest_in_range, axis=0), self.N).reshape(self.N, self.N)
    #print(mask)

    in_range = np.where(d2e < self.vision**2, 1.0, 0.0)
    #print(in_range)
    in_range = np.logical_and(in_range, mask).astype(float)
    #print(in_range)
    self.__cohere(in_range, dx, dy, self.cohere_max_turn)
    self.__align(in_range, self.align_max_turn)

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
    #np.fill_diagonal(d2, np.inf) # no self-influence
    return (dx, dy, d2)

  def __align(self, in_range, max_turn):

    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0

    #no.log("weights=%s" % weights)

    # matrix x vector <piecewise-x> vector = vector
    #delta = np.clip(in_range @ self.boids.theta * weights - self.boids.theta, -max_turn, max_turn)
    mean_vx = in_range @ np.cos(self.boids.theta) * weights
    mean_vy = in_range @ np.sin(self.boids.theta) * weights
    # no.log(mean_vx)
    # no.log(mean_vy)
    delta = np.clip(np.arctan2(mean_vy, mean_vx), -max_turn, max_turn)
    #delta = np.arctan2(mean_vy, mean_vx)
    # no.log("delta=%s" % delta)
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

    # delta_y = in_range @ dy @ weights
    # # no.log(self.boids.x.values)
    # self.boids.vx += weight * delta_x / self.timeline().dt()
    # self.boids.vy += weight * delta_y / self.timeline().dt()
    #self.boids.theta = (self.boids.theta + delta) % TWO_PI

  def __track(self):
    """ adjust origin to CoG of flock, has the effect of camera tracking """
    self.boids.x -= self.boids.x.mean() - self.range/2
    self.boids.y -= self.boids.y.mean() - self.range/2


  def __init_visualisation(self):

    fig = plt.figure(constrained_layout=True, figsize=(10,10))
    g = plt.scatter(self.boids.x, self.boids.y, s=20)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.axis("off")

    def on_keypress(event):
      if event.key == "p":
        self.halt()
      # if event.key == "r":
      #   no.run(self)
      elif event.key == "q":
        self.halt()
      else:
       no.log("%s doesnt do anything. p to pause/resume, q to quit" % event.key)

    fig.canvas.mpl_connect('key_press_event', on_keypress)

    return fig, g

  def __update_visualisation(self):

    self.g.set_offsets(np.c_[self.boids.x, self.boids.y])
    # plt.xlim(np.min(self.boids.x), np.max(self.boids.x))
    # plt.ylim(np.min(self.boids.y), np.max(self.boids.y))


    plt.pause(PAUSE)

