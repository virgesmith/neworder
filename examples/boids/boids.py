import numpy as np
import pandas as pd
import neworder as no

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PAUSE=1e-6

TEST_MODE=False #True

class Boids(no.Model):

  def __init__(self, N):
    super().__init__(no.LinearTimeline(0.0, 1.0), no.MonteCarlo.deterministic_identical_stream)

    self.N = N
    self.speed = 0.01

    self.range = 1.0

    if TEST_MODE:
      self.N = 2
      self.boids = pd.DataFrame(
        index = pd.Index(name="id", data=no.df.unique_index(self.N)),
        data={
          "x": [0.01,1],
          "y": [0,1],
          "z": 0.0,
          "vx": [0.5,-0.5],
          "vy": [0.5,-0.5],
          "vz": 0.0
        })
    else:
      # initially in [0,1]^3
      vx = self.mc().ustream(N)
      vy = self.mc().ustream(N)
      vz = self.mc().ustream(N)
      self.boids = pd.DataFrame(
        index = pd.Index(name="id", data=no.df.unique_index(self.N)),
        data={
          "x": self.mc().ustream(N) * self.range,
          "y": self.mc().ustream(N) * self.range,
          "z": 0.0, #self.mc().ustream(N) * self.range,
          "vx": vx - np.mean(vx), # ensure v is balanced
          "vy": vy - np.mean(vy),
          "vz": 0.0 #vz - np.mean(vz)
        }
      )
    self.__normalise_velocity(self.speed)
    #no.log(np.sqrt(self.boids.vx ** 2 + self.boids.vy ** 2 + self.boids.vz ** 2))

    self.ax, self.g = self.__init_visualisation()

  def step(self):

    vision = 0.5
    exclusion = 0.1

    sep_wt = 1e-5
    align_wt = 1e-5
    cohere_wt = 1e-5

    #no.log(self.boids)

    # TODO can just work on v not pos?

    dx, dy, dz, d2 = self.__dists()
    #no.log(d2)
    nearest = np.where(np.amin(d2, axis=0) == d2, True, False)
    #no.log(nearest)
    #self.halt()
    nearest_in_range = np.where(np.logical_and(nearest, d2 < exclusion**2), 1.0, 0.0)
    #no.log(in_range)

    self.__separate(nearest_in_range, dx, dy, dz, sep_wt)
    #in_range = np.where(np.logical_and(np.logical_not(in_range), np.logical_and(d2 < vision**2, d2 >= exclusion**2)), 1.0, 0.0)
    in_range = np.where(np.logical_and(d2 < vision**2, d2 >= exclusion**2), 1.0, 0.0)
    self.__align(in_range, align_wt)
    self.__cohere(in_range, dx, dy, dz, cohere_wt)

    #no.log(self.boids)

    self.__normalise_velocity(self.speed)

    self.boids.x = (self.boids.x + self.boids.vx * self.timeline().dt())
    self.boids.y = (self.boids.y + self.boids.vy * self.timeline().dt())
    #self.boids.z = (self.boids.z + self.boids.vz * self.timeline().dt())

    self.__track()
    self.__update_visualisation()
    #self.halt()


  def __dists(self):
    """ Compute vector distances and squared distance matrix, adjusted diagonal to remove self-influence """
    dx = self.boids.x.values.repeat(self.N).reshape((self.N,self.N))
    dy = self.boids.y.values.repeat(self.N).reshape((self.N,self.N))
    dz = self.boids.z.values.repeat(self.N).reshape((self.N,self.N))
    dx -= dx.T
    dy -= dy.T
    dz -= dz.T
    d2 = dx**2 + dy**2 + dz**2
    np.fill_diagonal(d2, np.inf) # no self-influence
    return (dx, dy, dz, d2)

  def __normalise_velocity(self, speed):
    f = speed / np.sqrt(self.boids.vx ** 2 + self.boids.vy ** 2 + self.boids.vz ** 2)
    self.boids.vx *= f
    self.boids.vy *= f
    self.boids.vz *= f

  def __align(self, in_range, weight):

    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0

    vx = in_range @ self.boids.vx.values @ weights
    vy = in_range @ self.boids.vy.values @ weights
    vz = in_range @ self.boids.vz.values @ weights
    # no.log(vx)
    # no.log(vy)
    # no.log(vz)

    self.boids.vx += weight * vx
    self.boids.vy += weight * vy
    #self.boids.vz += weight * vz
    #self.halt()

  def __cohere(self, in_range, dx, dy, dz, weight):
    #no.log(in_range)
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0
    #no.log(weights)
    # compute vectors to centre of gravity of neighbours (if any)
    delta_x = in_range @ dx @ weights
    delta_y = in_range @ dy @ weights
    delta_z = in_range @ dz @ weights
    # no.log(self.boids.x.values)
    self.boids.vx -= weight * delta_x / self.timeline().dt()
    self.boids.vy -= weight * delta_y / self.timeline().dt()
    self.boids.vz -= weight * delta_z / self.timeline().dt()

  def __separate(self, in_range, dx, dy, dz, weight):
    #no.log(in_range)
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0
    #no.log(weights)
    # compute vectors to centre of gravity of neighbours (if any)
    delta_x = in_range @ dx @ weights
    delta_y = in_range @ dy @ weights
    delta_z = in_range @ dz @ weights
    # no.log(self.boids.x.values)
    self.boids.vx += weight * delta_x / self.timeline().dt()
    self.boids.vy += weight * delta_y / self.timeline().dt()
    self.boids.vz += weight * delta_z / self.timeline().dt()

  def __track(self):
    """ adjust origin to CoG of flock, has the effect of camera tracking """
    self.boids.x -= self.boids.x.mean() - self.range/2
    self.boids.y -= self.boids.y.mean() - self.range/2


  def __init_visualisation(self):
    # https://stackoverflow.com/questions/41602588/matplotlib-3d-scatter-animations
    #plt.style.use('dark_background')
    # axes instance
    # fig = plt.figure(figsize=(6,6))
    # ax = Axes3D(fig)
    # g = ax.scatter(self.boids.x, self.boids.y, self.boids.z, s=20) #c=self.boids.index.values,
    # ax.set_xlim(-self.range, 2*self.range)
    # ax.set_ylim(-self.range, 2*self.range)
    # ax.set_zlim(-self.range, 2*self.range)
    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])
    # ax.set_axis_off()
    # plt.pause(PAUSE)
    # return ax, g

    plt.figure(figsize=(6,6))
    g = plt.scatter(self.boids.x, self.boids.y, s=20)
    plt.xlim(-2, 3)
    plt.ylim(-2, 3)
    return None, g


  def __update_visualisation(self):
    # self.ax.set_xlim(np.min(self.boids.x), np.max(self.boids.x))
    # self.ax.set_ylim(np.min(self.boids.y), np.max(self.boids.y))
    # self.ax.set_zlim(np.min(self.boids.z), np.max(self.boids.z))
    #self.g._offsets3d = (self.boids.x, self.boids.y, self.boids.z)
    #plt.savefig("/tmp/n-body%04d.png" % i)


    self.g.set_offsets(np.c_[self.boids.x, self.boids.y])
    # plt.xlim(np.min(self.boids.x), np.max(self.boids.x))
    # plt.ylim(np.min(self.boids.y), np.max(self.boids.y))


    plt.pause(PAUSE)

