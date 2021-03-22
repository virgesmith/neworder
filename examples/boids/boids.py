import numpy as np
import pandas as pd
import neworder as no

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# for 3d roations see https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
# from numpy import cross, eye, dot
# from scipy.linalg import expm, norm

# def M(axis, theta):
#     return expm(cross(eye(3), axis/norm(axis)*theta))

# v, axis, theta = [3,5,0], [4,4,1], 1.2
# M0 = M(axis, theta)

# print(dot(M0,v))
## [ 2.74911638  4.77180932  1.91629719]


PAUSE=1e-6

TEST_MODE=False#True

class Boids(no.Model):

  def __init__(self, N):
    super().__init__(no.LinearTimeline(0.0, 1.0), no.MonteCarlo.deterministic_identical_stream)

    self.N = N
    self.speed = 0.05

    self.range = 1.0

    self.vision = 0.1
    self.exclusion = 0.01
    self.sep_wt = 1e-3
    self.align_wt = 1e-2
    self.cohere_wt = 1e-4

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
          "vz": 0, #vz - np.mean(vz)
          # "vx": 0.0,
          # "vy": 0.0,
          # "vz": 0.0
        }
      )
    self.__normalise_velocity(self.speed)

    self.ax, self.g = self.__init_visualisation()

  def step(self):

    #no.log(self.boids)
    dx, dy, dz, d2 = self.__dists()
    mindists = np.tile(np.amin(d2,axis=0), self.N).reshape((self.N, self.N))
    nearest = np.where(mindists == d2, True, False)
    #print(np.argmin(d2,axis=0))
    #np.savetxt("d2.csv", d2, delimiter=",")

    #print(nearest)
    too_close = np.where(d2 < self.exclusion**2, 1.0, 0.0)
    #print(too_close)
    nearest_in_range = np.logical_and(nearest, too_close).astype(float) #, 1.0, 0.0)
    #no.log(np.sum(nearest_in_range, axis=0))
    #print(nearest_in_range)
    #print(np.sum(nearest_in_range, axis=0))

    self.__separate(nearest_in_range, dx, dy, dz, self.sep_wt)

    # this masks the align/cohere steps for boids that need to separate
    mask = np.repeat(1.0-np.sum(nearest_in_range, axis=0), self.N).reshape(self.N, self.N)
    #print(mask)

    in_range = np.where(d2 < self.vision**2, 1.0, 0.0)
    #print(in_range)
    in_range = np.logical_and(in_range, mask).astype(float)
    #print(in_range)
    self.__cohere(in_range, dx, dy, dz, self.cohere_wt)
    self.__align(in_range, self.align_wt)

    #no.log(self.boids)

    self.__normalise_velocity(self.speed)
    #assert np.allclose(np.sqrt(self.boids.vx.values ** 2 + self.boids.vy.values ** 2 + self.boids.vz.values ** 2), self.speed)

    self.boids.x = (self.boids.x + self.boids.vx * self.timeline().dt())
    self.boids.y = (self.boids.y + self.boids.vy * self.timeline().dt())
    #self.boids.z = (self.boids.z + self.boids.vz * self.timeline().dt())

    #self.__track()
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

    #no.log(in_range)
    #no.log(weights)

    # matrix x vector <piecewise-x> vector = vector
    vx = in_range @ self.boids.vx.values * weights
    vy = in_range @ self.boids.vy.values * weights
    vz = in_range @ self.boids.vz.values * weights

    self.boids.vx += weight * vx
    self.boids.vy += weight * vy
    #self.boids.vz += weight * vz

  def __cohere(self, in_range, dx, dy, dz, weight):
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0
    # compute vectors to centre of gravity of neighbours (if any)
    # matrix * matrix * vector = vector
    delta_x = in_range @ dx @ weights
    delta_y = in_range @ dy @ weights
    #delta_z = in_range @ dz @ weights
    self.boids.vx -= weight * delta_x / self.timeline().dt()
    self.boids.vy -= weight * delta_y / self.timeline().dt()
    #self.boids.vz -= weight * delta_z / self.timeline().dt()

  def __separate(self, in_range, dx, dy, dz, weight):
    #no.log(in_range)
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights==np.inf] = 0.0
    #no.log(weights)
    # compute vectors to centre of gravity of neighbours (if any)
    delta_x = in_range @ dx @ weights
    delta_y = in_range @ dy @ weights
    #delta_z = in_range @ dz @ weights
    # no.log(self.boids.x.values)
    self.boids.vx += weight * delta_x / self.timeline().dt()
    self.boids.vy += weight * delta_y / self.timeline().dt()
    #self.boids.vz += weight * delta_z / self.timeline().dt()

  def __track(self):
    """ adjust origin to CoG of flock, has the effect of camera tracking """
    self.boids.x -= self.boids.x.mean() - self.range/2
    self.boids.y -= self.boids.y.mean() - self.range/2
    self.boids.z -= self.boids.z.mean() - self.range/2


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

    plt.figure(constrained_layout=True, figsize=(10,10))
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

