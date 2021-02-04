
import numpy as np
import pandas as pd
import neworder as no
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("dark_background")

def _plot(bodies):
  plt.cla()
  plt.scatter(bodies.x, bodies.y, s=bodies.m * 20)
  plt.xlim(-0.2,1.2)
  plt.ylim(-0.2,1.2)


class NBody(no.Model):

  def __init__(self, N, G, dt):
    super().__init__(no.LinearTimeline(0, dt), no.MonteCarlo.deterministic_identical_stream)

    self.G = G
    self.dt = dt
    m_max = 1.0
    x_max = 1.0
    y_max = 1.0

    self.bodies = pd.DataFrame(index=no.df.unique_index(N), data={
        # "m": [0.1,1],
        # "x": [0.3, 0.5],
        # "y": [0.5, 0.5],
        # "vx": [0.0, 0.0],
        # "vy": [0.0, 0.0],
        "m": self.mc().ustream(N) * m_max,
        "x": self.mc().ustream(N) * x_max,
        "y": self.mc().ustream(N) * y_max,
        "vx": 0.0, # 1 * (self.mc().ustream(N) - 0.5),
        "vy": 0.0, #1 * (self.mc().ustream(N) - 0.5),
        "ax": np.nan,
        "ay": np.nan,
        "pe": np.nan
      })

    self.__calc_a()
    #self.bodies.to_csv("bodies0.csv")
    self.E0 = np.sum(self.bodies.pe) # assumes no KE

    _plot(self.bodies)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.pause(0.001)
    #self.debug = pd.DataFrame()

  def __calc_a(self):

    for _,b in self.bodies.iterrows():
      dist2s = (self.bodies.x.values - b.x) ** 2 + (self.bodies.y.values - b.y) ** 2
      # first remove self-interactions
      dist2s = np.where(dist2s == 0.0, np.inf, dist2s)
      # now avoid problems where particles too close
      dist2s += .01
      # TODO avoid massive accelerations
      dists = np.sqrt(dist2s)
      #dists = np.maximum(np.sqrt(dist2s), 0.1)
      xhat = (self.bodies.x.values - b.x) / dists
      yhat = (self.bodies.y.values - b.y) / dists
      #no.log(xhat**2 + yhat**2)
      b.ax = self.G * np.sum(self.bodies.m * xhat / dist2s)
      b.ay = self.G * np.sum(self.bodies.m * yhat / dist2s)
      b.pe = -self.G * b.m * np.sum(self.bodies.m / dists)

    #self.debug = self.debug.append(self.bodies[["x", "y", "vx", "vy", "ax", "ay"]].loc[0])

  def __update_pos(self):
    self.bodies.x += self.bodies.vx * self.dt
    self.bodies.y += self.bodies.vy * self.dt

  def __update_v(self, frac=1):
    dt = self.dt * frac
    self.bodies.vx += self.bodies.ax * dt
    self.bodies.vy += self.bodies.ay * dt

  def check(self):
    # check momentum and energy conservation
    px = np.sum(self.bodies.m * self.bodies.vx)
    py = np.sum(self.bodies.m * self.bodies.vy)
    ke = (self.bodies.m * (self.bodies.vx ** 2 + self.bodies.vy ** 2)).sum() / 2
    pe = np.sum(self.bodies.pe)
    no.log("p=%g,%g" % (px, py))
    no.log("e=%f" % (ke+pe-self.E0))

    return True #np.fabs(ke+pe) < 1

  def step(self):
    self.__update_v(0.5)
    self.__update_pos()
    self.__calc_a()
    self.__update_v(0.5)

    #no.log(self.bodies)
    #self.bodies.to_csv("bodies1.csv")
    #self.halt()

    #no.log(self.bodies)
    _plot(self.bodies)
    plt.pause(0.001)

  def finalise(self):
    #self.debug.to_csv("debug.csv")
    pass
