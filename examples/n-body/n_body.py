
import numpy as np
import pandas as pd
import neworder as no
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NBody(no.Model):

  def __init__(self, N, G, dt):
    super().__init__(no.LinearTimeline(0, dt), no.MonteCarlo.deterministic_identical_stream)

    self.G = G
    self.dt = dt
    m_max = 1.0
    x_max = 1.0
    y_max = 1.0

    r = self.mc().ustream(N) - 0.5
    theta = self.mc().ustream(N) * np.pi
    x = r * np.cos(theta) * x_max
    y = r * np.sin(theta) * y_max

    self.bodies = pd.DataFrame(index=no.df.unique_index(N), data={
        "m": self.mc().ustream(N) * m_max,
        "x": x,
        "y": y,
        "z": 0.1*(self.mc().ustream(N) - 0.5),
        # create angular momentum
        "vx": -y,
        "vy": x,
        "vz": 0.0,
        "ax": np.nan,
        "ay": np.nan,
        "az": np.nan,
        "ke": np.nan,
        "pe": np.nan
      })

    # balance momentum by changing velocity of body 0
    self.bodies.vx[0] -= (self.bodies.m * self.bodies.vx).sum() / self.bodies.m[0]
    self.bodies.vy[0] -= (self.bodies.m * self.bodies.vy).sum() / self.bodies.m[0]
    self.bodies.vz[0] -= (self.bodies.m * self.bodies.vz).sum() / self.bodies.m[0]

    self.__calc_a()
    self.E0 = np.sum(self.bodies.pe + self.bodies.ke)

    self.g = self.__init_visualisation()

  # also calc energy of system
  def __calc_a(self):
    n = len(self.bodies)
    dx = self.bodies.x.values.repeat(n).reshape((n,n))
    dy = self.bodies.y.values.repeat(n).reshape((n,n))
    dz = self.bodies.z.values.repeat(n).reshape((n,n))
    dist2s = (dx - dx.T)**2 + (dy - dy.T)**2 + (dz - dz.T)**2 #fudge factor
    dist2s = np.where(dist2s == 0.0, np.inf, dist2s)
    dist2s += 0.01
    dists = np.sqrt(dist2s)
    for i,b in self.bodies.iterrows():
      # dist2s = (self.bodies.x.values - b.x) ** 2 \
      #        + (self.bodies.y.values - b.y) ** 2 \
      #        + (self.bodies.z.values - b.z) ** 2
      # first remove self-interactions
      #dist2s = np.where(dist2s == 0.0, np.inf, dist2s)
      # fudge to avoid acceleration spikes when particles too close
      #dist2s += .01
      xhat = (self.bodies.x.values - b.x) / dists[i]
      yhat = (self.bodies.y.values - b.y) / dists[i]
      zhat = (self.bodies.z.values - b.z) / dists[i]
      b.ax = self.G * np.sum(self.bodies.m * xhat / dist2s[i])
      b.ay = self.G * np.sum(self.bodies.m * yhat / dist2s[i])
      b.az = self.G * np.sum(self.bodies.m * zhat / dist2s[i])
      b.ke = b.m * (b.vx ** 2 + b.vy ** 2 + b.vz ** 2)
      b.pe = -self.G * b.m * np.sum(self.bodies.m / dists[i])

  def __update_pos(self):
    self.bodies.x += self.bodies.vx * self.dt
    self.bodies.y += self.bodies.vy * self.dt
    self.bodies.z += self.bodies.vz * self.dt

  def __update_v(self, frac=1):
    dt = self.dt * frac
    self.bodies.vx += self.bodies.ax * dt
    self.bodies.vy += self.bodies.ay * dt
    self.bodies.vz += self.bodies.az * dt

  def check(self):
    # check momentum and energy conservation
    px = np.sum(self.bodies.m * self.bodies.vx)
    py = np.sum(self.bodies.m * self.bodies.vy)
    pz = np.sum(self.bodies.m * self.bodies.vz)
    ke = np.sum(self.bodies.ke)
    pe = np.sum(self.bodies.pe)
    no.log("p=%g,%g,%g" % (px, py, pz))
    no.log("delta E=%f" % (ke+pe-self.E0))

    return np.fabs(ke+pe-self.E0) < 20.2

  def step(self):
    # 2nd order accurate, see https://medium.com/swlh/create-your-own-n-body-simulation-with-python-f417234885e9
    self.__update_v(0.5)
    self.__update_pos()
    self.__calc_a()
    self.__update_v(0.5)

    #_plot(self.bodies)
    self.__update_visualisation()

  def __init_visualisation(self):
    # https://stackoverflow.com/questions/41602588/matplotlib-3d-scatter-animations
    plt.ion()
    plt.style.use('dark_background')
    # axes instance
    self.fig = plt.figure(figsize=(8,8))
    self.fig.suptitle("[q to quit]", y=0.05, x= 0.05)
    ax = Axes3D(self.fig)
    self.ax = ax

    g = ax.scatter(self.bodies.x, self.bodies.y, self.bodies.z, c=self.bodies.index.values, s=self.bodies.m * 5000 / self.bodies.m.sum())
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.5,0.5)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_axis_off()
    self.fig.canvas.flush_events()

    def on_keypress(event):
      if event.key == "q":
        self.halt()
    self.fig.canvas.mpl_connect('key_press_event', on_keypress)

    return g

  def __update_visualisation(self):
    self.g._offsets3d = (self.bodies.x, self.bodies.y, self.bodies.z)
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()
    #plt.savefig("/tmp/n-body%04d.png" % i)

