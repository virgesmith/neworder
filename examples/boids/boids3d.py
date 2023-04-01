from __future__ import annotations
from typing import Any
from time import sleep
import numpy as np
import pandas as pd # type: ignore
import neworder as no
import matplotlib.pyplot as plt # type: ignore


class Boids3d(no.Model):

  ALIGN_COEFF = 2
  COHERE_COEFF = 0.02
  SEPARATE_COEFF = 0.2

  def __init__(self, N: int, range: float, vision: float, exclusion: float, speed: float) -> None:
    super().__init__(no.LinearTimeline(0.0, 0.01), no.MonteCarlo.nondeterministic_stream)

    self.N = N
    self.range = range
    self.vision = vision
    self.exclusion = exclusion
    self.speed = speed

    # constrained 3d space
    self.domain = no.Space(np.zeros(3), np.full(3, self.range), edge=no.Edge.CONSTRAIN)

    # initially in [0,1]^dim
    self.boids = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(self.N)),
      data={
        "x": self.range / 4 + self.mc.ustream(N) * self.range / 2,
        "y": self.range / 4 + self.mc.ustream(N) * self.range / 2,
        "z": self.range / 4 + self.mc.ustream(N) * self.range / 2,
        "vx": self.mc.ustream(N) - 0.5,
        "vy": self.mc.ustream(N) - 0.5,
        "vz": self.mc.ustream(N) - 0.5,
      }
    )

    self.__normalise_v()

    self.fig, self.g = self.__init_visualisation()

    # suppress divsion by zero warnings
    np.seterr(divide='ignore')

    self.paused = False

  def step(self) -> None:
    if self.paused:
      sleep(0.2)
      self.__update_visualisation()
      return

    d2, (dx, dy, dz) = self.domain.dists2((self.boids.x, self.boids.y, self.boids.z))
    np.fill_diagonal(d2, np.inf) # no self-influence

    # mark boids that need to separate
    too_close = d2 < self.exclusion ** 2
    self.__separate(too_close.astype(float), dx, dy, dz)

    # mark boids that interact
    in_range = np.logical_and(d2 < self.vision ** 2, ~too_close).astype(float)
    self.__cohere(in_range, dx, dy, dz)
    self.__align(in_range)

    self.__normalise_v()

    (self.boids.x, self.boids.y, self.boids.z), (self.boids.vx, self.boids.vy, self.boids.vz) = self.domain.move(
      (self.boids.x, self.boids.y, self.boids.z),
      (self.boids.vx, self.boids.vy, self.boids.vz),
      self.timeline.dt(),
      ungroup=True
    )

    sleep(0.001)
    self.__update_visualisation()

  def __align(self, in_range: np.ndarray) -> None:

    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights == np.inf] = 0.0

    mean_vx = (in_range * self.boids.vx.values) @ weights
    mean_vy = (in_range * self.boids.vy.values) @ weights
    mean_vz = (in_range * self.boids.vz.values) @ weights

    self.boids.vx -= mean_vx * Boids3d.ALIGN_COEFF
    self.boids.vy -= mean_vy * Boids3d.ALIGN_COEFF
    self.boids.vz -= mean_vz * Boids3d.ALIGN_COEFF


  def __cohere(self, in_range: np.ndarray, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> None:
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights == np.inf] = 0.0
    x = (in_range * dx) @ weights
    y = (in_range * dy) @ weights
    z = (in_range * dz) @ weights

    self.boids.vx -= x * Boids3d.COHERE_COEFF
    self.boids.vy -= y * Boids3d.COHERE_COEFF
    self.boids.vz -= z * Boids3d.COHERE_COEFF

  def __separate(self, in_range: np.ndarray, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> None:
    if np.all(in_range == 0):
      return
    x = (in_range * dx).sum(axis=0)
    y = (in_range * dy).sum(axis=0)
    z = (in_range * dz).sum(axis=0)
    self.boids.vx += x * Boids3d.SEPARATE_COEFF
    self.boids.vy += y * Boids3d.SEPARATE_COEFF
    self.boids.vz += z * Boids3d.SEPARATE_COEFF

  def __normalise_v(self) -> None:
    norm = np.clip(np.sqrt(self.boids.vx ** 2 + self.boids.vy ** 2 + self.boids.vz ** 2), a_min = 0.00001, a_max=None)
    self.boids.vx *= self.speed / norm
    self.boids.vy *= self.speed / norm
    self.boids.vz *= self.speed / norm

  def __init_visualisation(self) -> tuple[Any, Any]:
    plt.ion()

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("[q to quit]", y=0.05, x=0.05)
    ax = plt.axes(projection="3d")

    g = ax.scatter(self.boids.x, self.boids.y, self.boids.z, s=5, c="k")
    ax.set_xlim(0.0, self.range)
    ax.set_ylim(0.0, self.range)
    ax.set_zlim(0.0, self.range)
    plt.axis("off")
    fig.canvas.flush_events()

    def on_keypress(event):
      if event.key == "p":
        self.paused = not self.paused
      elif event.key == "q":
        self.halt()
      else:
       no.log("%s doesnt do anything. p to pause/resume, q to quit" % event.key)

    fig.canvas.mpl_connect('key_press_event', on_keypress)
    return fig, g

  def __update_visualisation(self) -> None:
    self.g._offsets3d = (self.boids.x, self.boids.y, self.boids.z)
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()
