from __future__ import annotations
from typing import Any
from time import sleep
import numpy as np
import pandas as pd # type: ignore
import neworder as no
import matplotlib.pyplot as plt # type: ignore


class Boids2d(no.Model):

  ALIGN_COEFF = 2.0
  COHERE_COEFF = 0.02
  SEPARATE_COEFF = 0.2

  def __init__(self, N: int, range: float, vision: float, exclusion: float, speed: float) -> None:
    super().__init__(no.LinearTimeline(0.0, 0.01), no.MonteCarlo.nondeterministic_stream)

    self.N = N
    self.range = range
    self.vision = vision
    self.exclusion = exclusion
    self.speed = speed

    # continuous wraparound 2d space
    self.domain = no.Space(np.zeros(2), np.full(2, self.range), edge=no.Edge.WRAP)

    # initially in [0,1]^dim
    self.boids = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(self.N)),
      data={
        "x": self.mc.ustream(N) * self.range,
        "y": self.mc.ustream(N) * self.range,
        "vx": self.mc.ustream(N) - 0.5,
        "vy": self.mc.ustream(N) - 0.5,
      }
    )

    self.__normalise_v()

    self.fig, self.g = self.__init_visualisation()

    # suppress divsion by zero warnings
    np.seterr(divide='ignore')

  def step(self) -> None:

    d2, (dx, dy) = self.domain.dists2((self.boids.x, self.boids.y))
    np.fill_diagonal(d2, np.inf) # no self-influence

    too_close = d2 < self.exclusion**2
    self.__separate(too_close.astype(float), dx, dy)

    # mask those that needed to separate
    in_range = np.logical_and(d2 < self.vision ** 2, ~too_close).astype(float)
    self.__cohere(in_range, dx, dy)
    self.__align(in_range)

    self.__normalise_v()

    (self.boids.x, self.boids.y), _ = self.domain.move(
      (self.boids.x, self.boids.y),
      (self.boids.vx, self.boids.vy),
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

    self.boids.vx -= mean_vx * Boids2d.ALIGN_COEFF
    self.boids.vy -= mean_vy * Boids2d.ALIGN_COEFF

  def __cohere(self, in_range: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> None:
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights == np.inf] = 0.0
    x = (in_range * dx) @ weights
    y = (in_range * dy) @ weights

    self.boids.vx -= x * Boids2d.COHERE_COEFF
    self.boids.vy -= y * Boids2d.COHERE_COEFF

  def __separate(self, in_range: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> None:
    if np.all(in_range == 0):
      return

    x = (in_range * dx).sum(axis=0)
    y = (in_range * dy).sum(axis=0)
    self.boids.vx += x * Boids2d.SEPARATE_COEFF
    self.boids.vy += y * Boids2d.SEPARATE_COEFF

  def __normalise_v(self) -> None:
    norm = np.clip(np.sqrt(self.boids.vx ** 2 + self.boids.vy ** 2), a_min = 0.00001, a_max=None)
    self.boids.vx *= self.speed / norm
    self.boids.vy *= self.speed / norm

  def __init_visualisation(self) -> tuple[Any, Any]:
    plt.ion()

    fig = plt.figure(constrained_layout=True, figsize=(8, 8))
    g = plt.quiver(self.boids.x, self.boids.y, self.boids.vx / self.speed, self.boids.vy / self.speed, scale=50, width=0.005, headwidth=2)
    plt.xlim(0.0, self.range)
    plt.ylim(0.0, self.range)
    plt.axis("off")

    fig.canvas.mpl_connect('key_press_event', lambda event: self.halt() if event.key == "q" else None)
    fig.canvas.flush_events()

    return fig, g

  def __update_visualisation(self) -> None:
    self.g.set_offsets(np.c_[self.boids.x, self.boids.y])
    self.g.set_UVC(self.boids.vx / self.speed, self.boids.vy / self.speed)
    self.fig.canvas.flush_events()
