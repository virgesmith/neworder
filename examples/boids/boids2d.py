from __future__ import annotations
from typing import Any
from time import sleep
import numpy as np
import pandas as pd # type: ignore
import neworder as no
import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import ListedColormap

class Boids2d(no.Model):

  ALIGN_COEFF = 0.1
  COHERE_COEFF = 2
  SEPARATE_COEFF = .001
  AVOID_COEFF = 0.05

  def __init__(self, N: int, range: float, vision: float, exclusion: float, speed: float) -> None:
    super().__init__(no.LinearTimeline(0.0, 0.01), no.MonteCarlo.nondeterministic_stream)

    self.N = N
    self.range = range
    self.vision = vision
    self.exclusion = exclusion
    self.speed = speed

    # continuous wraparound 2d space
    self.domain = no.Space(np.zeros(2), np.full(2, self.range), edge=no.Edge.WRAP)

    self.N_predators = 2
    self.paused = False

    # initially in [0,1]^dim
    self.boids = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(self.N)),
      data={
        "x": self.mc.ustream(N) * self.range,
        "y": self.mc.ustream(N) * self.range,
        "vx": self.mc.ustream(N) - 0.5,
        "vy": self.mc.ustream(N) - 0.5,
        "c": 0.0
      }
    )

    self.__normalise()

    self.fig, self.g = self.__init_visualisation()

    # suppress division by zero warnings
    np.seterr(divide='ignore')

  def step(self) -> None:
    if self.paused:
      sleep(0.2)
      self.__update_visualisation()
      return

    d2, (dx, dy) = self.domain.dists2((self.boids.x, self.boids.y))
    np.fill_diagonal(d2, np.inf) # no self-influence

    # separate
    too_close = d2 < self.exclusion**2
    self.__separate(too_close, d2, dx, dy)

    # avoid predator
    in_range = d2 < self.vision ** 2
    self.__avoid(in_range, d2, dx, dy)

    # mask those that needed to separate
    in_range = np.logical_and(d2 < self.vision ** 2, ~too_close).astype(float)

    self.__cohere(in_range, dx, dy)
    self.__align(in_range)

    self.__normalise()

    # set colours
    self.boids.c = 0
    self.boids.loc[in_range[0:self.N_predators].sum(axis=0) != 0, "c"] = 1/3
    self.boids.loc[too_close[0:self.N_predators].sum(axis=0) != 0, "c"] = 2/3
    self.boids.loc[0:self.N_predators - 1, "c"] = 1

    (self.boids.x, self.boids.y), (self.boids.vx, self.boids.vy) = self.domain.move(
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

    self.boids.vx += mean_vx * Boids2d.ALIGN_COEFF
    self.boids.vy += mean_vy * Boids2d.ALIGN_COEFF

  def __cohere(self, in_range: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> None:
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights == np.inf] = 0.0
    x = (in_range * dx) @ weights
    y = (in_range * dy) @ weights

    self.boids.vx += x * Boids2d.COHERE_COEFF
    self.boids.vy += y * Boids2d.COHERE_COEFF

  def __separate(self, in_range: np.ndarray, d2: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> None:
    # TODO clip d2?
    # impact on v is proportional to 1/f
    f = Boids2d.SEPARATE_COEFF / d2 * in_range
    self.boids.vx += (f * dx).sum(axis=0)
    self.boids.vy += (f * dy).sum(axis=0)

  def __avoid(self, in_range: np.ndarray, d2: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> None:
    f = Boids2d.AVOID_COEFF / d2[0:self.N_predators, :] * in_range[0:self.N_predators, :]
    self.boids.vx += (f * dx[0:self.N_predators, :]).sum(axis=0)
    self.boids.vy += (f * dy[0:self.N_predators, :]).sum(axis=0)

  def __normalise(self) -> None:

    norm = np.clip(np.sqrt(self.boids.vx ** 2 + self.boids.vy ** 2), a_min = 0.00001, a_max=None)
    self.boids.vx *= self.speed / norm
    self.boids.vy *= self.speed / norm

    # predators are faster
    self.boids.loc[0:self.N_predators - 1, "vx"] *= 1.3
    self.boids.loc[0:self.N_predators - 1, "vy"] *= 1.3

  def __init_visualisation(self) -> tuple[Any, Any]:
    plt.ion()

    fig = plt.figure(constrained_layout=True, figsize=(8, 8))
    g = plt.quiver(self.boids.x, self.boids.y,
                   self.boids.vx / self.speed, self.boids.vy / self.speed,
                   scale=75, width=0.005, headwidth=2,
                   cmap=ListedColormap(["k", "green", "orange", "r"]))

    plt.xlim(0.0, self.range)
    plt.ylim(0.0, self.range)
    plt.axis("off")

    fig.canvas.flush_events()
    def on_keypress(event: Any) -> None:
      if event.key == "p":
        self.paused = not self.paused
      elif event.key == "q":
        self.halt()
      else:
       no.log("%s doesnt do anything. p to pause/resume, q to quit" % event.key)

    fig.canvas.mpl_connect('key_press_event', on_keypress)

    return fig, g

  def __update_visualisation(self) -> None:
    self.g.set_offsets(np.c_[self.boids.x, self.boids.y])
    self.g.set_UVC(self.boids.vx / self.speed, self.boids.vy / self.speed, self.boids.c.values)
    self.fig.canvas.flush_events()
