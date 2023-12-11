from __future__ import annotations
from typing import Any
from time import sleep
import numpy as np
import pandas as pd
import neworder as no
import matplotlib.pyplot as plt # type: ignore

_cmap = {
  0: (0.2, 0.2, 0.2, 1.0),
  1: (0.0, 0.7, 0.0, 1.0),
  2: (1.0, 0.6, 0.0, 1.0),
  3: (1.0, 0.0, 0.0, 1.0)
}

class Boids3d(no.Model):

  ALIGN_COEFF = 0.1
  COHERE_COEFF = 2
  SEPARATE_COEFF = .002
  AVOID_COEFF = 0.1
  REVERT_COEFF = 0.05

  def __init__(self, N: int, range: float, vision: float, exclusion: float, speed: float) -> None:
    super().__init__(no.LinearTimeline(0.0, 0.01), no.MonteCarlo.nondeterministic_stream)

    self.N = N
    self.range = range
    self.vision = vision
    self.exclusion = exclusion
    self.speed = speed

    # unconstrained 3d space
    self.domain = no.Space(np.zeros(3), np.full(3, self.range), edge=no.Edge.UNBOUNDED)

    self.N_predators = 1

    # initially in [0,1]^dim
    self.boids = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(self.N)),
      data={
        "x": self.mc.ustream(N) * self.range,
        "y": self.mc.ustream(N) * self.range,
        "z": self.mc.ustream(N) * self.range,
        "vx": self.mc.ustream(N) - 0.5,
        "vy": self.mc.ustream(N) - 0.5,
        "vz": self.mc.ustream(N) - 0.5,
        "c": 0
      }
    )

    self.__normalise()

    self.fig, self.g = self.__init_visualisation()

    self.paused = False

    # suppress divsion by zero warnings
    np.seterr(divide='ignore')

  def step(self) -> None:
    if self.paused:
      sleep(0.2)
      self.__update_visualisation()
      return

    d2, (dx, dy, dz) = self.domain.dists2((self.boids.x, self.boids.y, self.boids.z))
    np.fill_diagonal(d2, np.inf) # no self-influence

    # separate
    too_close = d2 < self.exclusion**2
    self.__separate(too_close, d2, dx, dy, dz)

    # avoid predator
    in_range = d2 < self.vision ** 2
    self.__avoid(in_range, d2, dx, dy, dz)

    # mask those that needed to separate
    in_range = np.logical_and(in_range, ~too_close).astype(float)

    self.__cohere(in_range, dx, dy, dz)
    self.__align(in_range)

    # favour returning to the origin
    self.__revert()

    self.__normalise()

    # set colours
    self.boids.c = 0
    self.boids.loc[in_range[0:self.N_predators].sum(axis=0) != 0, "c"] = 1
    self.boids.loc[too_close[0:self.N_predators].sum(axis=0) != 0, "c"] = 2
    self.boids.loc[0:self.N_predators - 1, "c"] = 3

    (self.boids.x, self.boids.y, self.boids.z), (self.boids.vx, self.boids.vy, self.boids.vz) = self.domain.move(
      (self.boids.x, self.boids.y, self.boids.z),
      (self.boids.vx, self.boids.vy, self.boids.vz),
      self.timeline.dt,
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

    self.boids.vx += mean_vx * Boids3d.ALIGN_COEFF
    self.boids.vy += mean_vy * Boids3d.ALIGN_COEFF
    self.boids.vz += mean_vz * Boids3d.ALIGN_COEFF

  def __cohere(self, in_range: np.ndarray, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> None:
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights == np.inf] = 0.0
    x = (in_range * dx) @ weights
    y = (in_range * dy) @ weights
    z = (in_range * dz) @ weights

    self.boids.vx += x * Boids3d.COHERE_COEFF
    self.boids.vy += y * Boids3d.COHERE_COEFF
    self.boids.vz += z * Boids3d.COHERE_COEFF

  def __separate(self, in_range: np.ndarray, d2: np.ndarray, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> None:
    # TODO clip d2?
    # impact on v is proportional to 1/f
    f = Boids3d.SEPARATE_COEFF / d2 * in_range
    self.boids.vx += (f * dx).sum(axis=0)
    self.boids.vy += (f * dy).sum(axis=0)
    self.boids.vz += (f * dz).sum(axis=0)

  def __avoid(self, in_range: np.ndarray, d2: np.ndarray, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> None:
    f = Boids3d.AVOID_COEFF / d2[0:self.N_predators, :] * in_range[0:self.N_predators, :]
    self.boids.vx += (f * dx[0:self.N_predators, :]).sum(axis=0)
    self.boids.vy += (f * dy[0:self.N_predators, :]).sum(axis=0)
    self.boids.vz += (f * dz[0:self.N_predators, :]).sum(axis=0)

  def __revert(self) -> None:
    """Return to the origin"""
    self.boids.vx -= (self.boids.x - self.range / 2) * Boids3d.REVERT_COEFF
    self.boids.vy -= (self.boids.y - self.range / 2) * Boids3d.REVERT_COEFF
    self.boids.vz -= (self.boids.z - self.range / 2) * Boids3d.REVERT_COEFF

  def __normalise(self) -> None:
    # normalise speed
    norm = np.clip(np.sqrt(self.boids.vx ** 2 + self.boids.vy ** 2 + self.boids.vz ** 2), a_min = 0.00001, a_max=None)
    self.boids.vx *= self.speed / norm
    self.boids.vy *= self.speed / norm
    self.boids.vz *= self.speed / norm

    # predators are faster
    self.boids.loc[0:self.N_predators - 1, "vx"] *= 1.5
    self.boids.loc[0:self.N_predators - 1, "vy"] *= 1.5
    self.boids.loc[0:self.N_predators - 1, "vz"] *= 1.5

  def __init_visualisation(self) -> tuple[Any, Any]:
    plt.ion()

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("[p to pause, q to quit]", y=0.05, x=0.1)
    ax = plt.axes() # projection="3d")

    g = ax.scatter(_project(self.boids.x, self.boids.z, self.range / 2),
                   _project(self.boids.y, self.boids.z, self.range / 2),
                   c=self.boids.c.map(_cmap), s=_size(self.boids.z))

    ax.set_xlim(0.0, self.range)
    ax.set_ylim(0.0, self.range)
    plt.axis("off")
    plt.tight_layout()
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
    self.g.set_offsets(np.c_[_project(self.boids.x, self.boids.z, self.range / 2),
                             _project(self.boids.y, self.boids.z, self.range / 2)])
    self.g.set_sizes(_size(self.boids.z))
    self.g.set_facecolor(self.boids.c.map(_cmap))
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

def _project(a: np.ndarray, z: np.ndarray, c: float) -> np.ndarray:
  d = 1.0
  # centre, adjust, recentre
  return (a - c) * d / (1 + z) + c

def _size(z: np.ndarray) -> np.ndarray:
  return  5.0 / (0.5 + z) # np.clip(.5 + z, a_min=0.1, a_max=None)