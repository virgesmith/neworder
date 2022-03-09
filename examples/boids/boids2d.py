from typing import Any
import numpy as np
import pandas as pd # type: ignore
import neworder as no
import matplotlib.pyplot as plt # type: ignore

TWO_PI = 2 * np.pi


class Boids(no.Model):

  def __init__(self, N: int, range: float, vision: float, exclusion: float, speed: float) -> None:
    super().__init__(no.LinearTimeline(0.0, 0.01), no.MonteCarlo.nondeterministic_stream)

    self.N = N
    self.range = range
    self.vision = vision
    self.exclusion = exclusion
    self.speed = speed

    # continuous wraparound 2d space
    self.domain = no.Space(np.zeros(2), np.full(2, self.range), edge=no.Edge.WRAP)

    # maximum turns for each interaction
    self.sep_max_turn = TWO_PI / 240 # 1.5 degrees
    self.align_max_turn = TWO_PI / 72 # 5 degrees
    self.cohere_max_turn = TWO_PI / 120 # 3 degrees

    # initially in [0,1]^dim
    self.boids = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(self.N)),
      data={
        "x": self.mc.ustream(N) * self.range,
        "y": self.mc.ustream(N) * self.range,
        "theta": self.mc.ustream(N) * TWO_PI # zero is +ve x axis
      }
    )

    self.fig, self.g = self.__init_visualisation()

    # suppress divsion by zero warnings
    np.seterr(divide='ignore')

  def step(self) -> None:
    d2, (dx, dy) = self.domain.dists2((self.boids.x, self.boids.y))
    np.fill_diagonal(d2, np.inf) # no self-influence

    mindists = np.tile(np.amin(d2, axis=0), self.N).reshape((self.N, self.N))
    nearest = np.where(mindists == d2, True, False)

    too_close = np.where(d2 < self.exclusion**2, 1.0, 0.0)
    nearest_in_range = np.logical_and(nearest, too_close).astype(float)

    self.__separate(nearest_in_range, dx, dy, self.sep_max_turn)

    # this masks the align/cohere steps for boids that need to separate
    mask = np.repeat(1.0 - np.sum(nearest_in_range, axis=0), self.N).reshape(self.N, self.N)

    in_range = np.logical_and(np.where(d2 < self.vision**2, 1.0, 0.0), mask).astype(float)
    self.__cohere(in_range, dx, dy, self.cohere_max_turn)
    self.__align(in_range, self.align_max_turn)

    (self.boids.x, self.boids.y), (vx, vy) = self.domain.move((self.boids.x, self.boids.y),
                                                              (np.cos(self.boids.theta) * self.speed, np.sin(self.boids.theta) * self.speed),
                                                              self.timeline.dt(), ungroup=True)

    self.__update_visualisation()
    # if self.timeline.index() > 300:
    #   self.halt()

  def __align(self, in_range: np.ndarray, max_turn: float) -> None:

    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights == np.inf] = 0.0

    # need to convert to x,y to average angles correctly
    # matrix x vector <piecewise-x> vector = vector
    mean_vx = in_range @ np.cos(self.boids.theta) * weights
    mean_vy = in_range @ np.sin(self.boids.theta) * weights
    # print(mean_vx, mean_vy)
    delta = np.clip(np.arctan2(mean_vy, mean_vx) - self.boids.theta, -max_turn, max_turn)
    self.boids.theta = (self.boids.theta + delta) % TWO_PI

  def __cohere(self, in_range: np.ndarray, dx: np.ndarray, dy: np.ndarray, max_turn: float) -> None:
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights == np.inf] = 0.0
    # compute vectors to cen  tre of gravity of neighbours (if any), relative to boid location
    # matrix * matrix * vector = vector
    xbar = in_range @ dx @ weights
    ybar = in_range @ dy @ weights
    delta = np.clip(np.arctan2(ybar, xbar) - self.boids.theta, -max_turn, max_turn)
    self.boids.theta = (self.boids.theta + delta) % TWO_PI

  def __separate(self, in_range: np.ndarray, dx: np.ndarray, dy: np.ndarray, max_turn: float) -> None:
    if np.all(in_range == 0):
      return
    weights = 1.0 / np.sum(in_range, axis=0)
    weights[weights == np.inf] = 0.0

    x = in_range @ dx @ weights
    y = in_range @ dy @ weights
    delta = np.clip(np.arctan2(y, x) - self.boids.theta, -max_turn, max_turn)
    self.boids.theta = (self.boids.theta - delta) % TWO_PI

  def __init_visualisation(self) -> tuple[Any, Any]:
    plt.ion()

    fig = plt.figure(constrained_layout=True, figsize=(8, 8))
    g = plt.quiver(self.boids.x, self.boids.y, np.cos(self.boids.theta), np.sin(self.boids.theta), scale=50, width=0.005, headwidth=2)
    plt.xlim(0.0, self.range)
    plt.ylim(0.0, self.range)
    plt.axis("off")

    fig.canvas.mpl_connect('key_press_event', lambda event: self.halt() if event.key == "q" else None)

    fig.canvas.flush_events()

    return fig, g

  def __update_visualisation(self) -> None:

    self.g.set_offsets(np.c_[self.boids.x, self.boids.y])
    self.g.set_UVC(np.cos(self.boids.theta), np.sin(self.boids.theta))
    # plt.savefig("/tmp/boids2d%04d.png" % self.timeline.index(), dpi=80)
    self.fig.canvas.flush_events()
