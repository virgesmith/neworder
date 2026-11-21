from __future__ import annotations

from time import sleep
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import ListedColormap
from scipy.spatial import KDTree

import neworder as no

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.intp]


class Boids2d(no.Model):
    ALIGN_COEFF = 0.1
    COHERE_COEFF = 2
    SEPARATE_COEFF = 0.001
    AVOID_COEFF = 0.05

    def __init__(self, N: int, range: float, vision: float, exclusion: float, speed: float) -> None:
        super().__init__(no.timeline.numeric(0.0, step=0.01))
        self.mc = no.MonteCarlo(no.MonteCarlo.deterministic_independent_stream)

        self.N = N
        self.range = range
        self.vision = vision
        self.exclusion = exclusion
        self.speed = speed

        # continuous wraparound 2d space
        self.domain = no.Space(np.zeros(2), np.full(2, self.range), edge=no.Edge.WRAP)

        # the first N_predators boids are predators
        self.N_predators = 2
        self.paused = False

        # one row per boid, initially in [0, range]^2
        self.x = self.mc.ustream(2 * N).reshape(2, N).T * self.range
        self.v = self.mc.ustream(2 * N).reshape(2, N).T - 0.5
        # boid, near a predator, too close to a predator, predator
        self.colour = np.zeros(N, dtype=int)

        self._normalise()

        self.fig, self.g = self._init_visualisation()

    def step(self) -> None:
        if self.paused:
            sleep(0.2)
            self._update_visualisation()
            return

        i, j, delta, d2 = self._neighbours()
        too_close = d2 < self.exclusion**2
        predator = i < self.N_predators

        self._separate(i[too_close], delta[too_close], d2[too_close])
        self._avoid(j[predator], delta[predator], d2[predator])
        # boids that need to separate don't align or cohere
        self._cohere(i[~too_close], delta[~too_close])
        self._align(i[~too_close], j[~too_close])
        self._normalise()

        self.colour[:] = 0
        self.colour[j[predator & ~too_close]] = 1
        self.colour[j[predator & too_close]] = 2
        self.colour[: self.N_predators] = 3

        self.x, self.v = self.domain.move(self.x, self.v, self.timestep.dt)

        self._update_visualisation()

    def _neighbours(self) -> tuple[IndexArray, IndexArray, FloatArray, FloatArray]:
        """Every ordered pair (i, j) of boids within vision of each other, the offset of j from i, and its length squared"""
        # a periodic tree requires points in [0, range), but wrapping a position can round to exactly range
        x = np.where(self.x < self.range, self.x, 0.0)
        pairs = KDTree(x, boxsize=self.range).query_pairs(self.vision, output_type="ndarray")
        i = np.concatenate((pairs[:, 0], pairs[:, 1]))
        j = np.concatenate((pairs[:, 1], pairs[:, 0]))
        delta = x[j] - x[i]
        # the shortest offset may be across an edge of the domain
        delta -= self.range * np.round(delta / self.range)
        # coincident boids would otherwise repel each other infinitely
        d2 = np.maximum(np.sum(delta**2, axis=1), 1e-12)
        return i, j, delta, d2

    def _separate(self, i: IndexArray, delta: FloatArray, d2: FloatArray) -> None:
        # impact on v is proportional to 1/d^2
        self.v -= _sum_by(i, delta / d2[:, None], self.N) * Boids2d.SEPARATE_COEFF

    def _avoid(self, prey: IndexArray, delta: FloatArray, d2: FloatArray) -> None:
        self.v += _sum_by(prey, delta / d2[:, None], self.N) * Boids2d.AVOID_COEFF

    def _cohere(self, i: IndexArray, delta: FloatArray) -> None:
        # steer towards the centre of the neighbours
        self.v += _mean_by(i, delta, self.N) * Boids2d.COHERE_COEFF

    def _align(self, i: IndexArray, j: IndexArray) -> None:
        self.v += _mean_by(i, self.v[j], self.N) * Boids2d.ALIGN_COEFF

    def _normalise(self) -> None:
        norm = np.clip(np.linalg.norm(self.v, axis=1), a_min=0.00001, a_max=None)
        self.v *= (self.speed / norm)[:, None]
        # predators are faster
        self.v[: self.N_predators] *= 1.3

    def _init_visualisation(self) -> tuple[Any, Any]:
        plt.ion()

        fig = plt.figure(constrained_layout=True, figsize=(8, 8))
        g = plt.quiver(
            self.x[:, 0],
            self.x[:, 1],
            self.v[:, 0] / self.speed,
            self.v[:, 1] / self.speed,
            scale=75,
            width=0.005,
            headwidth=2,
            cmap=ListedColormap(["k", "green", "orange", "r"]),
        )

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
                no.log(f"{event.key} doesnt do anything. p to pause/resume, q to quit")

        fig.canvas.mpl_connect("key_press_event", on_keypress)

        return fig, g

    def _update_visualisation(self) -> None:
        self.g.set_offsets(self.x)
        self.g.set_UVC(self.v[:, 0] / self.speed, self.v[:, 1] / self.speed, self.colour / 3)
        self.fig.canvas.flush_events()


def _sum_by(index: IndexArray, values: FloatArray, n: int) -> FloatArray:
    """Sums the rows of values that share an index"""
    return np.column_stack([np.bincount(index, values[:, k], minlength=n) for k in range(values.shape[1])])


def _mean_by(index: IndexArray, values: FloatArray, n: int) -> FloatArray:
    """Averages the rows of values that share an index, zero where there are none"""
    count = np.bincount(index, minlength=n)[:, None]
    return np.divide(_sum_by(index, values, n), count, out=np.zeros((n, values.shape[1])), where=count > 0)
