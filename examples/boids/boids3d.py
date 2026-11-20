from __future__ import annotations

from time import sleep
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

import neworder as no

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool]

# boid, near a predator, too close to a predator, predator
_colours = np.array(
    [
        (0.2, 0.2, 0.2, 1.0),
        (0.0, 0.7, 0.0, 1.0),
        (1.0, 0.6, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
    ]
)


class Boids3d(no.Model):
    ALIGN_COEFF = 0.1
    COHERE_COEFF = 2
    SEPARATE_COEFF = 0.002
    AVOID_COEFF = 0.1
    REVERT_COEFF = 0.05

    def __init__(self, N: int, range: float, vision: float, exclusion: float, speed: float) -> None:
        super().__init__(no.timeline.numeric(0.0, step=0.01))
        self.mc = no.MonteCarlo(no.MonteCarlo.nondeterministic_stream)

        self.N = N
        self.range = range
        self.vision = vision
        self.exclusion = exclusion
        self.speed = speed

        # unconstrained 3d space
        self.domain = no.Space(np.zeros(3), np.full(3, self.range), edge=no.Edge.UNBOUNDED)

        # the first N_predators boids are predators
        self.N_predators = 1

        # one row per boid, initially in [0, range]^3
        self.x = self.mc.ustream(3 * N).reshape(3, N).T * self.range
        self.v = self.mc.ustream(3 * N).reshape(3, N).T - 0.5
        self.colour = np.zeros(N, dtype=int)

        self._normalise()

        self.fig, self.g = self._init_visualisation()

        self.paused = False

    def step(self) -> None:
        if self.paused:
            sleep(0.2)
            self._update_visualisation()
            return

        # the flock condenses into a dense ball, so a dense distance matrix is faster than a neighbour search
        d2 = np.maximum(cdist(self.x, self.x, "sqeuclidean"), 1e-12)
        np.fill_diagonal(d2, np.inf)  # no self-influence
        too_close = d2 < self.exclusion**2
        in_range = d2 < self.vision**2

        self._separate(too_close, d2)
        self._avoid(in_range, d2)
        # boids that need to separate don't align or cohere
        neighbours = (in_range & ~too_close).astype(float)
        self._cohere(neighbours)
        self._align(neighbours)
        # favour returning to the centre
        self._revert()
        self._normalise()

        P = self.N_predators
        self.colour[:] = 0
        self.colour[(neighbours[:P] > 0).any(axis=0)] = 1
        self.colour[too_close[:P].any(axis=0)] = 2
        self.colour[:P] = 3

        self.x, self.v = self.domain.move(self.x, self.v, self.timestep.dt)

        self._update_visualisation()

    def _separate(self, too_close: BoolArray, d2: FloatArray) -> None:
        # impact on v is proportional to 1/d^2, away from each boid that's too close
        f = np.where(too_close, Boids3d.SEPARATE_COEFF / d2, 0.0)
        self.v -= _weighted_offsets(f, self.x, self.x)

    def _avoid(self, in_range: BoolArray, d2: FloatArray) -> None:
        P = self.N_predators
        f = np.where(in_range[:P], Boids3d.AVOID_COEFF / d2[:P], 0.0)
        # each boid moves away from the predators in range
        self.v -= _weighted_offsets(f.T, self.x, self.x[:P])

    def _cohere(self, neighbours: FloatArray) -> None:
        # steer towards the centre of the neighbours
        centre = _mean_of(neighbours, self.x)
        self.v += np.where(neighbours.any(axis=1)[:, None], centre - self.x, 0.0) * Boids3d.COHERE_COEFF

    def _align(self, neighbours: FloatArray) -> None:
        self.v += _mean_of(neighbours, self.v) * Boids3d.ALIGN_COEFF

    def _revert(self) -> None:
        self.v -= (self.x - self.range / 2) * Boids3d.REVERT_COEFF

    def _normalise(self) -> None:
        norm = np.clip(np.linalg.norm(self.v, axis=1), a_min=0.00001, a_max=None)
        self.v *= (self.speed / norm)[:, None]
        # predators are faster
        self.v[: self.N_predators] *= 1.5

    def _init_visualisation(self) -> tuple[Any, Any]:
        plt.ion()

        fig = plt.figure(figsize=(10, 10))
        fig.suptitle("[p to pause, q to quit]", y=0.05, x=0.1)
        ax = plt.axes()

        g = ax.scatter(*self._projection(), c=_colours[self.colour], s=_size(self.x[:, 2]))

        ax.set_xlim(0.0, self.range)
        ax.set_ylim(0.0, self.range)
        plt.axis("off")
        plt.tight_layout()
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
        self.g.set_offsets(np.column_stack(self._projection()))
        self.g.set_sizes(_size(self.x[:, 2]))
        self.g.set_facecolor(_colours[self.colour])
        # draw_idle lets an interactive backend skip frames it can't keep up with, rather than blocking on each one
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _projection(self) -> tuple[FloatArray, FloatArray]:
        """Perspective projection of the boids onto the x-y plane"""
        c = self.range / 2
        scale = 1.0 / (1.0 + self.x[:, 2])
        return (self.x[:, 0] - c) * scale + c, (self.x[:, 1] - c) * scale + c


def _size(z: FloatArray) -> FloatArray:
    return 5.0 / (0.5 + z)


def _weighted_offsets(w: FloatArray, x: FloatArray, y: FloatArray) -> FloatArray:
    """sum_j w_ij (y_j - x_i), for each i, without forming the offsets"""
    return w @ y - x * w.sum(axis=1)[:, None]


def _mean_of(neighbours: FloatArray, values: FloatArray) -> FloatArray:
    """The mean of the neighbours' values, zero for a boid with no neighbours"""
    n = neighbours.sum(axis=1)[:, None]
    return np.divide(neighbours @ values, n, out=np.zeros_like(values), where=n > 0)
