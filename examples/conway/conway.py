from __future__ import annotations

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib import colors
from matplotlib.image import AxesImage  # type: ignore

import neworder as no


class Conway(no.Model):
    __glider = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=int)

    def __init__(self, nx: int, ny: int, edge: no.Edge = no.Edge.WRAP) -> None:
        super().__init__(no.LinearTimeline(0, 1), no.MonteCarlo.nondeterministic_stream)

        # create n automata at regular positions
        init_state = np.zeros((nx * ny))
        init_state[::2] = 1
        init_state[::7] = 1

        self.domain = no.StateGrid(init_state.reshape(ny, nx), edge=edge)

        # self.domain.state[20:23, 20:23] = Conway.__glider

        self.fig, self.g = self.__init_visualisation()

    # !step!
    def step(self) -> None:
        n = self.domain.count_neighbours(lambda x: x > 0)

        deaths = np.logical_or(n < 2, n > 3)
        births = n == 3

        self.domain.state = self.domain.state * ~deaths + births

        self.__update_visualisation()

    # !step!

    def check(self) -> bool:
        # randomly place a glider (not across edge)
        if self.timeline.index == 0:
            x = self.mc.raw() % (self.domain.state.shape[0] - 2)
            y = self.mc.raw() % (self.domain.state.shape[1] - 2)
            self.domain.state[x : x + 3, y : y + 3] = np.rot90(
                Conway.__glider, self.mc.raw() % 4
            )
        return True

    def __init_visualisation(self) -> tuple[plt.Figure, AxesImage]:
        plt.ion()
        cmap = colors.ListedColormap(
            [
                "black",
                "white",
                "purple",
                "blue",
                "green",
                "yellow",
                "orange",
                "red",
                "brown",
            ]
        )
        fig = plt.figure(constrained_layout=True, figsize=(8, 8))
        g = plt.imshow(self.domain.state, cmap=cmap, vmax=9)
        plt.axis("off")

        fig.canvas.flush_events()
        fig.canvas.mpl_connect(
            "key_press_event", lambda event: self.halt() if event.key == "q" else None
        )

        return fig, g

    def __update_visualisation(self) -> None:
        self.g.set_data(self.domain.state)
        # plt.savefig("/tmp/conway%04d.png" % self.timeline.index, dpi=80)
        # if self.timeline.index > 100:
        #   self.halt()

        self.fig.canvas.flush_events()
