"""pyramid plots"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import numpy as np


# see https://stackoverflow.com/questions/27694221/using-python-libraries-to-plot-two-horizontal-bar-charts-sharing-same-y-axis
def plot(
    ages: np.ndarray[np.int64, np.dtype[np.int64]],
    males: np.ndarray[np.float64, np.dtype[np.float64]],
    females: np.ndarray[np.float64, np.dtype[np.float64]],
) -> tuple[plt.Figure, plt.Axes, Any, Any]:
    xmax = 4000  # max(max(males), max(females))

    fig, axes = plt.subplots(ncols=2, sharey=True)
    plt.gca().set_ylim([min(ages), max(ages) + 1])
    fig.suptitle("2011")
    axes[0].set(title="Males")
    axes[0].set(xlim=[0, xmax])
    axes[1].set(title="Females")
    axes[1].set(xlim=[0, xmax])
    axes[0].yaxis.tick_right()
    # axes[1].set(yticks=ages)
    axes[0].invert_xaxis()
    for ax in axes.flat:
        ax.margins(0.03)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.125)
    mbar = axes[0].barh(ages, males, align="center", color="blue")
    fbar = axes[1].barh(ages, females, align="center", color="red")
    plt.pause(0.1)
    plt.ion()
    # plt.savefig("./pyramid2011.png")

    return fig, axes, mbar, fbar


def update(
    title: str,
    fig: plt.Figure,
    axes: plt.Axes,
    mbar: Any,
    fbar: Any,
    ages: np.ndarray[np.int64, np.dtype[np.int64]],
    males: np.ndarray[np.float64, np.dtype[np.float64]],
    females: np.ndarray[np.float64, np.dtype[np.float64]],
) -> tuple[Any, Any]:
    for rect, h in zip(mbar, males, strict=False):
        rect.set_width(h)
    for rect, h in zip(fbar, females, strict=False):
        rect.set_width(h)

    fig.suptitle(title)
    # plt.savefig("./pyramid%s.png" % title)
    plt.pause(0.1)
    return mbar, fbar


# def hist(a):
#   plt.hist(a, bins=range(120))
#   plt.show()
