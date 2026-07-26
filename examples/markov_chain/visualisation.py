import numpy as np
import numpy.typing as npt
from markov_chain import ConditionalMarkovChain, MarkovChain, MarkovChainBase  # ty:ignore[unresolved-import]
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def _plot_occupancy(ax: Axes, model: MarkovChainBase, equilibrium: npt.NDArray[np.float64], title: str) -> None:
    """
    Plots the simulated state occupancy (as a proportion of the population) over time, as a
    stacked bar chart, with the analytically-computed equilibrium proportions overlaid for
    comparison (dashed lines).
    """
    dt = model.timeline.dt
    proportions = model.summary[model.states] / model.npeople

    bottom = np.zeros(len(model.summary))
    for state in model.states:
        ax.bar(model.summary.t, proportions[state], width=dt, bottom=bottom, label=f"State {state}")
        bottom += proportions[state]

    for level in np.cumsum(equilibrium)[:-1]:
        ax.axhline(level, color="black", linestyle="--", linewidth=0.75)

    ax.legend()
    ax.set_title(title)
    ax.set_ylabel("Proportion of population")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 1)


def show(pooled_model: MarkovChain, grouped_model: ConditionalMarkovChain | None = None) -> None:
    """
    Plots state occupancy over time (with analytic equilibrium overlaid, dashed) for pooled_model.
    If grouped_model is also given, a second panel shows the same thing for it, side by side, so
    the different equilibria (dashed lines) are visible at a glance.
    """
    fig, axes = plt.subplots(1, 2 if grouped_model else 1, figsize=(12, 5) if grouped_model else (6, 5), squeeze=False)

    _plot_occupancy(axes[0, 0], pooled_model, pooled_model.stationary_distribution(), "Pooled transition matrix")
    if grouped_model is not None:
        _plot_occupancy(
            axes[0, 1],
            grouped_model,
            grouped_model.mixed_stationary_distribution(),
            "Per-group transition matrices",
        )

    fig.suptitle("State occupancy over time, with analytic equilibrium (dashed)")
    fig.tight_layout()
    plt.show()
