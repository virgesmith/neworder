import numpy as np
import numpy.typing as npt
import pandas as pd
from markov_chain import MarkovChain  # ty:ignore[unresolved-import]
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def _plot_occupancy(
    ax: Axes,
    model: MarkovChain,
    summary: pd.DataFrame,
    equilibrium: npt.NDArray[np.float64],
    title: str,
) -> None:
    """
    Plots the simulated state occupancy (as a proportion of the population) over time, as a
    stacked bar chart, with the analytically-computed equilibrium proportions overlaid for
    comparison (dashed lines).
    """
    dt = model.timeline.dt
    proportions = summary[model.states] / model.npeople

    bottom = np.zeros(len(summary))
    for state in model.states:
        ax.bar(summary.t, proportions[state], width=dt, bottom=bottom, label=f"State {state}")
        bottom += proportions[state]

    for level in np.cumsum(equilibrium)[:-1]:
        ax.axhline(level, color="black", linestyle="--", linewidth=0.75)

    ax.legend()
    ax.set_title(title)
    ax.set_ylabel("Proportion of population")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 1)


def show(model: MarkovChain) -> None:
    """
    Plots state occupancy over time (with analytic equilibrium overlaid, dashed) for the pooled,
    single-matrix simulation. If the model also ran a grouped simulation (group_transition_matrices
    was supplied), a second panel shows the same thing for that run, side by side, so the different
    equilibria are visible at a glance.
    """
    has_groups = model.group_transition_matrices is not None
    fig, axes = plt.subplots(1, 2 if has_groups else 1, figsize=(12, 5) if has_groups else (6, 5), squeeze=False)

    _plot_occupancy(axes[0, 0], model, model.summary, model.stationary_distribution(), "Pooled transition matrix")
    if has_groups:
        _plot_occupancy(
            axes[0, 1],
            model,
            model.summary_conditional,
            model.mixed_stationary_distribution(),
            "Per-group transition matrices (transition_conditional)",
        )

    fig.suptitle("State occupancy over time, with analytic equilibrium (dashed)")
    fig.tight_layout()
    plt.show()
