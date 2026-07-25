import numpy as np
from markov_chain import MarkovChain  # ty:ignore[unresolved-import]
from matplotlib import pyplot as plt


def show(model: MarkovChain) -> None:
    """
    Plots the simulated state occupancy (as a proportion of the population) over time, as a
    stacked bar chart, with the analytically-computed equilibrium proportions overlaid for
    comparison (dashed lines).
    """
    dt = model.timeline.dt
    proportions = model.summary[model.states] / model.npeople

    bottom = np.zeros(len(model.summary))
    for state in model.states:
        plt.bar(model.summary.t, proportions[state], width=dt, bottom=bottom, label=f"State {state}")
        bottom += proportions[state]

    for level in np.cumsum(model.stationary_distribution())[:-1]:
        plt.axhline(level, color="black", linestyle="--", linewidth=0.75)

    plt.legend()
    plt.title("State occupancy over time, with analytic equilibrium (dashed)")
    plt.ylabel("Proportion of population")
    plt.xlabel("Time")
    plt.ylim(0, 1)

    plt.show()
