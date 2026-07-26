import time

import numpy as np
import numpy.typing as npt
import visualisation  # ty:ignore[unresolved-import]
from markov_chain import ConditionalMarkovChain, MarkovChain  # ty:ignore[unresolved-import]

import neworder as no

# Logging and checking options
# no.verbose()
no.checked()

npeople = 100000
tmax = 100
dt = 1.0

# Set to True to use the much slower pure-python transition implementation instead of neworder's
# C++ implementation, to compare performance - see MarkovChain.transition_py()
use_python_impl = False

# params of poisson process transitions (p=lambda.exp(-lambda.x) where lambda=1/mean)
mu_01 = 13.0
mu_02 = 23.0
mu_12 = 29.0
mu_20 = 17.0

states = np.array([0, 1, 2])


# possible transitions:
# 0 -> 1
#  \    \
#    <-> 2
def transition_matrix_for(mu_01: float, mu_02: float, mu_12: float, mu_20: float) -> npt.NDArray[np.float64]:
    lambda_01 = 1.0 / mu_01
    lambda_02 = 1.0 / mu_02
    lambda_12 = 1.0 / mu_12
    lambda_20 = 1.0 / mu_20
    return np.array(
        [
            [1.0 - lambda_01 * dt - lambda_02 * dt, lambda_01 * dt, lambda_02 * dt],
            [0.0, 1.0 - lambda_12 * dt, lambda_12 * dt],
            [lambda_20 * dt, 0.0, 1.0 - lambda_20 * dt],
        ]
    )


transition_matrix = transition_matrix_for(mu_01, mu_02, mu_12, mu_20)

# Two subpopulations sharing the same transition topology but with a different relative bias between
# the two routes out of state 0 (uniformly scaling every rate by the same factor leaves the
# equilibrium unchanged - it's the ratio between rates, not their magnitude, that matters), to
# demonstrate no.df.transition_conditional alongside the single-matrix no.df.transition above - see
# ConditionalMarkovChain.mixed_stationary_distribution() for why pooling the two groups into one
# matrix gives a different equilibrium than modelling them separately.
group_transition_matrices = {
    "fast_to_1": transition_matrix_for(mu_01 / 2, mu_02 * 2, mu_12, mu_20),
    "fast_to_2": transition_matrix_for(mu_01 * 2, mu_02 / 2, mu_12, mu_20),
}

pooled_model = MarkovChain(no.LinearTimeline(0, tmax, tmax), npeople, states, transition_matrix, use_python_impl)
grouped_model = ConditionalMarkovChain(no.LinearTimeline(0, tmax, tmax), npeople, states, group_transition_matrices)

start = time.time()
no.run(pooled_model)
no.run(grouped_model)
no.log("run time = %.2fs" % (time.time() - start))

simulated = pooled_model.summary.iloc[-1][states].to_numpy() / npeople
equilibrium = pooled_model.stationary_distribution()
no.log(f"pooled simulated equilibrium proportions:  {np.round(simulated, 4)}")
no.log(f"pooled analytic equilibrium proportions:   {np.round(equilibrium, 4)}")

grouped_simulated = grouped_model.summary.iloc[-1][states].to_numpy() / npeople
mixed_equilibrium = grouped_model.mixed_stationary_distribution()
no.log(f"grouped simulated equilibrium proportions: {np.round(grouped_simulated, 4)}")
no.log(f"grouped analytic (mixed) equilibrium:      {np.round(mixed_equilibrium, 4)}")

visualisation.show(pooled_model, grouped_model)
