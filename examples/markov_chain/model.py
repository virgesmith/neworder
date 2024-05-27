import time

import numpy as np
import visualisation
from markov_chain import MarkovChain

import neworder as no

# Logging and checking options
# no.verbose()
no.checked()

npeople = 100000
tmax = 100
dt = 1.0

# params of poisson process transitions (p=lambda.exp(-lambda.x) where lambda=1/mean)
mu_01 = 13.0
mu_02 = 23.0
mu_12 = 29.0
mu_20 = 17.0

lambda_01 = 1.0 / mu_01
lambda_02 = 1.0 / mu_02
lambda_12 = 1.0 / mu_12
lambda_20 = 1.0 / mu_20

states = np.array([0, 1, 2])
# possible transitions:
# 0 -> 1
#  \    \
#    <-> 2
transition_matrix = np.array(
    [
        [1.0 - lambda_01 * dt - lambda_02 * dt, lambda_01 * dt, lambda_02 * dt],
        [0.0, 1.0 - lambda_12 * dt, lambda_12 * dt],
        [lambda_20 * dt, 0.0, 1.0 - lambda_20 * dt],
    ]
)

timeline = no.LinearTimeline(0, tmax, tmax)

model = MarkovChain(timeline, npeople, states, transition_matrix)

start = time.time()
no.run(model)
no.log("run time = %.2fs" % (time.time() - start))

visualisation.show(model)
