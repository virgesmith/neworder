"""
Markov chain continuous vs discrete
"""

import numpy as np
import neworder as no
from markov_chain import MarkovChain 

from matplotlib import pyplot as plt

# Logging and checking options
#no.verbose()
no.checked()

npeople = 10000
days = 28
dt = 0.2

# params of poisson process transitions (p=lambda.exp(-lambda.x) where lambda=1/mean)
mu_01 = 11.0 
mu_02 = 13.0
mu_12 = 7.0

lambda_01 = 1.0 / mu_01 
lambda_02 = 1.0 / mu_02
lambda_12 = 1.0 / mu_12

states = np.array([0, 1, 2])
# possible transitions:
# 0 -> 1
#  \    \
#    ->  2
transition_matrix = np.array([
  [1.0 - lambda_01 * dt - lambda_02 * dt, lambda_01 * dt,       lambda_02 * dt],
  [0.0,                                   1.0 - lambda_12 * dt, lambda_12 * dt],
  [0.0,                                   0.0,                  1.0           ]      
])

timeline = no.Timeline(0, days, [int(days/dt)])

model = MarkovChain(timeline, npeople, states, transition_matrix)

no.run(model)

no.log(model.summary.head())
model.summary.plot(kind='bar', width=1.0, stacked=True)
plt.title("State occupancy")
plt.ylabel("Count")
plt.xlabel("Timestep")
# this has to be a bug
#plt.xticks(np.arange(0, 28,))
plt.tick_params(which="minor", length=1.0)

no.log("Final state occupancy:")
no.log(model.summary.tail(1))
plt.show()
