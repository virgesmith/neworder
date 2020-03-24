from enum import Enum
import numpy as np

class State(Enum):
  INIT = 0
  STATE1 = 1
  STATE2 = 2

NUMSTATES = 3
ALLSTATES = np.array([State.INIT, State.STATE1, State.STATE2])

# params of poisson process transitions (p=lambda.exp(-lambda.x) where lambda=1/mean)
mu_01 = 5.0 # units of dt
mu_02 = 10.0

lambda_01 = 1.0 / mu_01 
lambda_02 = 1.0 / mu_02


