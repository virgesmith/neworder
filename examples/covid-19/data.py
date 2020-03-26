from enum import Enum
import numpy as np

class State(Enum):
  UNINFECTED = 0
  ASYMPTOMATIC = 1
  MILD = 2
  SEVERE = 3
  CRITICAL = 4
  RECOVERED = 5
  DECEASED = 6

NUMSTATES = 7
ALLSTATES = np.array([State.UNINFECTED, State.ASYMPTOMATIC, State.MILD, State.SEVERE, State.CRITICAL, State.RECOVERED, State.DECEASED])
INFECTIOUS = [State.ASYMPTOMATIC, State.MILD, State.SEVERE, State.CRITICAL]


initial_infection_rate = 0.001
# convert to a probability over dt: R0^(-g/dt)-1
R0 = 2.5
g = 5 # days, generation length for R0 


mu_12 = 5  # t(ASYMPTOMATIC->MILD)
mu_15 = 10 # t(ASYMPTOMATIC->RECOVERED)
mu_23 = 16 # t(MILD->SEVERE)
mu_25 = 4  # t(MILD->RECOVERED)
mu_34 = 5  # t(SEVERE->CRITICAL)
mu_35 = 5  # t(SEVERE->RECOVERED)
mu_46 = 3  # t(CRITICAL->DECEASED)
mu_45 = 3  # t(CRITICAL->RECOVERED)

# params of poisson process transitions (1/mean)
lambda_12 = 1.0 / mu_12
lambda_15 = 1.0 / mu_15
lambda_23 = 1.0 / mu_23
lambda_25 = 1.0 / mu_25
lambda_34 = 1.0 / mu_34
lambda_35 = 1.0 / mu_35
lambda_45 = 1.0 / mu_45
lambda_46 = 1.0 / mu_46

# nonlinearities
beds_pct = 0.01 # hospital beds per capita
ccu_beds_pct = 0.0005 # critical care beds per capital
