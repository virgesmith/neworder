from enum import Enum

class State(Enum):
  UNINFECTED = 0
  ASYMPTOMATIC = 1
  MILD = 2
  SEVERE = 3
  CRITICAL = 4
  RECOVERED = 5
  DECEASED = 6

INFECTIOUS = [State.ASYMPTOMATIC, State.MILD, State.SEVERE, State.CRITICAL]

# params of poisson process transitions (1/mean)
lambda_01 = 1/7.0 
lambda_12 = 1/7.0
lambda_15 = 1/7.0
lambda_23 = 1/7.0
lambda_25 = 1/7.0
lambda_34 = 1/7.0
lambda_35 = 1/7.0
lambda_45 = 1/7.0
lambda_46 = 1/7.0

# probabilities
contagiousness = 0.18 # ~f(INFECTED)

p_12 = 0.8      # p(ASYMPTOMATIC->MILD)
p_15 = 1 - p_12 # p(ASYMPTOMATIC->RECOVERED)
p_23 = 0.2      # p(MILD->SEVERE)
p_25 = 1 - p_23 # p(MILD->RECOVERED)
p_34 = 0.25     # p(SEVERE->CRITICAL)
p_35 = 1 - p_34 # p(SEVERE->RECOVERED)
p_46 = 0.20     # p(CRITICAL->DECEASED)
p_45 = 1 - p_46 # p(CRITICAL->RECOVERED)

beds_pct = 0.01 # hospital beds per capita
ccu_beds_pct = 0.0005 # critical care beds per capital
