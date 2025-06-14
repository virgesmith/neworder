from infection import Infection

import neworder as no

# centroid (latlon)
POINT = (53.925, -1.822)
# area size (m)
RADIUS = 2000
# number of agents
NUM_AGENTS = 1000
# initial number of infected agents
NUM_INFECTED = 1
# mean speed of movement
SPEED = 10
# max distance an infection can occur
INFECTION_RADIUS = 1
# number of steps infected before immunity
RECOVERY_TIME = 100
# probability of dying from infection at any point during the infection
MORTALITY = 0.01

m = Infection(
    POINT,
    RADIUS,
    NUM_AGENTS,
    NUM_INFECTED,
    SPEED,
    INFECTION_RADIUS,
    RECOVERY_TIME,
    MORTALITY,
)
no.run(m)
