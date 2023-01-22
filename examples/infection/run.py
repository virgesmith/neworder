import neworder as no

from infection import Infection

# centroid (latlon)
point = (53.925, -1.822)
# area size (m)
dist = 2000
# number of agents
N = 1000
# initial number of infected agents
I = 1
# mean speed of movement
speed = 10
# max distance an infection can occur
infection_radius = 1
# number of steps infected before immunity
recovery_time = 100
# probability of dying from infection at any point during the infection
mortality = 0.01

m = Infection(point, dist, N, I, speed, infection_radius, recovery_time, mortality)
no.run(m)
