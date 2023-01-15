import neworder as no

from infection import Infection

# number of agents
N = 1000
# initial number of infected agents
I = 10
# speed of movement
speed = 10
# max distance an infection can occur
infection_radius = 10

m = Infection(N, I, speed, infection_radius)
no.run(m)