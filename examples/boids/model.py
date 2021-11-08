import neworder as no

from boids2d import Boids

N = 500 # number of boids
range = 1.0 # extent of the domain
vision = 0.2 # distance boids "see"
exclusion = 0.02 # distance collision avoidance kicks in

speed = 0.5 # speed of movement

m = Boids(N, range, vision, exclusion, speed)

no.log("q to quit")
no.run(m)
