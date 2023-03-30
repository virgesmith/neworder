import neworder as no

from boids2d import Boids

N = 500 # number of boids
range = 1.0 # extent of the domain
vision = 0.3 # distance boids "see"
exclusion = 0.05 # distance collision avoidance kicks in

speed = 1.0 # speed of movement

m = Boids(N, range, vision, exclusion, speed)

no.log("q to quit")
no.run(m)
