import neworder as no

from boids3d import Boids3d as Boids

N = 1000 # number of boids
range = 1.0 # extent of the domain
vision = 0.3 # distance boids "see"
exclusion = 0.05 # distance collision avoidance kicks in
speed = 1.0

m = Boids(N, range, vision, exclusion, speed)

no.log("q to quit")
no.run(m)
