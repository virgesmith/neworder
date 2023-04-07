import neworder as no

from boids2d import Boids2d as Boids

# perfomance can be improved by **reducing** the number of threads numpy uses
# e.g. set OPENBLAS_NUM_THREADS=2


N = 500 # number of boids
range = 1.0 # extent of the domain
vision = 0.2 # distance boids "see"
exclusion = 0.05 # distance collision avoidance kicks in
speed = 1.0

m = Boids(N, range, vision, exclusion, speed)

no.log("q to quit")
no.run(m)
